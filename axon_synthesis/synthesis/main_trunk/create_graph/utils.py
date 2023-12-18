"""Some utils for graph creation."""
import logging

import numpy as np
import pandas as pd
from neurom.morphmath import angle_between_vectors
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from scipy.spatial import Voronoi

logger = logging.getLogger(__name__)


def use_ancestors(terminals, tuft_properties_path):
    """Use ancestor coords instead of the center of the cluster."""
    cluster_props_df = pd.read_json(tuft_properties_path)
    tmp = pd.merge(
        terminals,
        cluster_props_df,
        left_on=["morphology", "axon_id", "terminal_id"],
        right_on=["morphology", "axon_id", "cluster_id"],
        how="left",
    )
    mask = ~tmp["cluster_id"].isna()
    new_terminal_coords = pd.DataFrame(
        tmp.loc[mask, "common_ancestor_coords"].to_list(),
        columns=["x", "y", "z"],
    )
    tmp.loc[mask, ["x", "y", "z"]] = new_terminal_coords.to_numpy()
    terminals[["x", "y", "z"]] = tmp[["x", "y", "z"]]


def add_intermediate_points(pts, ref_coords, min_intermediate_distance, intermediate_number):
    """Add intermediate points between each pair of points."""
    terms = pts - ref_coords
    term_dists = np.linalg.norm(terms, axis=1)
    nb_inter = np.clip(
        term_dists // min_intermediate_distance,
        0,
        intermediate_number,
    )

    inter_pts = []
    for x, y, z, num in np.hstack([terms, np.atleast_2d(nb_inter).T]):
        inter_pts.append(
            (
                num,
                np.array(
                    [
                        np.linspace(0, x, int(num) + 2)[1:-1],
                        np.linspace(0, y, int(num) + 2)[1:-1],
                        np.linspace(0, z, int(num) + 2)[1:-1],
                    ],
                ).T
                + ref_coords,
            ),
        )
    return inter_pts


def add_random_points(all_pts, min_random_point_distance, bbox_buffer, rng, *, max_tries: int = 10):
    """Add random points in the bounding box of the given points."""
    if min_random_point_distance is not None:
        n_fails = 0
        bbox = np.vstack([all_pts.min(axis=0), all_pts.max(axis=0)])
        bbox[0] -= bbox_buffer
        bbox[1] += bbox_buffer
        tree = KDTree(all_pts)
        new_pts = []
        while n_fails < max_tries:
            xyz = np.array(
                [
                    rng.uniform(bbox[0, 0], bbox[1, 0]),
                    rng.uniform(bbox[0, 1], bbox[1, 1]),
                    rng.uniform(bbox[0, 2], bbox[1, 2]),
                ],
            )
            if np.isinf(
                tree.query(
                    xyz,
                    distance_upper_bound=min_random_point_distance,
                    k=2,
                )[0][1],
            ) and (
                len(new_pts) == 0
                or np.linalg.norm(
                    xyz - new_pts,
                    axis=1,
                ).min()
                > min_random_point_distance
            ):
                new_pts.append(xyz)
                n_fails = 0
            else:
                n_fails += 1

        logger.info("Random points added: %s", len(new_pts))
        if new_pts:
            all_pts = np.concatenate([all_pts, np.array(new_pts)])
        else:
            logger.warning(
                (
                    "Could not add random points! The current state is the following: "
                    "bbox=%s ; nb_pts=%s ; min distance=%s"
                ),
                bbox,
                len(all_pts),
                min_random_point_distance,
            )
    return all_pts


def add_bounding_box_pts(all_pts):
    """Add points of the bbox."""
    bbox = np.array([all_pts.min(axis=0), all_pts.max(axis=0)])
    bbox_pts = np.array(np.meshgrid(*np.array(bbox).T)).T.reshape((8, 3))
    new_all_pts = np.concatenate([all_pts, bbox_pts])
    return new_all_pts[np.sort(np.unique(new_all_pts, axis=0, return_index=True)[1])]


def add_voronoi_points(all_pts, voronoi_steps):
    """Add Voronoi points between the given points."""
    if len(all_pts) < 5:  # noqa: PLR2004
        return all_pts
    for _ in range(voronoi_steps):
        vor = Voronoi(all_pts, qhull_options="QJ")
        all_pts = np.concatenate([all_pts, vor.vertices])  # pylint: disable=no-member
    return all_pts


def drop_close_points(all_points_df, duplicate_precision):
    """Drop points that are closer to a given distance."""
    tree = KDTree(all_points_df[["x", "y", "z"]])
    close_pts = tree.query_pairs(duplicate_precision)

    if not close_pts:
        return all_points_df

    to_drop = set()
    for a, b in close_pts:
        label_a = all_points_df.index[a]
        label_b = all_points_df.index[b]
        if label_a not in to_drop and label_b not in to_drop:
            if all_points_df.loc[label_a, "is_terminal"]:
                to_drop.add(label_b)
            else:
                to_drop.add(label_a)

    return all_points_df.drop(list(to_drop))


def drop_outside_points(all_points_df, ref_pts=None, bbox=None):
    """Remove points outside the bounding box of reference points or brain regions."""
    if bbox is not None:
        outside_pts = all_points_df.loc[
            ((all_points_df[["x", "y", "z"]] < bbox[0]).any(axis=1))
            | ((all_points_df[["x", "y", "z"]] > bbox[1]).any(axis=1))
        ]
        all_points_df = all_points_df.drop(outside_pts.index)

    if ref_pts is not None:
        min_pts = ref_pts.min(axis=0)
        max_pts = ref_pts.max(axis=0)
        outside_pts = all_points_df.loc[
            ((all_points_df[["x", "y", "z"]] < min_pts).any(axis=1))
            | ((all_points_df[["x", "y", "z"]] > max_pts).any(axis=1))
        ]
        all_points_df = all_points_df.drop(outside_pts.index)

    return all_points_df


def create_edges(all_points, from_coord_cols, to_coord_cols):
    """Create undirected edges from the Delaunay triangulation of the given points.

    .. note::
        The source-target order has no meaning.
    """
    if len(all_points) < 5:  # noqa: PLR2004
        msg = ""
        raise RuntimeError(msg)
    tri = Delaunay(all_points, qhull_options="QJ")

    # Find all unique edges from the triangulation
    unique_edges = np.unique(
        np.apply_along_axis(
            np.sort,
            1,
            np.vstack(
                # pylint: disable=no-member
                np.stack((tri.simplices, np.roll(tri.simplices, -1, axis=1)), axis=2),
            ),
        ),
        axis=0,
    )

    edges_df = pd.DataFrame(
        {
            "from": unique_edges[:, 0],
            "to": unique_edges[:, 1],
        },
    )

    # Add coordinates and compute base weights equal to the lengths
    edges_df[from_coord_cols] = all_points.loc[edges_df["from"]].to_numpy()
    edges_df[to_coord_cols] = all_points.loc[edges_df["to"]].to_numpy()
    edges_df["weight"] = np.linalg.norm(
        edges_df[from_coord_cols].to_numpy() - edges_df[to_coord_cols].to_numpy(),
        axis=1,
    )

    return edges_df, tri


def add_terminal_penalty(edges_df, all_points_df):
    """Add penalty to edges to ensure the Steiner algorithm don't connect terminals directly."""
    # Compute penalty
    penalty = edges_df["weight"].max() + edges_df["weight"].mean()

    # Get terminal edges
    terminal_edges = edges_df[["from", "to"]].isin(
        all_points_df.loc[all_points_df["is_terminal"], "id"].to_numpy(),
    )

    # Add the penalty
    edges_df_terminals = edges_df.join(terminal_edges, rsuffix="_is_terminal")
    from_to_all_terminals = edges_df_terminals.groupby("from")[
        ["from_is_terminal", "to_is_terminal"]
    ].all()

    edges_df_terminals = edges_df_terminals.join(
        from_to_all_terminals["from_is_terminal"].rename("from_all_terminals"),
        on="from",
    )
    edges_df_terminals = edges_df_terminals.join(
        from_to_all_terminals["to_is_terminal"].rename("to_all_terminals"),
        on="to",
    )
    edges_df.loc[
        (edges_df_terminals[["from_is_terminal", "to_is_terminal"]].all(axis=1))
        & (~edges_df_terminals[["from_all_terminals", "to_all_terminals"]].all(axis=1)),
        "weight",
    ] += penalty


def add_orientation_penalty(
    edges_df,
    from_coord_cols,
    to_coord_cols,
    soma_center_coords,
    orientation_penalty_exponent,
    amplitude,
):
    """Add penalty to edges according to their orientation."""
    vectors = edges_df[to_coord_cols].to_numpy() - edges_df[from_coord_cols].to_numpy()
    origin_to_mid_vectors = (
        0.5 * (edges_df[to_coord_cols].to_numpy() + edges_df[from_coord_cols].to_numpy())
        - soma_center_coords
    )
    data = np.stack([origin_to_mid_vectors, vectors], axis=1)

    edge_angles = np.array([angle_between_vectors(i[0], i[1]) for i in data.tolist()])
    return 1 + amplitude * np.power(
        np.clip(np.sin(edge_angles), 1e-3, 1 - 1e-3),
        orientation_penalty_exponent,
    )


def add_depth_penalty(
    edges_df,
    from_coord_cols,
    to_coord_cols,
    atlas,
    sigma,
    amplitude,
):
    """Add penalty to edges according to the difference of orientation at start and end points."""
    # atlas_orientations = atlas.load_data("orientation", cls=OrientationField)
    # from_orientations = atlas_orientations.lookup(edges_df[from_coord_cols].to_numpy())
    # to_orientations = atlas_orientations.lookup(edges_df[to_coord_cols].to_numpy())

    # # Compare the two rotation matrices
    # transposed_orientations = np.transpose(from_orientations, axes=[0, 2, 1])
    # dot_prod = np.einsum("...ij,...jk->...ik", transposed_orientations, to_orientations)

    # # The trace of the dot product is equal to the cosine of the angle between the two matrices
    # # and we want to take the cosine of the absolute value of the angle, so we can simplify.
    # penalty = np.abs((np.trace(dot_prod, axis1=1, axis2=2) - 1) * 0.5)

    from_depths = np.nan_to_num(atlas.depths.lookup(edges_df[from_coord_cols].to_numpy()))
    to_depths = np.nan_to_num(atlas.depths.lookup(edges_df[to_coord_cols].to_numpy()))

    relative_delta = np.clip(np.abs(from_depths - to_depths) / (edges_df["weight"]), 0, 1)

    return 1 + amplitude * (1 - np.exp(-relative_delta / sigma))


def add_favored_reward(
    edges_df,
    from_coord_cols,
    to_coord_cols,
    favored_region_tree,
    sigma,
    amplitude,
):
    """Add rewards to edges depending on their distance to the favored points."""
    from_distances, _ = favored_region_tree.query(edges_df[from_coord_cols].to_numpy())
    to_distances, _ = favored_region_tree.query(edges_df[to_coord_cols].to_numpy())

    # TODO: For now we just take the mean of the distance between the start point to the closest
    # favored point and between the end point to the closest favored point, which is not accurate.
    return 1 + amplitude * (1 - np.exp(-0.5 * (from_distances + to_distances) / sigma))