"""Some utils for graph creation."""
import logging

import numpy as np
import pandas as pd
from neurom.morphmath import angle_between_vectors
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from scipy.spatial import Voronoi
from voxcell.nexus.voxelbrain import Atlas

logger = logging.getLogger(__name__)


def get_fiber_tracts(atlas_path, atlas_hierarchy_filename, atlas_region_filename):
    """Extract fiber tracts points from an atlas."""
    atlas = Atlas.open(atlas_path)
    region_map = atlas.load_region_map(atlas_hierarchy_filename)
    brain_regions = atlas.load_data(atlas_region_filename)
    fiber_tracts_ids = region_map.find("fiber tracts", attr="name", with_descendants=True)
    fiber_tracts_mask = np.isin(brain_regions.raw, list(fiber_tracts_ids))
    brain_regions.raw[~fiber_tracts_mask] = 0  # Zeroes the complement region
    fiber_tract_points = brain_regions.indices_to_positions(np.argwhere(brain_regions.raw))
    return fiber_tract_points


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
                    ]
                ).T
                + ref_coords,
            )
        )
    return inter_pts


def add_random_points(all_pts, min_random_point_distance, seed):
    """Add random points in the bounding box of the given points."""
    if min_random_point_distance is not None:
        n_fails = 0
        bbox = np.vstack([all_pts.min(axis=0), all_pts.max(axis=0)])
        rng = np.random.default_rng(seed)
        tree = KDTree(all_pts)
        new_pts = []
        while n_fails < 10:
            xyz = np.array(
                [
                    rng.uniform(bbox[0, 0], bbox[1, 0]),
                    rng.uniform(bbox[0, 1], bbox[1, 1]),
                    rng.uniform(bbox[0, 2], bbox[1, 2]),
                ]
            )
            if np.isinf(
                tree.query(xyz, distance_upper_bound=min_random_point_distance, k=2,)[
                    0
                ][1]
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


def add_voronoi_points(all_pts, voronoi_steps):
    """Add Voronoi points between the given points."""
    for i in range(voronoi_steps):
        vor = Voronoi(all_pts, qhull_options="QJ")
        all_pts = np.concatenate([all_pts, vor.vertices])  # pylint: disable=no-member
    return all_pts


def drop_close_points(all_points_df, duplicate_precision):
    """Drop points that are closer to a given distance."""
    tree = KDTree(all_points_df[["x", "y", "z"]])
    close_pts = tree.query_pairs(duplicate_precision)
    for a, b in close_pts:
        if a in all_points_df.index and b in all_points_df.index:
            all_points_df.drop(a, inplace=True)


def drop_outside_points(all_points_df, ref_pts):
    """Remove points outside the bounding box of reference points."""
    min_pts = ref_pts.min(axis=0)
    max_pts = ref_pts.max(axis=0)
    outside_pts = all_points_df.loc[
        ((all_points_df[["x", "y", "z"]] < min_pts).any(axis=1))
        | ((all_points_df[["x", "y", "z"]] > max_pts).any(axis=1))
    ]
    all_points_df.drop(outside_pts.index, inplace=True)


def create_edges(all_points, from_coord_cols, to_coord_cols, group_name):
    """Create edges from the Delaunay triangulation of the given points."""
    tri = Delaunay(all_points, qhull_options="QJ")

    # Find all unique edges from the triangulation
    unique_edges = np.unique(
        np.apply_along_axis(
            np.sort,
            1,
            np.vstack(
                # pylint: disable=no-member
                np.stack((tri.simplices, np.roll(tri.simplices, -1, axis=1)), axis=2)
            ),
        ),
        axis=0,
    )

    edges_df = pd.DataFrame(
        {
            "morph_file": group_name,
            "from": unique_edges[:, 0],
            "to": unique_edges[:, 1],
        }
    )

    # Add coordinates and compute lengths
    edges_df[from_coord_cols] = all_points.loc[edges_df["from"]].values
    edges_df[to_coord_cols] = all_points.loc[edges_df["to"]].values
    edges_df["length"] = np.linalg.norm(
        edges_df[from_coord_cols].values - edges_df[to_coord_cols].values,
        axis=1,
    )

    return edges_df, tri


def add_terminal_penalty(edges_df, terminal_edges, penalty):
    """Add penalty to terminals to ensure the Steiner algorithm don't connect terminals directly."""
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
        "length",
    ] += penalty


def add_orientation_penalty(
    edges_df,
    from_coord_cols,
    to_coord_cols,
    orientation_penalty,
    orientation_penalty_exponent,
    soma_center_coords,
):
    """Add penalty to terminals according to their orientation."""
    if orientation_penalty:
        vectors = edges_df[to_coord_cols].values - edges_df[from_coord_cols].values
        origin_to_mid_vectors = (
            0.5 * (edges_df[to_coord_cols].values + edges_df[from_coord_cols].values)
            - soma_center_coords
        )
        data = np.stack([origin_to_mid_vectors, vectors], axis=1)

        edge_angles = np.array([angle_between_vectors(i[0], i[1]) for i in data.tolist()])
        orientation_penalty = np.power(
            np.clip(np.sin(edge_angles), 1e-3, 1 - 1e-3),
            orientation_penalty_exponent,
        )
        edges_df["length"] *= orientation_penalty
