"""Some utils for graph creation."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import logging

import numpy as np
import pandas as pd
from neurom.morphmath import angle_between_vectors
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module
from scipy.spatial import KDTree
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import NodeProvider
from axon_synthesis.typing import SeedType

FORCE_2D = False
"""Force the Voronoï and Delaunay calculations to ignore the Z coordinate."""


def add_intermediate_points(
    pts: np.ndarray, ref_coords, min_intermediate_distance, intermediate_number
):
    """Add intermediate points between source points and target points."""
    terms = pts[:, :3] - ref_coords
    term_dists = np.linalg.norm(terms, axis=1)
    if min_intermediate_distance <= 0:
        nb_inter = np.repeat(intermediate_number, len(terms))
    else:
        nb_inter = np.clip(
            term_dists // min_intermediate_distance,
            0,
            intermediate_number,
        )

    if (nb_inter <= 0).all():
        return pts

    inter_pts = []
    for x, y, z, num in np.hstack([terms, np.atleast_2d(nb_inter).T]):
        if num == 0 or np.linalg.norm([x, y, z]) == 0:
            continue
        inter_pts.append(
            (
                num,
                np.hstack(
                    (
                        np.array(
                            [
                                np.linspace(0, x, int(num) + 2)[1:-1],
                                np.linspace(0, y, int(num) + 2)[1:-1],
                                np.linspace(0, z, int(num) + 2)[1:-1],
                            ],
                        ).T
                        + ref_coords,
                        np.ones((int(num), 1)) * NodeProvider.intermediate,
                    )
                ),
            ),
        )
    return np.concatenate([pts, *[i[1] for i in inter_pts]])


def generate_random_points(
    bbox: np.ndarray,
    min_random_point_distance: float,
    rng: SeedType,
    *,
    existing_pts_tree: KDTree | None = None,
    max_tries: int = 10,
):
    """Generate random points in a bounding box and with a min distance."""
    n_fails = 0
    rng = np.random.default_rng(rng)
    new_pts: list[np.ndarray] = []
    while n_fails < max_tries:
        xyz = np.array(
            [
                rng.uniform(bbox[0, 0], bbox[1, 0]),
                rng.uniform(bbox[0, 1], bbox[1, 1]),
                rng.uniform(bbox[0, 2], bbox[1, 2]),
            ],
        )
        if (
            existing_pts_tree is None
            or np.isinf(
                existing_pts_tree.query(
                    xyz,
                    distance_upper_bound=min_random_point_distance,
                    k=2,
                )[0][1],
            )
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
    return new_pts


def add_random_points(
    all_pts: np.ndarray,
    min_random_point_distance: float | None,
    bbox_buffer: float,
    rng: SeedType,
    *,
    max_tries: int = 10,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Add random points in the bounding box of the given points."""
    if min_random_point_distance is not None:
        all_xyz = all_pts[:, :3]
        bbox = np.vstack([all_xyz.min(axis=0), all_xyz.max(axis=0)])
        bbox[0] -= bbox_buffer
        bbox[1] += bbox_buffer
        tree = KDTree(all_xyz)
        new_pts = generate_random_points(
            bbox,
            min_random_point_distance,
            rng=rng,
            existing_pts_tree=tree,
            max_tries=max_tries,
        )

        if new_pts:
            if logger is not None:
                logger.debug("Random points added: %s", len(new_pts))
            all_pts = np.concatenate(
                [
                    all_pts,
                    np.hstack(
                        (np.array(new_pts), np.ones((len(new_pts), 1)) * NodeProvider.random)
                    ),
                ]
            )
        elif logger is not None:
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


def add_bounding_box_pts(all_pts: np.ndarray):
    """Add points of the bbox."""
    all_xyz = all_pts[:, :3]
    bbox = np.array([all_xyz.min(axis=0), all_xyz.max(axis=0)])
    bbox_pts = np.hstack(
        (
            np.array(np.meshgrid(*np.array(bbox).T)).T.reshape((8, 3)),
            np.ones((8, 1)) * NodeProvider.bbox,
        )
    )
    new_all_pts = np.concatenate([all_pts, bbox_pts])
    return new_all_pts[np.sort(np.unique(new_all_pts, axis=0, return_index=True)[1])]


def add_voronoi_points(
    all_pts: np.ndarray,
    voronoi_steps: int,
    initial_bbox: np.ndarray | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Add Voronoi points between the given points."""
    if len(all_pts) < 5:
        return all_pts
    for _ in range(voronoi_steps):
        if FORCE_2D:
            vor = Voronoi(all_pts[:, [0, 1]], qhull_options="QJ")
            step_pts = np.hstack([vor.vertices, np.zeros((len(vor.vertices), 1))])
        else:
            vor = Voronoi(all_pts[:, :3], qhull_options="QJ")
            step_pts = vor.vertices
        if initial_bbox is not None:
            step_pts = step_pts[
                np.all((step_pts >= initial_bbox[0]) & (step_pts <= initial_bbox[1]), axis=1)
            ]
        new_pts = np.hstack([step_pts, np.ones((len(step_pts), 1)) * NodeProvider.Voronoi])
        all_pts = np.concatenate([all_pts, new_pts])  # pylint: disable=no-member
    if logger is not None:
        logger.debug("Added %s Voronoï points using %s steps", len(new_pts), voronoi_steps)
    return all_pts


def drop_close_points(
    all_points_df: pd.DataFrame,
    duplicate_precision: float,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Drop points that are closer to a given distance."""
    tree = KDTree(all_points_df[COORDS_COLS])
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

    if logger is not None:
        logger.debug(
            "Dropped %s close points using %s precision", len(to_drop), duplicate_precision
        )
    return all_points_df.drop(list(to_drop))


def drop_outside_points(all_points_df, ref_pts=None, bbox=None):
    """Remove points outside the bounding box of reference points or brain regions."""
    if bbox is not None:
        outside_pts = all_points_df.loc[
            ((all_points_df[COORDS_COLS] < bbox[0]).any(axis=1))
            | ((all_points_df[COORDS_COLS] > bbox[1]).any(axis=1))
        ]
        all_points_df = all_points_df.drop(outside_pts.index)

    if ref_pts is not None:
        ref_xyz = ref_pts[:, :3]
        min_pts = ref_xyz.min(axis=0)
        max_pts = ref_xyz.max(axis=0)
        diff = (max_pts - min_pts) * 0.1
        min_pts -= diff
        max_pts += diff
        outside_pts = all_points_df.loc[
            ((all_points_df[COORDS_COLS] < min_pts).any(axis=1))
            | ((all_points_df[COORDS_COLS] > max_pts).any(axis=1))
        ]
        all_points_df = all_points_df.drop(outside_pts.index)

    return all_points_df


def create_edges(all_points, from_coord_cols, to_coord_cols):
    """Create undirected edges from the Delaunay triangulation of the given points.

    .. note::
        The source-target order has no meaning.
    """
    if len(all_points) < 5:
        msg = (
            "Not enough points to create edges, at least 5 points are needed but only "
            f"{len(all_points)} found"
        )
        raise RuntimeError(msg)

    if FORCE_2D:
        tri = Delaunay(all_points[["x", "y"]], qhull_options="QJ")
    else:
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
    edges_df["length"] = np.linalg.norm(
        edges_df[from_coord_cols].to_numpy() - edges_df[to_coord_cols].to_numpy(),
        axis=1,
    )

    return edges_df, tri


def add_terminal_penalty(edges, nodes):
    """Add penalty to edges to ensure the Steiner algorithm don't connect terminals directly."""
    # Compute penalty
    penalty = edges["weight"].max() + edges["weight"].mean()

    # Get terminal edges
    terminal_edges = edges[["from", "to"]].isin(
        nodes.loc[nodes["is_terminal"], "id"].to_numpy(),
    )

    # Add the penalty
    edges_are_terminals = edges.join(terminal_edges, rsuffix="_is_terminal")
    from_to_all_terminals = edges_are_terminals.groupby("from")[
        ["from_is_terminal", "to_is_terminal"]
    ].all()

    edges_are_terminals = edges_are_terminals.join(
        from_to_all_terminals["from_is_terminal"].rename("from_all_terminals"),
        on="from",
    )
    edges_are_terminals = edges_are_terminals.join(
        from_to_all_terminals["to_is_terminal"].rename("to_all_terminals"),
        on="to",
    )
    edges.loc[
        (edges_are_terminals[["from_is_terminal", "to_is_terminal"]].all(axis=1))
        & (~edges_are_terminals[["from_all_terminals", "to_all_terminals"]].all(axis=1)),
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
    # atlas_orientations = atlas.load_data("orientation", cls=OrientationField)
    # from_orientations = atlas_orientations.lookup(edges_df[from_coord_cols].to_numpy())
    # to_orientations = atlas_orientations.lookup(edges_df[to_coord_cols].to_numpy())

    # # Compare the two rotation matrices
    # transposed_orientations = np.transpose(from_orientations, axes=[0, 2, 1])
    # dot_prod = np.einsum("...ij,...jk->...ik", transposed_orientations, to_orientations)

    # # The trace of the dot product is equal to the cosine of the angle between the two matrices
    # # and we want to take the cosine of the absolute value of the angle, so we can simplify.
    # penalty = np.abs((np.trace(dot_prod, axis1=1, axis2=2) - 1) * 0.5)

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
    depths,
    sigma,
    amplitude,
):
    """Add penalty to edges according to their depth in the Atlas."""
    from_depths = np.nan_to_num(depths.lookup(edges_df[from_coord_cols].to_numpy()))
    to_depths = np.nan_to_num(depths.lookup(edges_df[to_coord_cols].to_numpy()))

    relative_delta = np.clip(np.abs(from_depths - to_depths) / (edges_df["length"]), 0, 1)

    return (1 + amplitude * (1 - np.exp(-relative_delta / sigma))).to_numpy()


def add_preferred_reward(
    edges_df,
    from_coord_cols,
    to_coord_cols,
    preferred_region_tree,
    sigma,
    amplitude,
):
    """Add rewards to edges depending on their distance to the preferred points."""
    from_distances, _ = preferred_region_tree.query(edges_df[from_coord_cols].to_numpy())
    to_distances, _ = preferred_region_tree.query(edges_df[to_coord_cols].to_numpy())

    # TODO: For now we just take the mean of the distance between the start point to the closest
    # preferred point and between the end point to the closest preferred point, which is not
    # accurate.
    return 1 + amplitude * (1 - np.exp(-0.5 * (from_distances + to_distances) / sigma))
