"""Create the edges between the terminals and the obstacles (if any).

This is needed to easily compute a Steiner Tree (Euclidean Steiner Tree is complicated).
"""

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
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import define
from attrs import field
from attrs import validators
from scipy.spatial import KDTree
from voxcell import VoxelData

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TARGET_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.constants import NodeProvider
from axon_synthesis.synthesis.main_trunk.create_graph.plot import plot_triangulation
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_bounding_box_pts
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_depth_penalty
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_intermediate_points
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_orientation_penalty
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_preferred_reward
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_random_points
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_terminal_penalty
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_voronoi_points
from axon_synthesis.synthesis.main_trunk.create_graph.utils import create_edges
from axon_synthesis.synthesis.main_trunk.create_graph.utils import drop_close_points
from axon_synthesis.synthesis.main_trunk.create_graph.utils import drop_outside_points
from axon_synthesis.typing import FileType
from axon_synthesis.typing import RegionIdsType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import compute_bbox
from axon_synthesis.utils import sublogger


@define
class CreateGraphConfig:
    """Class to store the parameters needed for graph creation.

    Attributes:
        intermediate_number: The number of intermediate points added before Voronoï process.
        min_intermediate_distance: The min distance between two successive intermediate points.
        min_random_point_distance: The min distance used to add random points.
        random_point_bbox_buffer: The distance used to add a buffer around the bbox of the points.
        voronoi_steps: The number of Voronoi steps.
        duplicate_precision: The precision used to detect duplicated points.
        use_orientation_penalty: If set to True, a penalty is added to edges whose direction is not
            radial.
        orientation_penalty_exponent: The exponent used for the orientation penalty.
        orientation_penalty_amplitude: The amplitude of the orientation penalty.
        use_depth_penalty: If set to True, a penalty is added to edges whose direction is not
            parallel to the iso-depth curves.
        depth_penalty_sigma: The sigma used for depth penalty.
        depth_penalty_amplitude: The amplitude of the depth penalty.
        preferred_regions: The list of brain regions in which edge weights are divided by the
            preferring factor.
        preferred_region_min_random_point_distance: The min distance used to pick random points in
            preferred regions.
        preferring_sigma: The sigma used to compute the preferring factor for the given regions.
        preferring_amplitude: The amplitude used to compute the preferring factor for the given
            regions.
        preferred_region_tree: The KDTree object containing the preferred region points.
        use_terminal_penalty: If set to True, a penalty is added to edges that are connected to a
            terminal.
    """

    # Intermediate points
    intermediate_number: int = field(default=5, validator=validators.ge(0))
    min_intermediate_distance: float = field(default=1000, validator=validators.gt(0))

    # Random points
    min_random_point_distance: float | None = field(
        default=None, validator=validators.optional(validators.ge(0))
    )
    random_point_bbox_buffer: float = field(default=0, validator=validators.ge(0))

    # Voronoï points
    voronoi_steps: int = field(default=1, validator=validators.ge(0))

    # Duplicated points
    duplicate_precision: float = field(default=1e-3, validator=validators.gt(0))

    # Orientation penalty
    use_orientation_penalty: bool = field(default=True)
    orientation_penalty_exponent: float = field(default=0.1, validator=validators.ge(0))
    orientation_penalty_amplitude: float = field(default=1, validator=validators.gt(0))

    # Depth penalty
    use_depth_penalty: bool = field(default=True)
    depth_penalty_sigma: float = field(default=0.25, validator=validators.gt(0))
    depth_penalty_amplitude: float = field(default=10, validator=validators.gt(0))

    # preferred regions
    preferred_regions: RegionIdsType | None = field(
        default=None,
    )
    preferred_region_min_random_point_distance: float | None = field(
        default=None, validator=validators.optional(validators.ge(0))
    )
    preferring_sigma: float = field(default=100, validator=validators.gt(0))
    preferring_amplitude: float = field(default=1, validator=validators.gt(0))
    preferred_region_tree: KDTree | None = field(default=None)

    # Terminal penalty
    use_terminal_penalty: bool = field(default=False)

    def __attrs_post_init__(self):
        """Check consistency of new instances."""
        if self.preferred_regions is not None and self.preferred_region_min_distance is None:
            msg = (
                "A preferred region list was given without any related distance, please provide "
                "at least one of the following parameters: "
                "'preferred_region_min_random_point_distance', 'min_random_point_distance'",
            )
            raise ValueError(msg)

    @property
    def preferred_region_min_distance(self):
        """Get the min distance used to generate random points in the preferred regions."""
        return (
            self.preferred_region_min_random_point_distance
            if self.preferred_region_min_random_point_distance is not None
            else self.min_random_point_distance
        )

    def pick_preferred_region_random_points(self, rng=None, max_tries: int = 10):
        """Create random points in preferred regions with min distance between them."""
        if self.preferred_region_tree is None:
            msg = (
                "The 'preferred_region_tree' must be computed before picking random points inside "
                "the preferred region"
            )
            raise RuntimeError(msg)
        rng = np.random.default_rng(rng)
        n_fails = 0
        new_pts: list[np.ndarray] = []
        min_dist = self.preferred_region_min_distance
        while n_fails < max_tries:
            xyz = rng.choice(self.preferred_region_tree.data)
            if (
                len(new_pts) == 0
                or np.linalg.norm(
                    xyz - new_pts,
                    axis=1,
                ).min()
                > min_dist
            ):
                new_pts.append(xyz)
                n_fails = 0
            else:
                n_fails += 1

        if len(new_pts) == 0:
            # Ensure at least one point is added (the closest point from the barycenter of the
            # given points)
            new_pts.append(
                self.preferred_region_tree.data[
                    np.argmin(
                        np.linalg.norm(
                            self.preferred_region_tree.data
                            - self.preferred_region_tree.data.mean(axis=0),
                            axis=1,
                        )
                    )
                ]
            )
        return np.hstack([new_pts, np.ones((len(new_pts), 1)) * NodeProvider.random])

    def compute_region_tree(self, atlas: AtlasHelper, *, force: bool = False):
        """Compute the preferred region tree using the given Atlas."""
        if self.preferred_regions and (self.preferred_region_tree is None or force):
            preferred_region_points, _missing_ids = atlas.get_region_points(self.preferred_regions)
            self.preferred_region_tree = KDTree(preferred_region_points)
        else:
            self.preferred_region_tree = None

    def region_tree_from_file(self, file):
        """Get the preferred region tree from a file."""
        data = np.load(file, allow_pickle=False)
        self.preferred_region_tree = KDTree(data)


def _add_points(source_coords, pts, config, depths, forbidden_regions, rng, logger) -> tuple:
    """Add all points to the source and target points."""
    # Add intermediate points
    all_pts = add_intermediate_points(
        pts,
        source_coords,
        config.min_intermediate_distance,
        config.intermediate_number,
    )

    # Add random points from other regions
    all_pts = add_random_points(
        all_pts,
        config.min_random_point_distance,
        config.random_point_bbox_buffer,
        rng,
        logger=logger,
    )

    # Add random points from the preferred regions
    if config.preferred_region_tree is not None:
        preferred_region_pts = config.pick_preferred_region_random_points(rng=rng)
        logger.debug("Random points added in the preferred regions: %s", len(preferred_region_pts))
        all_pts = np.concatenate(
            [
                all_pts,
                preferred_region_pts,
            ]
        )

    # Add the bounding box points to ensure a minimum number of points
    all_pts = add_bounding_box_pts(all_pts)

    # Get bbox of points before adding Voronoi points that can go very far
    current_bbox = compute_bbox(all_pts[:, :3], 0.1)
    if depths is not None:
        # Clip the bbox the keep it inside the atlas
        current_bbox[0] = np.clip(current_bbox[0], a_min=depths.bbox[0], a_max=np.inf)
        current_bbox[1] = np.clip(current_bbox[1], a_min=-np.inf, a_max=depths.bbox[1])

    # Add Voronoï points
    all_pts = add_voronoi_points(
        all_pts, config.voronoi_steps, initial_bbox=current_bbox, logger=logger
    )

    if forbidden_regions is not None:
        forbidden_points = forbidden_regions.lookup(all_pts[:, :3])
        all_pts = all_pts[~forbidden_points]

    return all_pts, current_bbox


def one_graph(
    source_coords: np.ndarray,
    target_points: pd.DataFrame,
    config: CreateGraphConfig,
    depths: VoxelData | None = None,
    *,
    forbidden_regions: VoxelData | None = None,
    output_path: FileType | None = None,
    figure_path: FileType | None = None,
    rng: SeedType = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Create the nodes and edges for one axon based on the target points and the atlas."""
    logger = sublogger(logger, __name__)

    rng = np.random.default_rng(rng)

    unique_target_points = target_points.drop_duplicates(subset=["axon_id", "terminal_id"])

    logger.debug("%s points", len(unique_target_points))

    # Terminal points
    pts = np.concatenate([[source_coords], unique_target_points[TARGET_COORDS_COLS].to_numpy()])
    pts = np.hstack(
        (
            pts,
            np.atleast_2d(
                [NodeProvider.source] + [NodeProvider.target] * len(unique_target_points)
            ).T,
        )
    )

    # Add other points
    all_pts, current_bbox = _add_points(
        source_coords, pts, config, depths, forbidden_regions, rng, logger
    )

    # Gather points
    nodes_df = pd.DataFrame(all_pts, columns=[*COORDS_COLS, "NodeProvider"])

    # Mark the source and target points as terminals and the others as intermediates
    nodes_df["is_terminal"] = [True] * len(pts) + [False] * (len(all_pts) - len(pts))

    # Associate the terminal IDs to the nodes
    nodes_df["terminal_id"] = (
        [-1]
        + unique_target_points["terminal_id"].to_numpy().tolist()
        + [-1] * (len(all_pts) - len(pts))
    )

    # Remove close points
    nodes_df = drop_close_points(nodes_df, config.duplicate_precision)

    # Remove outside points
    nodes_df = drop_outside_points(
        nodes_df,
        bbox=current_bbox,
    )

    # Add some intermediate points to ensure there are enough points to build a graph
    if len(nodes_df) <= 5:
        all_pts = add_intermediate_points(
            nodes_df[[*COORDS_COLS, "NodeProvider"]].to_numpy(),
            source_coords,
            0,
            5 - len(nodes_df),
        )
        filling_pts = pd.DataFrame(all_pts[len(nodes_df) :], columns=[*COORDS_COLS, "NodeProvider"])
        filling_pts[["is_terminal", "terminal_id"]] = (False, -1)
        nodes_df = pd.concat([nodes_df, filling_pts], ignore_index=True)

    # Reset index and set IDs
    nodes_df = nodes_df.reset_index(drop=True)
    nodes_df.loc[:, "id"] = nodes_df.index.to_numpy()

    # Create edges using the Delaunay triangulation of the union of the terminals,
    # intermediate and Voronoï points
    edges_df, _ = create_edges(
        nodes_df.loc[:, COORDS_COLS],
        FROM_COORDS_COLS,
        TO_COORDS_COLS,
    )

    logger.debug("%s edges created", len(edges_df))

    # Compute cumulative penalties
    penalties = np.ones(len(edges_df))

    # Increase the weight of edges whose angle with radial direction is close to pi/2
    if config.use_orientation_penalty:
        logger.debug("Add orientation penalties to edge weights")
        penalties *= add_orientation_penalty(
            edges_df,
            FROM_COORDS_COLS,
            TO_COORDS_COLS,
            source_coords,
            config.orientation_penalty_exponent,
            config.orientation_penalty_amplitude,
        )

    # Increase the weight of edges which do not follow an iso-depth curve
    if config.use_depth_penalty and depths is not None:
        logger.debug("Add depth penalties to edge weights")
        penalties *= add_depth_penalty(
            edges_df,
            FROM_COORDS_COLS,
            TO_COORDS_COLS,
            depths,
            config.depth_penalty_sigma,
            config.depth_penalty_amplitude,
        )

    # Reduce the lengths of edges that are close to the preferred regions
    if config.preferred_region_tree is not None:
        logger.debug("Add preferred region rewards to edge weights")
        penalties *= add_preferred_reward(
            edges_df,
            FROM_COORDS_COLS,
            TO_COORDS_COLS,
            config.preferred_region_tree,
            config.preferring_sigma,
            config.preferring_amplitude,
        )

    # TODO: increase weights of more impossible edges?

    # Apply cumulative penalties
    edges_df["weight"] = edges_df["length"] * penalties

    # TODO: Remove points and edges from forbidden regions?

    # Add penalty to edges between two terminals (except if a terminal is only
    # connected to other terminals) in order to ensure the terminals are also terminals
    # in the solution
    # NOTE: This behavior is disabled by default because we don't generate the actual
    # terminals of the tufts with Steiner Tree, we just generate long range trunk that
    # passes near the target points.
    if config.use_terminal_penalty:
        logger.debug("Add terminal penalties to edge weights")
        add_terminal_penalty(edges_df, nodes_df)

    logger.debug("All penalties added to edge weights")

    if output_path is not None:
        logger.debug("Export nodes and edges to %s", output_path)
        Path(output_path).unlink(missing_ok=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        nodes_df.to_hdf(output_path, key="nodes")
        edges_df.to_hdf(output_path, key="edges", mode="a")

    if figure_path is not None:
        plot_triangulation(
            edges_df,
            source_coords,
            pts[1:, :3],  # The first one is the source
            figure_path,
            logger=logger,
            attractors=config.preferred_region_tree.data
            if config.preferred_region_tree is not None
            else None,
        )

    # pylint: disable=unsupported-assignment-operation
    nodes_df["NodeProvider"] = nodes_df.loc[:, "NodeProvider"].map(
        {i: i.name for i in NodeProvider}
    )

    return nodes_df, edges_df
