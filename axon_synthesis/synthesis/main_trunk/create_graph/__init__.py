"""Create the edges between the terminals and the obstacles (if any).

This is needed to easily compute a Steiner Tree (Euclidean Steiner Tree is complicated).
"""
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
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_favored_reward
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_intermediate_points
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_orientation_penalty
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_random_points
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_terminal_penalty
from axon_synthesis.synthesis.main_trunk.create_graph.utils import add_voronoi_points
from axon_synthesis.synthesis.main_trunk.create_graph.utils import create_edges
from axon_synthesis.synthesis.main_trunk.create_graph.utils import drop_close_points
from axon_synthesis.synthesis.main_trunk.create_graph.utils import drop_outside_points
from axon_synthesis.typing import FileType
from axon_synthesis.typing import RegionIdsType
from axon_synthesis.typing import SeedType
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
        favored_regions: The list of brain regions in which edge weights are divided by the
            favoring factor.
        favoring_sigma: The sigma used to favor the given regions.
        favoring_amplitude: The amplitude used to favor the given regions.
        favored_region_tree: The KDTree object containing the favored region points.
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

    # Favored regions
    favored_regions: RegionIdsType | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(RegionIdsType.__args__)),  # type: ignore[attr-defined]
    )
    favoring_sigma: float = field(default=100, validator=validators.gt(0))
    favoring_amplitude: float = field(default=1, validator=validators.gt(0))
    favored_region_tree: KDTree | None = field(default=None)

    # Terminal penalty
    use_terminal_penalty: bool = field(default=False)

    def compute_region_tree(self, atlas: AtlasHelper, *, force: bool = False):
        """Compute the favored region tree using the given Atlas."""
        if self.favored_regions is not None and (self.favored_region_tree is None or force):
            favored_region_points, _missing_ids = atlas.get_region_points(self.favored_regions)
            self.favored_region_tree = KDTree(favored_region_points)
        else:
            self.favored_region_tree = None


def one_graph(
    source_coords: np.ndarray,
    target_points: pd.DataFrame,
    config: CreateGraphConfig,
    favored_region_tree: KDTree | None = None,
    bbox: np.ndarray | None = None,
    depths: VoxelData | None = None,
    *,
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

    # Add intermediate points
    all_pts = add_intermediate_points(
        pts,
        source_coords,
        config.min_intermediate_distance,
        config.intermediate_number,
    )

    # Add random points
    all_pts = add_random_points(
        all_pts,
        config.min_random_point_distance,
        config.random_point_bbox_buffer,
        rng,
        logger=logger,
    )

    # Add the bounding box points to ensure a minimum number of points
    all_pts = add_bounding_box_pts(all_pts)

    # Add Voronoï points
    all_pts = add_voronoi_points(all_pts, config.voronoi_steps)

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
        pts,
        bbox=bbox,
    )

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

    # Compute cumulative penalties
    penalties = np.ones(len(edges_df))

    # Increase the weight of edges whose angle with radial direction is close to pi/2
    if config.use_orientation_penalty:
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
        penalties *= add_depth_penalty(
            edges_df,
            FROM_COORDS_COLS,
            TO_COORDS_COLS,
            depths,
            config.depth_penalty_sigma,
            config.depth_penalty_amplitude,
        )

    # Reduce the lengths of edges that are close to fiber tracts
    if favored_region_tree is not None:
        penalties *= add_favored_reward(
            edges_df,
            FROM_COORDS_COLS,
            TO_COORDS_COLS,
            favored_region_tree,
            config.favoring_sigma,
            config.favoring_amplitude,
        )

    # TODO: increase weights of more impossible edges?

    # Apply cumulative penalties
    # import pdb
    # pdb.set_trace()
    edges_df["weight"] = edges_df["length"] * penalties

    # TODO: Remove points and edges from forbidden regions?

    # Add penalty to edges between two terminals (except if a terminal is only
    # connected to other terminals) in order to ensure the terminals are also terminals
    # in the solution
    # NOTE: This behavior is disabled by default because we don't generate the actual
    # terminals of the tufts with Steiner Tree, we just generate long range trunk that
    # passes near the target points.
    if config.use_terminal_penalty:
        add_terminal_penalty(edges_df, nodes_df)

    logger.info("%s edges", len(edges_df))

    if output_path is not None:
        Path(output_path).unlink(missing_ok=True)
        nodes_df.to_hdf(output_path, key="nodes")
        edges_df.to_hdf(output_path, key="edges", mode="a")

    if figure_path is not None:
        plot_triangulation(
            edges_df,
            source_coords,
            pts[1:, :3],  # The first one is the source
            figure_path,
            logger=logger,
        )

    # pylint: disable=unsupported-assignment-operation
    nodes_df["NodeProvider"] = nodes_df.loc[:, "NodeProvider"].map(
        {i: i.name for i in NodeProvider}
    )

    return nodes_df, edges_df
