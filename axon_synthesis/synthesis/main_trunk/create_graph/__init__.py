"""Create the edges between the terminals and the obstacles (if any).

This is needed to easily compute a Steiner Tree (Euclidean Steiner Tree is complicated).
"""
import logging
import sys

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from scipy.spatial import KDTree

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.atlas import load as load_atlas
from axon_synthesis.config import Config
from axon_synthesis.main_trunk.clustering import ClusterTerminals
from axon_synthesis.main_trunk.create_graph.plot import plot_triangulation
from axon_synthesis.main_trunk.create_graph.utils import add_depth_penalty
from axon_synthesis.main_trunk.create_graph.utils import add_favored_reward
from axon_synthesis.main_trunk.create_graph.utils import add_intermediate_points
from axon_synthesis.main_trunk.create_graph.utils import add_orientation_penalty
from axon_synthesis.main_trunk.create_graph.utils import add_random_points
from axon_synthesis.main_trunk.create_graph.utils import add_terminal_penalty
from axon_synthesis.main_trunk.create_graph.utils import add_voronoi_points
from axon_synthesis.main_trunk.create_graph.utils import create_edges
from axon_synthesis.main_trunk.create_graph.utils import drop_close_points
from axon_synthesis.main_trunk.create_graph.utils import drop_outside_points
from axon_synthesis.main_trunk.create_graph.utils import get_region_points
from axon_synthesis.main_trunk.create_graph.utils import use_ancestors
from axon_synthesis.target_points import FindTargetPoints

logger = logging.getLogger(__name__)


class CreateGraph(luigi_tools.task.WorkflowTask):
    """Task to create the graph on which the Steiner algorithm will be applied."""

    terminals_path = luigi.parameter.OptionalPathParameter(
        description="Path to the terminals CSV file.",
        default=None,
        exists=True,
    )
    tuft_properties_path = luigi.parameter.OptionalPathParameter(
        description="Path to the tuft properties JSON file.",
        default=None,
        exists=True,
    )
    output_nodes = luigi.Parameter(description="Output nodes file.", default="graph_nodes.csv")
    output_edges = luigi.Parameter(description="Output edges file.", default="graph_edges.csv")
    intermediate_number = luigi.NumericalParameter(
        description="The number of intermediate points added before Voronoï process.",
        var_type=int,
        default=5,
        min_value=0,
        max_value=sys.float_info.max,
    )
    min_intermediate_distance = luigi.NumericalParameter(
        description="The min distance between two successive intermediate points.",
        var_type=int,
        default=1000,
        min_value=0,
        max_value=sys.float_info.max,
    )
    orientation_penalty_exponent = luigi.NumericalParameter(
        description="The exponent used for the orientation penalty.",
        var_type=float,
        default=0.1,
        min_value=0,
        max_value=sys.float_info.max,
    )
    voronoi_steps = luigi.NumericalParameter(
        description="The number of Voronoi steps.",
        var_type=int,
        default=1,
        min_value=1,
        max_value=sys.float_info.max,
    )
    duplicate_precision = luigi.NumericalParameter(
        description="The precision used to detect duplicated points.",
        var_type=float,
        default=1e-3,
        min_value=0,
        max_value=sys.float_info.max,
    )
    min_random_point_distance = luigi.parameter.OptionalNumericalParameter(
        description="The minimal distance used to add random points.",
        var_type=float,
        default=None,
        min_value=0,
        max_value=sys.float_info.max,
    )
    random_point_bbox_buffer = luigi.parameter.NumericalParameter(
        description="The distance used to add a buffer around the bbox of the points.",
        var_type=float,
        default=0,
        min_value=0,
        max_value=sys.float_info.max,
    )
    seed = luigi.IntParameter(
        description="The seed used to generate random points.",
        default=0,
    )
    use_ancestors = luigi.BoolParameter(
        description=(
            "If set to True, the common ancestors are used to build the graph points instead of "
            "the cluster centers."
        ),
        default=False,
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )
    use_terminal_penalty = luigi.BoolParameter(
        description=(
            "If set to True, a penalty is added to edges that are connected to a terminal."
        ),
        default=False,
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )
    use_orientation_penalty = luigi.BoolParameter(
        description=("If set to True, a penalty is added to edges whose direction is not radial."),
        default=True,
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )
    orientation_penalty_amplitude = luigi.NumericalParameter(
        description="The amplitude of the orientation penalty.",
        var_type=float,
        default=1,
        min_value=0,
        max_value=sys.float_info.max,
        left_op=luigi.parameter.operator.lt,
    )
    use_depth_penalty = luigi.BoolParameter(
        description=(
            "If set to True, a penalty is added to edges whose direction is not parallel to the "
            "iso-depth curves."
        ),
        default=True,
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )
    depth_penalty_sigma = luigi.NumericalParameter(
        description="The sigma used for depth penalty.",
        var_type=float,
        default=100,
        min_value=0,
        max_value=sys.float_info.max,
        left_op=luigi.parameter.operator.lt,
    )
    depth_penalty_amplitude = luigi.NumericalParameter(
        description="The amplitude of the depth penalty.",
        var_type=float,
        default=1,
        min_value=0,
        max_value=sys.float_info.max,
        left_op=luigi.parameter.operator.lt,
    )
    layers_indices = luigi.ListParameter(
        description="The list of layer numbers.",
        default=[1, 2, 3, 4, 5, 6],
    )
    favored_regions = luigi.OptionalListParameter(
        description=(
            "Provide a list of brain regions in which the edge weights are divided by the favoring "
            "factor."
        ),
        default=None,
    )
    favoring_sigma = luigi.NumericalParameter(
        description="The sigma used to favor some regions.",
        var_type=float,
        default=100,
        min_value=0,
        max_value=sys.float_info.max,
        left_op=luigi.parameter.operator.lt,
    )
    favoring_amplitude = luigi.NumericalParameter(
        description="The amplitude used to favor some regions.",
        var_type=float,
        default=1,
        min_value=0,
        max_value=sys.float_info.max,
        left_op=luigi.parameter.operator.lt,
    )
    plot_debug = luigi.BoolParameter(
        description=(
            "If set to True, each group will create an interactive figure so it is possible to "
            "check the graph edges."
        ),
        default=False,
    )

    def requires(self):
        input_data_type = Config().input_data_type
        if input_data_type == "biological_morphologies":
            return ClusterTerminals()
        elif input_data_type == "white_matter":
            return FindTargetPoints()
        elif input_data_type == "custom":
            return []
        else:
            raise ValueError(f"The value of 'input_data_type' is unknown ({input_data_type}).")

    def run(self):
        # pylint: disable=too-many-statements
        config = Config()

        terminals = pd.read_csv(self.terminals_path or self.input()["terminals"].path)
        terminals.to_csv(self.output()["input_terminals"].path, index=False)

        favored_region_tree = None
        brain_regions = None
        if self.favored_regions or self.use_depth_penalty:
            atlas, brain_regions, region_map = load_atlas(
                str(config.atlas_path),
                config.atlas_region_filename,
                config.atlas_hierarchy_filename,
            )
            if self.use_depth_penalty:
                atlas_helper = AtlasHelper(atlas, self.layers_indices)
            if self.favored_regions:
                favored_region_points = get_region_points(
                    brain_regions,
                    region_map,
                    self.favored_regions,
                )
                favored_region_tree = KDTree(favored_region_points)

        if self.use_ancestors:
            if self.terminals_path:
                raise ValueError("Can not use ancestors when 'terminals_path' is not None")
            use_ancestors(
                terminals,
                self.tuft_properties_path or self.input()["tuft_properties"].path,
            )

        soma_centers = terminals.loc[terminals["axon_id"] == -1].copy()
        terminals = terminals.loc[terminals["axon_id"] != -1].copy()

        all_nodes = []
        all_edges = []
        from_coord_cols = ["x_from", "y_from", "z_from"]
        to_coord_cols = ["x_to", "y_to", "z_to"]

        for group_name, group in terminals.groupby("morph_file"):
            logger.debug("%s: %s points", group_name, len(group))

            # Terminal points
            pts = group[["x", "y", "z"]].values
            soma_center = soma_centers.loc[soma_centers["morph_file"] == group_name]
            soma_center_coords = soma_center[["x", "y", "z"]].values[0]

            # Add intermediate points
            inter_pts = add_intermediate_points(
                pts,
                soma_center_coords,
                self.min_intermediate_distance,
                self.intermediate_number,
            )
            all_pts = np.concatenate([pts] + [i[1] for i in inter_pts if i[0] > 0])

            # Add random points
            all_pts = add_random_points(
                all_pts,
                self.min_random_point_distance,
                self.random_point_bbox_buffer,
                self.seed,
            )

            # Add Voronoï points
            all_pts = add_voronoi_points(all_pts, self.voronoi_steps)

            # Gather points
            all_points_df = pd.DataFrame(all_pts, columns=["x", "y", "z"])
            all_points_df["morph_file"] = group_name
            all_points_df["is_terminal"] = [True] * len(pts) + [False] * (len(all_pts) - len(pts))

            # Remove close points
            drop_close_points(all_points_df, self.duplicate_precision)

            # Remove outside points
            drop_outside_points(all_points_df, pts if self.use_ancestors else None, brain_regions)

            # Reset index and set IDs
            all_points_df.reset_index(drop=True, inplace=True)
            all_points_df["id"] = all_points_df.index

            # Save nodes
            all_nodes.append(all_points_df[["morph_file", "x", "y", "z", "is_terminal", "id"]])

            # Create edges using the Delaunay triangulation of the union of the terminals,
            # intermediate and Voronoï points
            edges_df, tri = create_edges(
                all_points_df[["x", "y", "z"]],
                from_coord_cols,
                to_coord_cols,
                group_name,
            )

            # Compute cumulative penalties
            penalties = np.ones(len(edges_df))

            # Increase the length of edges whose angle with radial direction is close to pi/2
            if self.use_orientation_penalty:
                penalties *= add_orientation_penalty(
                    edges_df,
                    from_coord_cols,
                    to_coord_cols,
                    soma_center_coords,
                    self.orientation_penalty_exponent,
                    self.orientation_penalty_amplitude,
                )

            # Increase the length of edges which do not follow an iso-depth curve
            if self.use_depth_penalty:
                penalties *= add_depth_penalty(
                    edges_df,
                    from_coord_cols,
                    to_coord_cols,
                    atlas_helper,
                    self.depth_penalty_sigma,
                    self.depth_penalty_amplitude,
                )

            # Reduce the lengths of edges that are close to fiber tracts
            if favored_region_tree is not None:
                penalties *= add_favored_reward(
                    edges_df,
                    from_coord_cols,
                    to_coord_cols,
                    favored_region_tree,
                    self.favoring_sigma,
                    self.favoring_amplitude,
                )

            # TODO: increase lengths of more impossible edges

            # Apply cumulative penalties
            edges_df["length"] *= penalties

            # Add penalty to edges between two terminals (except if a terminal is only
            # connected to other terminals) in order to ensure the terminals are also terminals
            # in the solution
            # NOTE: This behavior is disabled by default because we don't generate the actual
            # terminals of the tufts with Steiner Tree, we just generate long range trunk that
            # passes near the target points.
            if self.use_terminal_penalty:
                add_terminal_penalty(edges_df, all_points_df)

            # Save edges
            all_edges.append(edges_df)
            logger.info("%s: %s edges", group_name, len(edges_df))

            if self.plot_debug:
                plot_triangulation(
                    edges_df,
                    from_coord_cols,
                    to_coord_cols,
                    tri,
                    all_points_df[["x", "y", "z"]],
                    pts,
                )

        # Export the nodes
        all_nodes_df = pd.concat(all_nodes)
        all_nodes_df.to_csv(self.output()["nodes"].path, index=False)

        # Export the edges
        all_edges_df = pd.concat(all_edges)
        all_edges_df.to_csv(self.output()["edges"].path, index=False)

    def output(self):
        return {
            "nodes": TaggedOutputLocalTarget(self.output_nodes, create_parent=True),
            "edges": TaggedOutputLocalTarget(self.output_edges, create_parent=True),
            "input_terminals": TaggedOutputLocalTarget(
                "graph_input_terminals.csv",
                create_parent=True,
            ),
        }
