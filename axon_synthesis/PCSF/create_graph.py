"""Create the edges between the terminals and the obstacles (if any).

This is needed to easily compute a Steiner Tree (Euclidean Steiner Tree is complicated).
"""
import logging
import sys
import time

import luigi
import luigi_tools
import matplotlib
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from neurom.morphmath import angle_between_vectors
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from scipy.spatial import Voronoi
from voxcell.nexus.voxelbrain import Atlas

from axon_synthesis.config import Config
from axon_synthesis.PCSF.clustering import ClusterTerminals
from axon_synthesis.target_points import FindTargetPoints

logger = logging.getLogger(__name__)


class CreateGraph(luigi_tools.task.WorkflowTask):
    """Task to create the graph on which the Steiner algorithm will be applied."""

    terminals_path = luigi.parameter.OptionalPathParameter(
        description="Path to the terminals CSV file.",
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
        var_type=int,
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
    )
    terminal_penalty = luigi.BoolParameter(
        description=(
            "If set to True, a penalty is added to edges that are connected to a terminal."
        ),
        default=False,
    )
    orientation_penalty = luigi.BoolParameter(
        description=("If set to True, a penalty is added to edges whose direction is not radial."),
        default=True,
    )
    atlas_path = luigi.parameter.OptionalPathParameter(
        description="Path to the atlas directory.",
        default=None,
        exists=True,
    )
    atlas_hierarchy_filename = luigi.Parameter(
        description="Atlas hierarchy file.",
        default="hierarchy.json",
    )
    atlas_region_filename = luigi.Parameter(
        description="Atlas regions file.",
        default="brain_regions",
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
        else:
            raise ValueError(f"The value of 'input_data_type' is unknown ({input_data_type}).")

    def run(self):
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        terminals = pd.read_csv(self.terminals_path or self.input()["terminals"].path)
        terminals.to_csv(self.output()["input_terminals"].path, index=False)

        if self.atlas_path is not None:
            atlas = Atlas.open(self.atlas_path)
            region_map = atlas.load_region_map(self.atlas_hierarchy_filename)
            brain_regions = atlas.load_data(self.atlas_region_filename)
            fiber_tracts_ids = region_map.find("fiber tracts", attr="name", with_descendants=True)
            fiber_tracts_mask = np.isin(brain_regions.raw, list(fiber_tracts_ids))
            brain_regions.raw[~fiber_tracts_mask] = 0  # Zeroes the complement region
            fiber_tract_points = brain_regions.indices_to_positions(np.argwhere(brain_regions.raw))
            fiber_tract_tree = KDTree(fiber_tract_points)  # noqa; pylint: disable=unused-variable
        else:
            fiber_tract_points = None
            fiber_tract_tree = None  # noqa

        if self.use_ancestors:
            if self.terminals_path:
                raise ValueError("Can not use ancestors when 'terminals_path' is not None")
            cluster_props_df = pd.read_json(self.input()["tuft_properties"].path)
            tmp = pd.merge(
                terminals,
                cluster_props_df,
                left_on=["morph_file", "axon_id", "terminal_id"],
                right_on=["morph_file", "axon_id", "cluster_id"],
                how="left",
            )
            mask = ~tmp["cluster_id"].isnull()
            new_terminal_coords = pd.DataFrame(
                tmp.loc[mask, "common_ancestor_coords"].to_list(),
                columns=["x", "y", "z"],
            )
            tmp.loc[mask, ["x", "y", "z"]] = new_terminal_coords.values
            terminals[["x", "y", "z"]] = tmp[["x", "y", "z"]]

        soma_centers = terminals.loc[terminals["axon_id"] == -1].copy()
        terminals = terminals.loc[terminals["axon_id"] != -1].copy()

        all_nodes = []
        all_edges = []
        from_coord_cols = ["x_from", "y_from", "z_from"]
        to_coord_cols = ["x_to", "y_to", "z_to"]

        if self.plot_debug:
            old_backend = matplotlib.get_backend()
            matplotlib.use("TkAgg")

        for group_name, group in terminals.groupby("morph_file"):
            logger.debug("%s: %s points", group_name, len(group))

            # Terminal points
            pts = group[["x", "y", "z"]].values
            soma_center = soma_centers.loc[soma_centers["morph_file"] == group_name]
            soma_center_coords = soma_center[["x", "y", "z"]].values[0]

            all_pts = pts

            # Add intermediate points
            terms = pts - soma_center[["x", "y", "z"]].values[0]
            term_dists = np.linalg.norm(terms, axis=1)
            nb_inter = np.clip(
                term_dists // self.min_intermediate_distance,
                0,
                self.intermediate_number,
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
                        + soma_center_coords,
                    )
                )
            all_pts = np.concatenate([all_pts] + [i[1] for i in inter_pts if i[0] > 0])

            # Add random points
            if self.min_random_point_distance is not None:
                n_fails = 0
                bbox = np.vstack([all_pts.min(axis=0), all_pts.max(axis=0)])
                rng = np.random.default_rng(self.seed)
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
                        tree.query(xyz, distance_upper_bound=self.min_random_point_distance, k=2,)[
                            0
                        ][1]
                    ) and (
                        len(new_pts) == 0
                        or np.linalg.norm(
                            xyz - new_pts,
                            axis=1,
                        ).min()
                        > self.min_random_point_distance
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
                        self.min_random_point_distance,
                    )

            # Add Voronoï points
            for i in range(self.voronoi_steps):
                vor = Voronoi(all_pts, qhull_options="QJ")
                all_pts = np.concatenate([all_pts, vor.vertices])  # pylint: disable=no-member

            # Gather points
            all_points_df = pd.DataFrame(all_pts, columns=["x", "y", "z"])
            all_points_df["morph_file"] = group_name
            all_points_df["is_terminal"] = [True] * len(pts) + [False] * (len(all_pts) - len(pts))

            # Remove close points
            tree = KDTree(all_points_df[["x", "y", "z"]])
            close_pts = tree.query_pairs(self.duplicate_precision)
            for a, b in close_pts:
                if a in all_points_df.index and b in all_points_df.index:
                    all_points_df.drop(a, inplace=True)

            # Remove outside points
            min_pts = pts.min(axis=0)
            max_pts = pts.max(axis=0)
            outside_pts = all_points_df.loc[
                ((all_points_df[["x", "y", "z"]] < min_pts).any(axis=1))
                | ((all_points_df[["x", "y", "z"]] > max_pts).any(axis=1))
            ]
            all_points_df.drop(outside_pts.index, inplace=True)

            # Reset index and set IDs
            all_points_df.reset_index(drop=True, inplace=True)
            all_points_df["id"] = all_points_df.index

            # Save nodes
            all_nodes.append(all_points_df[["morph_file", "x", "y", "z", "is_terminal", "id"]])
            all_points = all_points_df[["x", "y", "z"]]

            # Delaunay triangulation of the union of the terminals and the Voronoï points
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

            # Compute penalty
            penalty = edges_df["length"].max() + edges_df["length"].mean()

            # Get terminal edges
            terminal_edges = edges_df[["from", "to"]].isin(
                all_points_df.loc[all_points_df["is_terminal"], "id"].values
            )

            # Add penalty to all terminal edges in order to ensure the terminals are also terminals
            # in the solution
            # NOTE: Disabled because we don't generate the actual terminals of the tufts for now,
            # we just generate long range trunk that passes near the target points.
            # edges_df.loc[terminal_edges.any(axis=1), "length"] += penalty

            # Add penalty to edges between two terminals (except if a terminal is only
            # connected to other terminals)
            if self.terminal_penalty:
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

            # Increase edge lengths of edges whose angle with radial direction is close to pi/2
            if self.orientation_penalty:
                vectors = edges_df[to_coord_cols].values - edges_df[from_coord_cols].values
                origin_to_mid_vectors = (
                    0.5 * (edges_df[to_coord_cols].values + edges_df[from_coord_cols].values)
                    - soma_center_coords
                )
                data = np.stack([origin_to_mid_vectors, vectors], axis=1)

                edge_angles = np.array([angle_between_vectors(i[0], i[1]) for i in data.tolist()])
                orientation_penalty = np.power(
                    np.clip(np.sin(edge_angles), 1e-3, 1 - 1e-3),
                    self.orientation_penalty_exponent,
                )
                edges_df["length"] *= orientation_penalty

            # TODO: increase lengths of more impossible edges

            # Save edges
            all_edges.append(edges_df)
            logger.info("%s: %s edges", group_name, len(edges_df))

            if self.plot_debug:
                try:
                    disabled_loggers = [
                        logging.getLogger("matplotlib.font_manager"),
                        logging.getLogger("PIL.PngImagePlugin"),
                    ]
                    for dl in disabled_loggers:
                        dl.disabled = True

                    # Prepare data for plot
                    mask_from = (edges_df[from_coord_cols] < pts.min(axis=0)).any(axis=1) | (
                        edges_df[from_coord_cols] > pts.max(axis=0)
                    ).any(axis=1)
                    mask_to = (edges_df[to_coord_cols] < pts.min(axis=0)).any(axis=1) | (
                        edges_df[to_coord_cols] > pts.max(axis=0)
                    ).any(axis=1)
                    out_pts = np.unique(
                        np.concatenate(
                            [
                                edges_df.loc[mask_from, "from"],
                                edges_df.loc[mask_to, "to"],
                            ]
                        )
                    )

                    masked_tri = tri.simplices[  # pylint: disable=no-member
                        # pylint: disable=no-member
                        ~np.isin(tri.simplices, out_pts).any(axis=1)
                    ]
                    triangles = np.unique(
                        np.apply_along_axis(
                            np.sort,
                            1,
                            np.vstack(
                                np.stack(
                                    (
                                        masked_tri,
                                        np.roll(masked_tri, -1, axis=1),
                                        np.roll(masked_tri, -2, axis=1),
                                    ),
                                    axis=2,
                                )
                            ),
                        ),
                        axis=0,
                    )
                    tri_col = Poly3DCollection(
                        all_points.values[triangles],
                        edgecolors="k",
                        facecolors=None,
                        linewidths=0.5,
                        alpha=0,
                    )

                    # Create the figure
                    fig = plt.figure(figsize=(12, 9))
                    ax = fig.gca(projection="3d")
                    ax.add_collection3d(tri_col)

                    # Plot the terminal points
                    ax.scatter3D(*pts.T, c="red")

                    # Set rotation center
                    pts_center = pts.mean(axis=0)
                    half_delta = 0.5 * (pts.max(axis=0) - pts.min(axis=0))
                    ax.set_xbound(pts_center[0] - half_delta[0], pts_center[0] + half_delta[0])
                    ax.set_ybound(pts_center[1] - half_delta[1], pts_center[1] + half_delta[1])
                    ax.set_zbound(pts_center[2] - half_delta[2], pts_center[2] + half_delta[2])

                    plt.show()
                    time.sleep(1)

                finally:
                    for dl in disabled_loggers:
                        dl.disabled = False

        if self.plot_debug:
            matplotlib.use(old_backend)

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
            "input_terminals": TaggedOutputLocalTarget("graph_input_terminals", create_parent=True),
        }
