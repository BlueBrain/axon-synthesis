"""Create the edges between the terminals and the obstacles (if any).

This is needed to easily compute a Steiner Tree (Euclidean Steiner Tree is complicated).
"""
import logging
import time
from pathlib import Path

import luigi
import luigi_tools
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from scipy.spatial import Voronoi

from PCSF.clustering import ClusterTerminals

logger = logging.getLogger(__name__)


class CreateGraph(luigi_tools.task.WorkflowTask):
    terminals_path = luigi.Parameter(description="Path to the terminals CSV file.", default=None)
    output_nodes = luigi.Parameter(description="Output nodes file.", default="graph_nodes.csv")
    output_edges = luigi.Parameter(description="Output edges file.", default="graph_edges.csv")
    voronoi_steps = luigi.NumericalParameter(
        description="The number Voronoi steps.",
        var_type=int,
        default=1,
        min_value=1,
        max_value=float("inf"),
    )
    duplicate_precision = luigi.NumericalParameter(
        description="The precision used to detect duplicated points.",
        var_type=float,
        default=1e-5,
        min_value=0,
        max_value=float("inf"),
    )
    plot_debug = luigi.BoolParameter(
        description=(
            "If set to True, each group will create an interactive figure so it is possible to "
            "check the graph edges."
        )
    )

    def requires(self):
        return ClusterTerminals()

    def run(self):
        terminals = pd.read_csv(self.terminals_path or self.input().path)
        output_nodes = Path(self.output()["nodes"].path)
        output_nodes.parent.mkdir(parents=True, exist_ok=True)
        output_edges = Path(self.output()["edges"].path)
        output_edges.parent.mkdir(parents=True, exist_ok=True)

        all_nodes = []
        all_edges = []
        from_coord_cols = ["x_from", "y_from", "z_from"]
        to_coord_cols = ["x_to", "y_to", "z_to"]

        if self.plot_debug:
            old_backend = matplotlib.get_backend()
            matplotlib.use("TkAgg")

        for group_name, group in terminals.groupby("morph_file"):
            logger.debug(f"{group_name}: {len(group)} points")

            # Terminal points
            pts = group[["x", "y", "z"]].values

            # Add Voronoï points
            all_pts = pts
            for i in range(self.voronoi_steps):
                vor = Voronoi(all_pts)
                all_pts = np.concatenate([all_pts, vor.vertices])

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
            tri = Delaunay(all_points)

            # Find all unique edges from the triangulation
            unique_edges = np.unique(
                np.apply_along_axis(
                    np.sort,
                    1,
                    np.vstack(
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

            # Remove edges between two terminals (except if a terminal is only connected to other
            # terminals)
            from_to_index, from_to_data = np.unique(edges_df[["from", "to"]].values, return_counts=True)
            terminal_edges = edges_df[["from", "to"]].isin(
                all_points_df.loc[all_points_df["is_terminal"], "id"].values
            )
            from_all_terminals = edges_df[["from"]].join(terminal_edges["from"].rename("from_all_terminals")).groupby("from").all()
            to_all_terminals = edges_df[["to"]].join(terminal_edges["to"].rename("to_all_terminals")).groupby("to").all()
            edges_df_terminals = edges_df.join(from_all_terminals, on="from")
            edges_df_terminals = edges_df_terminals.join(to_all_terminals, on="to")
            edges_df.drop(
                edges_df[
                    (terminal_edges.all(axis=1))
                    & (~edges_df_terminals[["from_all_terminals", "to_all_terminals"]].all(axis=1))
                ].index,
                inplace=True
            )
            # TODO: remove more impossible edges

            # Add coordinates
            edges_df[from_coord_cols] = all_points.loc[edges_df["from"]].values
            edges_df[to_coord_cols] = all_points.loc[edges_df["to"]].values
            edges_df["length"] = np.linalg.norm(
                edges_df[from_coord_cols].values - edges_df[to_coord_cols].values, axis=1
            )

            # Save edges
            all_edges.append(edges_df)
            logger.info(f"{group_name}: {len(edges_df)} edges")

            if self.plot_debug:
                # Prepare data for plot
                mask_from = (edges_df[from_coord_cols] < pts.min(axis=0)).any(axis=1) | (
                    edges_df[from_coord_cols] > pts.max(axis=0)
                ).any(axis=1)
                mask_to = (edges_df[to_coord_cols] < pts.min(axis=0)).any(axis=1) | (
                    edges_df[to_coord_cols] > pts.max(axis=0)
                ).any(axis=1)
                out_pts = np.unique(
                    np.concatenate([edges_df.loc[mask_from, "from"], edges_df.loc[mask_to, "to"]])
                )

                masked_tri = tri.simplices[~np.isin(tri.simplices, out_pts).any(axis=1)]
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

        if self.plot_debug:
            matplotlib.use(old_backend)

        # Export the nodes
        all_nodes_df = pd.concat(all_nodes)
        all_nodes_df.to_csv(output_nodes, index=False)

        # Export the edges
        all_edges_df = pd.concat(all_edges)
        all_edges_df.to_csv(output_edges, index=False)

    def output(self):
        return {
            "nodes": luigi_tools.target.OutputLocalTarget(self.output_nodes),
            "edges": luigi_tools.target.OutputLocalTarget(self.output_edges),
        }
