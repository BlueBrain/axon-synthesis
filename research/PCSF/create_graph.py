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
from scipy.spatial import Voronoi

from PCSF.clustering import ClusterTerminals

logger = logging.getLogger(__name__)


class CreateGraph(luigi_tools.task.WorkflowTask):
    terminals_path = luigi.Parameter(description="Path to the terminals CSV file.", default=None)
    output_dataset = luigi.Parameter(description="Output dataset file.", default="graph_edges.csv")
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
        print(f"Get terminals from {self.terminals_path or self.input().path}")
        output_file = Path(self.output().path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

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

            # Voronoï points
            vor = Voronoi(pts)

            # Gather points
            all_points = pd.DataFrame(np.concatenate([pts, vor.vertices]), columns=["x", "y", "z"])

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
            edges_df[from_coord_cols] = all_points.loc[edges_df["from"]].values
            edges_df[to_coord_cols] = all_points.loc[edges_df["to"]].values
            edges_df["length"] = np.linalg.norm(
                edges_df[from_coord_cols].values - edges_df[to_coord_cols].values, axis=1
            )
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

        # Export the edges
        all_edges_df = pd.concat(all_edges)
        all_edges_df.to_csv(output_file, index=False)

    def output(self):
        return luigi.LocalTarget(self.output_dataset)
