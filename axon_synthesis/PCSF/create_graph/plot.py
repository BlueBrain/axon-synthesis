"""Some plot utils for create graph."""
import logging
import time

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_triangulation(edges_df, from_coord_cols, to_coord_cols, tri, all_points, pts):
    """Plot the given triangulation for debugging purpose."""
    old_backend = matplotlib.get_backend()
    matplotlib.use("TkAgg")

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
        matplotlib.use(old_backend)
