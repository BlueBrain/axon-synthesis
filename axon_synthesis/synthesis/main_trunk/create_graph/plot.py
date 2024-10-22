"""Some plot utils for create graph."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import numpy as np
import plotly.graph_objs as go

from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.utils import build_layout_properties


def plot_triangulation(
    edges, source_point, target_points, figure_path, logger=None, attractors=None
):
    """Plot the given triangulation for debugging purpose."""
    segments = edges.copy(deep=False)
    segments["cutter"] = None

    fig = go.Figure()

    colors = segments["weight"] / segments["length"]
    cmin = colors.quantile(0.1)
    cmax = colors.quantile(0.9)

    edges_trace = go.Scatter3d(
        x=segments[[FROM_COORDS_COLS.X, TO_COORDS_COLS.X, "cutter"]].to_numpy().flatten().tolist(),
        y=segments[[FROM_COORDS_COLS.Y, TO_COORDS_COLS.Y, "cutter"]].to_numpy().flatten().tolist(),
        z=segments[[FROM_COORDS_COLS.Z, TO_COORDS_COLS.Z, "cutter"]].to_numpy().flatten().tolist(),
        line={
            "width": 1.5,
            "color": colors,
            "colorscale": "rainbow",
            "cmin": cmin,
            "cmax": cmax,
            "colorbar": {
                "orientation": "h",
            },
        },
        mode="lines",
        name="Steiner graph",
    )

    source_point_trace = go.Scatter3d(
        x=[source_point[0]],
        y=[source_point[1]],
        z=[source_point[2]],
        marker={"color": "purple", "size": 4},
        mode="markers",
        name="Source point",
    )

    target_points_trace = go.Scatter3d(
        x=target_points[:, 0],
        y=target_points[:, 1],
        z=target_points[:, 2],
        marker={"color": "chocolate", "size": 2},
        mode="markers",
        name="Target points",
    )

    fig.add_trace(edges_trace)
    fig.add_trace(source_point_trace)
    fig.add_trace(target_points_trace)

    pts = np.vstack([[source_point], target_points])

    if attractors is not None:
        attractors_trace = go.Scatter3d(
            x=attractors[:, 0],
            y=attractors[:, 1],
            z=attractors[:, 2],
            marker={"color": "green", "size": 3},
            mode="markers",
            name="Attractors",
        )
        fig.add_trace(attractors_trace)
        pts = np.vstack([pts, attractors])

    layout_props = build_layout_properties(pts, 0.1)

    fig.update_scenes(layout_props)
    fig.update_layout(title=Path(figure_path).stem, coloraxis_showscale=True)

    # Export figure
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(figure_path)

    if logger is not None:
        logger.debug("Exported triangulation figure to %s", figure_path)
