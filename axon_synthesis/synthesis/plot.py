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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.constants import TARGET_COORDS_COLS
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import build_layout_properties
from axon_synthesis.utils import get_morph_pts


def plot_final_morph(morph, target_points, output_path, initial_morph=None, logger=None):
    """Plot the given morphology.

    If `initial_morph` is not None then the given morphology is also plotted for comparison.
    """
    title = "Final morphology"
    fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=title)
    fig_data = [fig_builder.get_figure()["data"]]
    left_title = "Synthesized morphology"

    if initial_morph is not None:
        fig = make_subplots(
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=[left_title, "Initial morphology"],
        )

        if initial_morph.root_sections:
            fig_data.append(
                NeuronBuilder(initial_morph, "3d", line_width=4, title=title).get_figure()["data"]
            )
        else:
            fig_data.append(
                [
                    go.Scatter3d(
                        x=[initial_morph.soma.center[0]],
                        y=[initial_morph.soma.center[1]],
                        z=[initial_morph.soma.center[2]],
                        marker={"color": "black", "size": 4},
                        mode="markers",
                        name="Soma",
                    )
                ]
            )
    else:
        fig = make_subplots(cols=1, specs=[[{"type": "scene"}]], subplot_titles=[left_title])

    for col_num, data in enumerate(fig_data):
        fig.add_traces(data, rows=[1] * len(data), cols=[col_num + 1] * len(data))

    x, y, z = target_points[TARGET_COORDS_COLS].to_numpy().T
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={"size": 2, "color": "chocolate"},
        name="Target points",
    )
    fig.add_trace(node_trace, row=1, col=1)

    layout_props = build_layout_properties(
        np.vstack([morph.points, get_morph_pts(initial_morph)]), 0.1
    )

    fig.update_scenes(layout_props)
    fig.update_layout(title=morph.name)

    # Export figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)

    if initial_morph is not None:
        add_camera_sync(output_path)

    if logger is not None:
        logger.info("Exported figure to %s", output_path)


def plot_target_points(morph, source_point, target_points, output_path, logger=None):
    """Plot the source and target points along the given morphology."""
    title = "Initial morphology"

    fig = make_subplots(cols=1, specs=[[{"type": "scene"}]], subplot_titles=[title])

    if morph.root_sections:
        fig.add_traces(NeuronBuilder(morph, "3d", line_width=4, title=title).get_figure()["data"])
    else:
        fig.add_traces(
            go.Scatter3d(
                x=[morph.soma.center[0]],
                y=[morph.soma.center[1]],
                z=[morph.soma.center[2]],
                marker={"color": "black", "size": 4},
                mode="markers",
                name="Soma",
            )
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
        marker={"color": "chocolate", "size": 3},
        mode="markers",
        name="Target points",
    )

    fig.add_trace(source_point_trace)
    fig.add_trace(target_points_trace)

    layout_props = build_layout_properties(
        np.concatenate(
            [
                (
                    morph.points
                    if len(morph.root_sections) > 0
                    else np.atleast_2d(morph.soma.center)
                )[:, :3],
                [source_point],
                target_points,
            ]
        ),
        0.1,
    )

    fig.update_scenes(layout_props)
    fig.update_layout(title=morph.name)

    # Export figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)

    if logger is not None:
        logger.debug("Exported figure to %s", output_path)
