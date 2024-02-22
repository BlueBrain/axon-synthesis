"""Some plot utils for create graph."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.constants import TARGET_COORDS_COLS
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import build_layout_properties


def plot_final_morph(morph, target_points, output_path, initial_morph=None, logger=None):
    """Plot the given morphology.

    If `initial_morph` is not None then the given morphology is also plotted for comparison.
    """
    title = "Final morphology"
    fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=title)
    fig_data = [fig_builder.get_figure()["data"]]
    left_title = "Synthesized morphology"

    if initial_morph is not None:
        raw_builder = NeuronBuilder(initial_morph, "3d", line_width=4, title=title)

        fig = make_subplots(
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=[left_title, "Initial morphology"],
        )
        fig_data.append(raw_builder.get_figure()["data"])
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
        marker={"size": 2, "color": "orange"},
        name="Target points",
    )
    fig.add_trace(node_trace, row=1, col=1)

    layout_props = build_layout_properties(morph.points, 0.1)

    fig.update_scenes(layout_props)
    fig.update_layout(title=morph.name)

    # Export figure
    fig.write_html(output_path)

    if initial_morph is not None:
        add_camera_sync(output_path)

    if logger is not None:
        logger.info("Exported figure to %s", output_path)


def plot_target_points(morph, source_point, target_points, output_path):
    """Plot the source and target points along the given morphology."""
    title = "Initial morphology"
    fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=title)

    fig = make_subplots(cols=1, specs=[[{"type": "scene"}]], subplot_titles=[title])

    source_point_trace = go.Scatter3d(
        x=[source_point[0]],
        y=[source_point[1]],
        z=[source_point[2]],
        marker={"color": "rgb(255,0,0)", "size": 4},
        mode="markers",
        name="Source point",
    )

    target_points_trace = go.Scatter3d(
        x=target_points[:, 0],
        y=target_points[:, 1],
        z=target_points[:, 2],
        marker={"color": "rgb(0,0,255)", "size": 2},
        mode="markers",
        name="Target points",
    )

    fig.add_traces(fig_builder.get_figure()["data"])
    fig.add_trace(source_point_trace)
    fig.add_trace(target_points_trace)

    layout_props = build_layout_properties(
        np.concatenate([morph.points[:, :3], [source_point], target_points]), 0.1
    )

    fig.update_scenes(layout_props)
    fig.update_layout(title=morph.name)

    # Export figure
    fig.write_html(output_path)
