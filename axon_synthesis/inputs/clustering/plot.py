"""Some plot utils for clustering."""

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
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool import resampling
from neurom.core import Morphology
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.constants import COMMON_ANCESTOR_COORDS_COLS
from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import build_layout_properties
from axon_synthesis.utils import disable_loggers


def plot_clusters(morph, clustered_morph, group, group_name, cluster_df, output_path):
    """Plot clusters to a HTML figure."""
    plotted_morph = Morphology(
        resampling.resample_linear_density(morph, 0.005),
        name=Path(group_name).with_suffix("").name,
    )
    fig_builder = NeuronBuilder(plotted_morph, "3d", line_width=4, title=f"{plotted_morph.name}")

    x, y, z = group[COORDS_COLS].to_numpy().T
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={"size": 3, "color": "black"},
        name="Morphology nodes",
    )
    x, y, z = cluster_df[COORDS_COLS].to_numpy().T
    cluster_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={"size": 5, "color": "chocolate"},
        name="Cluster centers",
    )
    cluster_lines = [
        [
            [
                i["x"],
                cluster_df.loc[cluster_df["terminal_id"] == i["tuft_id"], "x"].iloc[0],
                None,
            ],
            [
                i["y"],
                cluster_df.loc[cluster_df["terminal_id"] == i["tuft_id"], "y"].iloc[0],
                None,
            ],
            [
                i["z"],
                cluster_df.loc[cluster_df["terminal_id"] == i["tuft_id"], "z"].iloc[0],
                None,
            ],
        ]
        for i in group.to_dict("records")
        if i["tuft_id"] >= 0
    ]
    edge_trace = go.Scatter3d(
        x=[j for i in cluster_lines for j in i[0]],
        y=[j for i in cluster_lines for j in i[1]],
        z=[j for i in cluster_lines for j in i[2]],
        hoverinfo="none",
        mode="lines",
        line={
            "color": "green",
            "width": 4,
        },
        name="Morphology nodes to cluster",
    )

    # Build the clustered morph figure
    clustered_builder = NeuronBuilder(
        clustered_morph,
        "3d",
        line_width=4,
        title=f"Clustered {clustered_morph.name}",
    )

    # Create the figure from the traces
    fig = make_subplots(
        cols=2,
        specs=[[{"is_3d": True}, {"is_3d": True}]],
        subplot_titles=("Node clusters", "Clustered morphology"),
    )

    morph_data = fig_builder.get_figure()["data"]
    fig.add_traces(morph_data, rows=[1] * len(morph_data), cols=[1] * len(morph_data))
    fig.add_trace(node_trace, row=1, col=1)
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(cluster_trace, row=1, col=1)

    clustered_morph_data = clustered_builder.get_figure()["data"]
    fig.add_traces(
        clustered_morph_data,
        rows=[1] * len(clustered_morph_data),
        cols=[2] * len(clustered_morph_data),
    )
    cluster_trace.showlegend = False
    fig.add_trace(cluster_trace, row=1, col=2)

    layout_props = build_layout_properties(morph.points, 0.1)

    fig.update_scenes(layout_props)
    fig.update_layout(title=group_name)

    # Export figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))

    add_camera_sync(str(output_path))


@disable_loggers(
    "matplotlib.font_manager",
    "matplotlib.pyplot",
    "matplotlib.backends.backend_pdf",
    "matplotlib.ticker",
    "PIL.PngImagePlugin",
)
def plot_cluster_properties(cluster_props_df, output_path):
    """Plot the cluster properties to a PDF figure."""
    if cluster_props_df.empty:
        return

    with PdfPages(str(output_path)) as pdf:
        ax = cluster_props_df.plot.scatter(
            x="path_distance",
            y="size",
            title="Cluster size vs path distance",
            legend=True,
        )
        ax.set_yscale("log")
        pdf.savefig()
        plt.close()

        ax = cluster_props_df.plot.scatter(
            x="radial_distance",
            y="size",
            title="Cluster size vs radial distance",
            legend=True,
        )
        ax.set_yscale("log")
        pdf.savefig()
        plt.close()

        ax = (
            plt.scatter(
                x=cluster_props_df["radial_distance"],
                y=np.linalg.norm(
                    np.stack(cluster_props_df["center_coords"].to_numpy())
                    - cluster_props_df[COMMON_ANCESTOR_COORDS_COLS].to_numpy(),
                    axis=1,
                ),
            )
            .get_figure()
            .gca()  # type: ignore[union-attr]
        )
        ax.set_title("Cluster radial length vs radial distance")
        pdf.savefig()
        plt.close()

        ax = cluster_props_df.plot.scatter(
            x="size",
            y="path_length",
            title="Path length vs cluster size",
            legend=True,
        )
        pdf.savefig()
        plt.close()

        ax = cluster_props_df.plot.scatter(
            x="path_distance",
            y="path_length",
            title="Path length vs path distance",
            legend=True,
        )
        pdf.savefig()
        plt.close()

        ax = cluster_props_df.plot.scatter(
            x="radial_distance",
            y="path_length",
            title="Path length vs radial distance",
            legend=True,
        )
        pdf.savefig()
        plt.close()
