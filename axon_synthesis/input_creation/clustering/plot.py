"""Some plot utils for clustering."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool import resampling
from neurom.core import Morphology
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import disable_loggers


def plot_clusters(morph, clustered_morph, group, group_name, cluster_df, output_path):
    """Plot clusters to a HTML figure."""
    with disable_loggers("matplotlib.font_manager"):
        plotted_morph = Morphology(
            resampling.resample_linear_density(morph, 0.005),
            name=Path(group_name).with_suffix("").name,
        )
        fig_builder = NeuronBuilder(
            plotted_morph, "3d", line_width=4, title=f"{plotted_morph.name}"
        )

        x, y, z = group[["x", "y", "z"]].values.T
        node_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={"size": 3, "color": "black"},
            name="Morphology nodes",
        )
        x, y, z = cluster_df[["x", "y", "z"]].values.T
        cluster_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={"size": 5, "color": "red"},
            name="Cluster centers",
        )
        cluster_lines = [
            [
                [
                    i["x"],
                    cluster_df.loc[cluster_df["terminal_id"] == i["cluster_id"], "x"].iloc[0],
                    None,
                ],
                [
                    i["y"],
                    cluster_df.loc[cluster_df["terminal_id"] == i["cluster_id"], "y"].iloc[0],
                    None,
                ],
                [
                    i["z"],
                    cluster_df.loc[cluster_df["terminal_id"] == i["cluster_id"], "z"].iloc[0],
                    None,
                ],
            ]
            for i in group.to_dict("records")
            if i["cluster_id"] >= 0
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
        fig.add_trace(cluster_trace, row=1, col=2)

        fig.update_scenes({"aspectmode": "data"})

        # Export figure
        fig.write_html(str(output_path))

        add_camera_sync(str(output_path))


def plot_cluster_properties(cluster_props_df, output_path):
    """Plot the cluster properties to a PDF figure."""
    if cluster_props_df.empty:
        return

    with disable_loggers(
        "matplotlib.font_manager", "matplotlib.backends.backend_pdf", "matplotlib.ticker"
    ):
        with PdfPages(str(output_path)) as pdf:
            ax = cluster_props_df.plot.scatter(
                x="path_distance",
                y="cluster_size",
                title="Cluster size vs path distance",
                legend=True,
            )
            ax.set_yscale("log")
            pdf.savefig()
            plt.close()

            ax = cluster_props_df.plot.scatter(
                x="radial_distance",
                y="cluster_size",
                title="Cluster size vs radial distance",
                legend=True,
            )
            ax.set_yscale("log")
            pdf.savefig()
            plt.close()

            ax = (
                plt.scatter(
                    x=cluster_props_df["radial_distance"],
                    y=(
                        cluster_props_df["cluster_center_coords"].apply(np.array)
                        - cluster_props_df["common_ancestor_coords"]
                    ).apply(np.linalg.norm),
                )
                .get_figure()
                .gca()
            )
            ax.set_title("Cluster radial length vs radial distance")
            pdf.savefig()
            plt.close()

            ax = cluster_props_df.plot.scatter(
                x="cluster_size",
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
