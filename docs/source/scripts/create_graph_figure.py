"""This script creates the figure associated to the graph creation."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from attrs import evolve
from scipy.spatial import KDTree

from axon_synthesis.constants import NodeProvider
from axon_synthesis.synthesis.main_trunk import create_graph
from axon_synthesis.synthesis.main_trunk import steiner_tree

logging.basicConfig(level=logging.DEBUG)

pd.options.display.max_rows = 100

create_graph.utils.FORCE_2D = True


def num_to_rgb(val, min_val=0, max_val=1):
    """Convert a float value to a RGB value for a rainbow color theme."""
    i = (val - min_val) * 255 / (max_val - min_val)
    coeff = np.pi / 255
    r = round(np.sin(coeff * (i - 127)) * 127 + 128)
    g = round(np.sin(coeff * (i - 63) * 2) * 127 + 128)
    b = round(np.sin(coeff * (i + 127)) * 127 + 128)

    return f"rgb({r},{g},{b})"


def degree2rad(degrees):
    """Convert degrees into radians."""
    return degrees * np.pi / 180


def disk_part(center=None, radius=1, start_angle=0, end_angle=90, n=50):
    """Create a path string for a circle arc."""
    if center is None:
        center = [0, 0]
    sta = degree2rad(start_angle)
    ea = degree2rad(end_angle)
    t = np.linspace(sta, ea, n)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    path = f"M {x[0]},{y[0]}"
    for xc, yc in zip(x[1:], y[1:]):
        path += f" L{xc},{yc}"
    return path


def plot(nodes, edges, figure_path, solution_edges=None, preferred_regions_pts=None):
    """Plot the given nodes and edges."""
    edges = edges.copy(deep=False)
    edges["cutter"] = None
    edges["density"] = edges["weight"] / edges["length"]

    fig = go.Figure()
    annotations = []

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.source.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.source.name, "y"],
            marker={"color": "black", "size": 20},
            mode="markers",
            name="Source point",
            legendgroup=1,
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.target.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.target.name, "y"],
            marker={"color": "rgb(255,127,0)", "size": 20},
            mode="markers",
            name="Target points",
            legendgroup=2,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.intermediate.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.intermediate.name, "y"],
            marker={"color": "blue", "size": 15},
            mode="markers",
            name="Intermediate points",
            legendgroup=3,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.random.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.random.name, "y"],
            marker={"color": "green", "size": 15},
            mode="markers",
            name="Random points",
            legendgroup=4,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.bbox.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.bbox.name, "y"],
            marker={"color": "rgb(255,0,255)", "size": 15},
            mode="markers",
            name="Bounding box points",
            legendgroup=5,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.Voronoi.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.Voronoi.name, "y"],
            marker={"color": "red", "size": 10},
            mode="markers",
            name="Voronoï points",
            legendgroup=6,
        )
    )

    if solution_edges is not None:
        solution_edges["cutter"] = None
        fig.add_trace(
            go.Scatter(
                x=solution_edges[["x_from", "x_to", "cutter"]].to_numpy().flatten().tolist(),
                y=solution_edges[["y_from", "y_to", "cutter"]].to_numpy().flatten().tolist(),
                line={
                    "width": 6,
                    "color": "red",
                    "dash": "dash",
                },
                mode="lines",
                name="Steiner solution",
                legendgroup=7,
            )
        )

    if preferred_regions_pts is None:
        fig.add_trace(
            go.Scatter(
                x=edges[["x_from", "x_to", "cutter"]].to_numpy().flatten().tolist(),
                y=edges[["y_from", "y_to", "cutter"]].to_numpy().flatten().tolist(),
                line={"width": 1, "color": "black"},
                mode="lines",
                name="Steiner graph edges",
                hoverinfo="text",
                legendgroup=8,
            )
        )
    else:
        edges["normalized_density"] = (edges["density"] - edges["density"].min()) / (
            edges["density"].max() - edges["density"].min()
        )
        edges["rgb_color"] = edges["normalized_density"].apply(num_to_rgb)

        for _, row in edges.iterrows():
            x = [row["x_from"], row["x_to"]]
            y = [row["y_from"], row["y_to"]]
            color = row["rgb_color"]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    line={
                        "width": 2,
                        "color": color,
                    },
                    mode="lines",
                    name="Steiner graph",
                    showlegend=False,
                )
            )

        # Add empty plot just to add the color bar and the legend entry
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                line={
                    "width": 2,
                    "color": "black",
                },
                mode="lines",
                name="Steiner graph edges",
                legendgroup=8,
            ),
        )
        annotations.append(
            {
                "text": "Linear density of edge weights",
                "font": {"size": 16, "family": "computer modern", "color": "black"},
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 1.23,
            },
        )
        fig.add_trace(
            go.Scatter(
                x=edges["x_from"],
                y=edges["y_from"],
                mode="markers",
                marker={
                    "opacity": 0,
                    "color": edges["normalized_density"],
                    "colorbar": {
                        "orientation": "h",
                        "thickness": 15,
                        "x": 0.5,
                        "y": 1,
                        "len": 0.75,
                        "ticklen": 10,
                        "tickfont": {"size": 16, "family": "computer modern", "color": "black"},
                    },
                    "colorscale": [
                        (0, "rgb(0, 0, 255)"),
                        (0.5, "rgb(127, 255, 127)"),
                        (1, "rgb(255, 0, 0)"),
                    ],
                },
                name="Steiner graph",
                showlegend=False,
                legendgroup=8,
            )
        )

        # Add the preferred region points
        for i in preferred_regions_pts:
            name = f": {i[3]}" if len(i) >= 4 and i[3] is not None else ""
            print("Add preferred region point", i)
            fig.add_trace(
                go.Scatter(
                    x=[i[0]],
                    y=[i[1]],
                    marker={
                        "color": "maroon",
                        "size": 15,
                    },
                    mode="markers",
                    name=f"Preferred region point{name}",
                    legendgroup=11,
                ),
            )
            for radius in range(0, 240, 50):
                # Add radial distances (this part of the code in not generic)
                fig.add_shape(
                    type="path",
                    xref="x",
                    yref="y",
                    path=disk_part([i[0], i[1]], radius=radius, start_angle=-90, end_angle=0),
                    x0=i[0] - radius,
                    y0=i[1] - radius,
                    x1=i[0] + radius,
                    y1=i[1] + radius,
                    line={"color": "grey", "width": 3, "dash": "dash"},
                )
            fig.add_shape(
                type="path",
                xref="x",
                yref="y",
                path=disk_part([i[0], i[1]], radius=250, start_angle=-53, end_angle=-37),
                x0=i[0] - radius,
                y0=i[1] - radius,
                x1=i[0] + radius,
                y1=i[1] + radius,
                line={"color": "grey", "width": 3, "dash": "dash"},
            )

    fig.add_trace(
        go.Scatter(
            x=0.5 * (edges["x_from"] + edges["x_to"]),
            y=0.5 * (edges["y_from"] + edges["y_to"]),
            mode="markers",
            showlegend=False,
            hovertemplate="Weight of edge %{hovertext}<extra></extra>",
            hovertext=[
                f"{idx} = {weight}" for idx, weight in edges["weight"].round(2).to_dict().items()
            ],
            marker=go.scatter.Marker(opacity=0),
            legendgroup=9,
        )
    )

    fig.update_layout(
        {
            "width": 600,
            "height": 500,
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0, "autoexpand": True, "pad": 0},
            "paper_bgcolor": "rgba(255, 255, 255, 255)",
            "plot_bgcolor": "rgba(255, 255, 255, 255)",
            "xaxis": {
                "scaleanchor": "x",
                "scaleratio": 1,
                "showgrid": False,
                "visible": False,
            },
            "yaxis": {
                "scaleanchor": "x",
                "scaleratio": 1,
                "showgrid": False,
                "visible": False,
            },
            "legend": {
                "xanchor": "left",
                "x": 0.95,
                "yanchor": "middle",
                "y": 0.5,
                "tracegroupgap": 20,
                "font": {"size": 16, "family": "computer modern", "color": "black"},
            },
        },
        annotations=annotations,
    )

    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(figure_path + ".html")
    fig.write_image(figure_path + ".png", scale=2)
    fig.write_image(figure_path + ".svg", scale=2)

    if preferred_regions_pts is None:
        fig.update_layout({"showlegend": False})
        fig.write_html(figure_path + "_no_legend.html")
        fig.write_image(figure_path + "_no_legend.png", scale=2)
        fig.write_image(figure_path + "_no_legend.svg", scale=2)


config = create_graph.CreateGraphConfig(
    intermediate_number=1,
    min_intermediate_distance=1,
    min_random_point_distance=50,
    random_point_bbox_buffer=0,
    voronoi_steps=1,
    use_orientation_penalty=False,
    use_depth_penalty=False,
    use_terminal_penalty=True,
)
source_coords = np.array([0, 0, 0])
target_points = pd.DataFrame(
    {
        "axon_id": [1, 1],
        "terminal_id": [0, 1],
        "target_x": [200, 200],
        "target_y": [-100, 100],
        "target_z": [0, 0],
    }
)
target_points = target_points.astype(
    dtype={"target_x": float, "target_y": float, "target_z": float}
)

# Case without preferred region
nodes, edges = create_graph.one_graph(source_coords, target_points, config, rng=1)

# ################################################################################ #
# Let's trick the approximation a bit to ensure the solution is optimal for having #
# a more understandable figure. This is required because the approximation can't   #
# find the optimal path in this specific case.                                     #
edges.loc[[5, 38, 54, 58, 59], "weight"] = 1
# ################################################################################ #

_, solution_edges = steiner_tree.compute_solution(
    nodes,
    edges,
)


plot(nodes, edges, "graph_creation")
plot(nodes, edges, "graph_creation_solution", solution_edges)

# ruff: noqa: T201
print("Nodes:")
print(nodes)
print("Edges:")
print(edges)
print("Solution total length:", edges.loc[edges["is_solution"], "length"].sum())


# Case with preferred region
points_preferred_regions = [[0, 100, 0]]
config.preferred_region_tree = KDTree(points_preferred_regions)
preferred_regions_config = evolve(
    config,
    preferring_sigma=100,
    preferring_amplitude=10,
    use_orientation_penalty=False,
    use_depth_penalty=False,
    use_terminal_penalty=False,
)

nodes_preferred_regions, edges_preferred_regions = create_graph.one_graph(
    source_coords, target_points, preferred_regions_config, rng=1
)

_, solution_edges_preferred_regions = steiner_tree.compute_solution(
    nodes_preferred_regions,
    edges_preferred_regions,
)

plot(
    nodes_preferred_regions,
    edges_preferred_regions,
    "graph_creation_preferred_regions",
    preferred_regions_pts=points_preferred_regions,
)
plot(
    nodes_preferred_regions,
    edges_preferred_regions,
    "graph_creation_solution_preferred_regions",
    solution_edges_preferred_regions,
    preferred_regions_pts=points_preferred_regions,
)
