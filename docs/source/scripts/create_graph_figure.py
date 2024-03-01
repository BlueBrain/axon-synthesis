"""This script creates the figure associated to the graph creation."""
import logging

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.constants import NodeProvider
from axon_synthesis.synthesis.main_trunk import create_graph
from axon_synthesis.synthesis.main_trunk import steiner_tree

logging.basicConfig(level=logging.DEBUG)

pd.options.display.max_rows = 100

create_graph.utils.FORCE_2D = True

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

nodes, edges = create_graph.one_graph(source_coords, target_points, config, rng=1)

# ############################################################## #
# Let's cheat a bit to ensure the solution is optimal for having #
# a more understandable figure.                                  #
edges.loc[[5, 38, 54, 58, 59], "weight"] = 1
# ############################################################## #

_, solution_edges = steiner_tree.compute_solution(
    nodes,
    edges,
)

edges["length"] = np.linalg.norm(
    edges[FROM_COORDS_COLS].to_numpy() - edges[TO_COORDS_COLS].to_numpy(),
    axis=1,
)


def plot(nodes, edges, figure_path, solution_edges=None):
    """Plot the given nodes and edges."""
    edges["cutter"] = None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.source.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.source.name, "y"],
            marker={"color": "black", "size": 20},
            mode="markers",
            name="Source point",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.target.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.target.name, "y"],
            marker={"color": "rgb(255,127,0)", "size": 20},
            mode="markers",
            name="Target points",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.intermediate.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.intermediate.name, "y"],
            marker={"color": "blue", "size": 15},
            mode="markers",
            name="Intermediate points",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.random.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.random.name, "y"],
            marker={"color": "green", "size": 15},
            mode="markers",
            name="Random points",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.bbox.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.bbox.name, "y"],
            marker={"color": "rgb(255,0,255)", "size": 15},
            mode="markers",
            name="Bounding box points",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nodes.loc[nodes["NodeProvider"] == NodeProvider.Voronoi.name, "x"],
            y=nodes.loc[nodes["NodeProvider"] == NodeProvider.Voronoi.name, "y"],
            marker={"color": "red", "size": 10},
            mode="markers",
            name="Voronoï points",
        )
    )

    if solution_edges is not None:
        solution_edges["cutter"] = None
        fig.add_trace(
            go.Scatter(
                x=solution_edges[["x_from", "x_to", "cutter"]].to_numpy().flatten().tolist(),
                y=solution_edges[["y_from", "y_to", "cutter"]].to_numpy().flatten().tolist(),
                line={"width": 3, "color": "red"},
                mode="lines",
                name="Steiner solution",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=edges[["x_from", "x_to", "cutter"]].to_numpy().flatten().tolist(),
            y=edges[["y_from", "y_to", "cutter"]].to_numpy().flatten().tolist(),
            line={"width": 1, "color": "black"},
            mode="lines",
            name="Steiner graph",
            hoverinfo="text",
        )
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
        )
    )

    fig.update_layout(
        {
            "autosize": True,
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "xaxis": {
                "visible": False,
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            "yaxis": {
                "visible": False,
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            "legend": {
                "xanchor": "left",
                "x": 0.9,
                "yanchor": "middle",
                "y": 0.5,
            },
        }
    )
    fig.write_html(figure_path + ".html")
    fig.write_image(figure_path + ".png", scale=2)
    fig.write_image(figure_path + ".svg", scale=2)


plot(nodes, edges, "graph_creation")
plot(nodes, edges, "graph_creation_solution", solution_edges)

# ruff: noqa: T201
print("Nodes:")
print(nodes)
print("Edges:")
print(edges)
print("Solution total length:", edges.loc[edges["is_solution"], "length"].sum())
