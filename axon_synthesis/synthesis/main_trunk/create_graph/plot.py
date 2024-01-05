"""Some plot utils for create graph."""
from pathlib import Path

import plotly.graph_objs as go

from axon_synthesis.synthesis.main_trunk.create_graph.utils import FROM_COORDS_COLS
from axon_synthesis.synthesis.main_trunk.create_graph.utils import TO_COORDS_COLS


def plot_triangulation(edges, source_point, target_points, figure_path):
    """Plot the given triangulation for debugging purpose."""
    segments = edges.copy(deep=False)
    segments["cutter"] = None

    fig = go.Figure()

    edges_trace = go.Scatter3d(
        x=segments[[FROM_COORDS_COLS.X, TO_COORDS_COLS.X, "cutter"]].to_numpy().flatten().tolist(),
        y=segments[[FROM_COORDS_COLS.Y, TO_COORDS_COLS.Y, "cutter"]].to_numpy().flatten().tolist(),
        z=segments[[FROM_COORDS_COLS.Z, TO_COORDS_COLS.Z, "cutter"]].to_numpy().flatten().tolist(),
        line={"width": 0.5, "color": "#888"},
        mode="lines",
        name="Steiner graph",
    )

    source_point_trace = go.Scatter3d(
        x=[source_point[0]],
        y=[source_point[1]],
        z=[source_point[2]],
        marker={"color": "rgb(255,0,0)", "size": 4},
        name="Source point",
    )

    target_points_trace = go.Scatter3d(
        x=target_points[:, 0],
        y=target_points[:, 1],
        z=target_points[:, 2],
        marker={"color": "rgb(0,0,255)", "size": 2},
        name="Target points",
    )

    fig.add_trace(edges_trace)
    fig.add_trace(source_point_trace)
    fig.add_trace(target_points_trace)

    fig.update_scenes({"aspectmode": "data"})

    fig.layout.update(title=Path(figure_path).stem)

    # Export figure
    fig.write_html(figure_path)
