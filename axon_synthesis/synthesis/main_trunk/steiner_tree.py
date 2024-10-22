"""Compute the Steiner Tree.

The solution is computed using the package pcst_fast: https://github.com/fraenkel-lab/pcst_fast
"""

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

import pandas as pd
import pcst_fast as pf
import plotly.graph_objs as go

from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.utils import build_layout_properties
from axon_synthesis.utils import sublogger


def plot_solution(nodes, edges, figure_path, logger=None):
    """Plot the given triangulation for debugging purpose."""
    segments = edges.copy(deep=False)
    segments["cutter"] = None

    segments["width"] = segments["weight"] / segments["length"]

    fig = go.Figure()

    edges_trace = go.Scatter3d(
        x=segments.loc[~segments["is_solution"], [FROM_COORDS_COLS.X, TO_COORDS_COLS.X, "cutter"]]
        .to_numpy()
        .flatten(),
        y=segments.loc[~segments["is_solution"], [FROM_COORDS_COLS.Y, TO_COORDS_COLS.Y, "cutter"]]
        .to_numpy()
        .flatten(),
        z=segments.loc[~segments["is_solution"], [FROM_COORDS_COLS.Z, TO_COORDS_COLS.Z, "cutter"]]
        .to_numpy()
        .flatten(),
        line={"width": 1, "color": "black"},
        mode="lines",
        name="Steiner graph",
    )

    solution_trace = go.Scatter3d(
        x=segments.loc[segments["is_solution"], [FROM_COORDS_COLS.X, TO_COORDS_COLS.X, "cutter"]]
        .to_numpy()
        .flatten(),
        y=segments.loc[segments["is_solution"], [FROM_COORDS_COLS.Y, TO_COORDS_COLS.Y, "cutter"]]
        .to_numpy()
        .flatten(),
        z=segments.loc[segments["is_solution"], [FROM_COORDS_COLS.Z, TO_COORDS_COLS.Z, "cutter"]]
        .to_numpy()
        .flatten(),
        line={"width": 5, "color": "red"},
        mode="lines",
        name="Steiner solution",
    )

    edge_weights_trace = go.Scatter3d(
        x=0.5 * (segments[FROM_COORDS_COLS.X] + segments[TO_COORDS_COLS.X]).to_numpy(),
        y=0.5 * (segments[FROM_COORDS_COLS.Y] + segments[TO_COORDS_COLS.Y]).to_numpy(),
        z=0.5 * (segments[FROM_COORDS_COLS.Z] + segments[TO_COORDS_COLS.Z]).to_numpy(),
        mode="markers",
        opacity=0,
        showlegend=False,
        hovertemplate="Weight of edge %{hovertext}<extra></extra>",
        hovertext=[
            f"{idx} = {val['weight']} (length ratio = {val['width']})"
            for idx, val in segments[["weight", "width"]].round(2).to_dict("index").items()
        ],
        marker={"size": 4},
    )

    source_point_trace = go.Scatter3d(
        x=[nodes.loc[0, COORDS_COLS.X]],
        y=[nodes.loc[0, COORDS_COLS.Y]],
        z=[nodes.loc[0, COORDS_COLS.Z]],
        marker={"color": "purple", "size": 4},
        mode="markers",
        name="Source point",
    )

    intermediate_points_trace = go.Scatter3d(
        x=nodes.loc[~nodes["is_terminal"], COORDS_COLS.X].to_numpy(),
        y=nodes.loc[~nodes["is_terminal"], COORDS_COLS.Y].to_numpy(),
        z=nodes.loc[~nodes["is_terminal"], COORDS_COLS.Z].to_numpy(),
        marker={"color": "green", "size": 2},
        mode="markers",
        name="Intermediate points",
    )

    target_points_trace = go.Scatter3d(
        x=nodes.loc[nodes["is_terminal"], COORDS_COLS.X].to_numpy()[1:],
        y=nodes.loc[nodes["is_terminal"], COORDS_COLS.Y].to_numpy()[1:],
        z=nodes.loc[nodes["is_terminal"], COORDS_COLS.Z].to_numpy()[1:],
        marker={"color": "chocolate", "size": 4},
        mode="markers",
        name="Target points",
    )

    fig.add_trace(edges_trace)
    fig.add_trace(solution_trace)
    fig.add_trace(edge_weights_trace)
    fig.add_trace(intermediate_points_trace)
    fig.add_trace(source_point_trace)
    fig.add_trace(target_points_trace)

    layout_props = build_layout_properties(nodes[COORDS_COLS].to_numpy(), 0.1)

    fig.update_scenes(layout_props)
    fig.update_layout(title=Path(figure_path).stem)

    # Export figure
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(figure_path)

    if logger is not None:
        logger.debug("Exported Steiner Tree solution figure to %s", figure_path)


def compute_solution(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    output_path_nodes: FileType | None = None,
    output_path_edges: FileType | None = None,
    figure_path: FileType | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Compute the Steiner Tree solution from the given nodes and edges."""
    logger = sublogger(logger, __name__)

    nodes["is_solution"] = False
    edges["is_solution"] = False

    logger.debug(
        "%s nodes and %s edges",
        len(nodes),
        len(edges),
    )

    # Prepare prizes: we want to connect all terminals so we give them an 'infinite' prize
    prizes = 100.0 * nodes["is_terminal"] * edges["weight"].sum()

    # Compute Steiner Tree
    solution_nodes, solution_edges = pf.pcst_fast(  # pylint: disable=c-extension-no-member
        edges[["from", "to"]].to_numpy(),
        prizes,
        edges["weight"].to_numpy(),
        -1,
        1,
        "gw",
        0,
    )

    nodes.loc[
        (nodes["id"].isin(solution_nodes)),
        "is_solution",
    ] = True

    group_edge_ids = edges.reset_index()["index"]
    edge_ids = pd.Series(-1, index=edges.index)
    reverted_group_edge_ids = pd.Series(
        group_edge_ids.index.to_numpy(), index=group_edge_ids.to_numpy()
    )
    edge_ids.loc[reverted_group_edge_ids.index] = reverted_group_edge_ids
    edges.loc[
        (edge_ids.isin(solution_edges)),
        "is_solution",
    ] = True

    if output_path_nodes is not None:
        # Export the solution nodes
        nodes.to_feather(str(output_path_nodes))
    if output_path_edges is not None:
        # Export the solution edges
        edges.to_feather(str(output_path_edges))

    in_solution_nodes = nodes.loc[nodes["is_solution"]]
    in_solution_edges = edges.loc[edges["is_solution"]]

    logger.debug(
        "The solution contains %s among %s nodes and %s among %s edges",
        len(in_solution_nodes),
        len(nodes),
        len(in_solution_edges),
        len(edges),
    )

    # Add node data to solution edges
    in_solution_edges = in_solution_edges.merge(
        in_solution_nodes[["terminal_id", "is_terminal"]].rename(
            columns={"terminal_id": "source_terminal_id", "is_terminal": "source_is_terminal"}
        ),
        left_on="from",
        right_index=True,
        how="left",
    )
    in_solution_edges = in_solution_edges.merge(
        in_solution_nodes[["terminal_id", "is_terminal"]].rename(
            columns={"terminal_id": "target_terminal_id", "is_terminal": "target_is_terminal"}
        ),
        left_on="to",
        right_index=True,
        how="left",
    )

    if figure_path is not None:
        plot_solution(
            nodes,
            edges,
            figure_path,
            logger=logger,
        )

    return in_solution_nodes, in_solution_edges
