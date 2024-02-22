"""Compute the Steiner Tree.

The solution is computed using the package pcst_fast: https://github.com/fraenkel-lab/pcst_fast
"""
import logging

import pandas as pd
import pcst_fast as pf

from axon_synthesis.typing import FileType
from axon_synthesis.utils import sublogger


def compute_solution(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    output_path: FileType | None = None,
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
        edges[["from", "to"]].values,
        prizes,
        edges["weight"].values,
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

    if output_path is not None:
        # Export the solutions
        nodes.to_hdf(str(output_path), "solution_nodes", mode="w")
        edges.to_hdf(str(output_path), "solution_edges", mode="a")

    in_solution_nodes = nodes.loc[nodes["is_solution"]]
    in_solution_edges = edges.loc[edges["is_solution"]]

    logger.debug(
        "The solution contains %s among %s nodes and %s among %s edges",
        len(nodes),
        len(in_solution_nodes),
        len(edges),
        len(in_solution_edges),
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

    return in_solution_nodes, in_solution_edges
