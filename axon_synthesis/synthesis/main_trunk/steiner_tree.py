"""Compute the Steiner Tree.

The solution is computed using the package pcst_fast: https://github.com/fraenkel-lab/pcst_fast
"""
import logging
from pathlib import Path

import pandas as pd
import pcst_fast as pf

from axon_synthesis.typing import FileType
from axon_synthesis.utils import get_logger


def compute_solution(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    output_dir: FileType | None = None,
    logger_adapter: logging.LoggerAdapter | None = None,
):
    """Compute the Steiner Tree solution from the given nodes and edges."""
    logger = get_logger(__name__, logger_adapter)

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
    solution_nodes, solution_edges = pf.pcst_fast(
        edges[["from", "to"]].values,
        prizes,
        edges["weight"].values,
        -1,
        1,
        "gw",
        0,
    )

    logger.info("The solution has %s edges", len(solution_edges))

    nodes.loc[
        (nodes["id"].isin(solution_nodes)),
        "is_solution",
    ] = True

    group_edge_ids = edges.reset_index()["index"]
    edge_ids = pd.Series(-1, index=edges.index)
    reverted_group_edge_ids = pd.Series(group_edge_ids.index, index=group_edge_ids.values)
    edge_ids.loc[reverted_group_edge_ids.index] = reverted_group_edge_ids
    edges.loc[
        (edge_ids.isin(solution_edges)),
        "is_solution",
    ] = True

    if output_dir is not None:
        # Export the solutions
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        nodes.to_csv(output_dir / "nodes.csv", index=False)
        edges.to_csv(output_dir / "edges.csv", index=False)

    return nodes, edges
