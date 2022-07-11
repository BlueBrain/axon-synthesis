"""Compute the Steiner Tree.

The solution is computed using the package pcst_fast: https://github.com/fraenkel-lab/pcst_fast
"""
import logging

import luigi
import luigi_tools
import pandas as pd
import pcst_fast as pf
from data_validation_framework.target import TaggedOutputLocalTarget

from axon_synthesis.PCSF.create_graph import CreateGraph

logger = logging.getLogger(__name__)


class SteinerTree(luigi_tools.task.WorkflowTask):
    """Task to compute the Steiner solution of the given graph."""

    nodes_path = luigi.OptionalStrParameter(description="Path to the nodes CSV file.", default=None)
    edges_path = luigi.OptionalStrParameter(description="Path to the edges CSV file.", default=None)
    output_nodes = luigi.Parameter(description="Output nodes file.", default="solution_nodes.csv")
    output_edges = luigi.Parameter(description="Output edges file.", default="solution_edges.csv")

    def requires(self):
        return CreateGraph()

    def run(self):
        nodes = pd.read_csv(self.nodes_path or self.input()["nodes"].path)
        edges = pd.read_csv(self.edges_path or self.input()["edges"].path)

        nodes["is_solution"] = False
        edges["is_solution"] = False

        node_groups = nodes.groupby("morph_file")
        edge_groups = edges.groupby("morph_file")

        group_names = node_groups.groups.keys()
        assert set(group_names) == set(
            edge_groups.groups.keys()
        ), "The nodes and edges have different 'morph_file' entries"

        for group_name in group_names:
            group_nodes = node_groups.get_group(group_name)
            group_edges = edge_groups.get_group(group_name)

            logger.debug(
                "%s: %s nodes and %s edges",
                group_name,
                len(group_nodes),
                len(group_edges),
            )

            # Prepare prizes: we want to connect all terminals so we give them an 'infinite' prize
            prizes = 100.0 * group_nodes["is_terminal"] * group_edges["length"].sum()

            # Compute Stein Tree
            solution_nodes, solution_edges = pf.pcst_fast(
                group_edges[["from", "to"]].values,
                prizes,
                group_edges["length"].values,
                -1,
                1,
                "gw",
                0,
            )

            logger.info("%s: The solution has %s edges", group_name, len(solution_edges))

            nodes.loc[
                ((nodes["morph_file"] == group_name) & (nodes["id"].isin(solution_nodes))),
                "is_solution",
            ] = True

            group_edge_ids = group_edges.reset_index()["index"]
            edge_ids = pd.Series(-1, index=edges.index)
            reverted_group_edge_ids = pd.Series(group_edge_ids.index, index=group_edge_ids.values)
            edge_ids.loc[reverted_group_edge_ids.index] = reverted_group_edge_ids
            edges.loc[
                ((edges["morph_file"] == group_name) & (edge_ids.isin(solution_edges))),
                "is_solution",
            ] = True

        # Export the solutions
        nodes.to_csv(self.output()["nodes"].path, index=False)
        edges.to_csv(self.output()["edges"].path, index=False)

    def output(self):
        return {
            "nodes": TaggedOutputLocalTarget(self.output_nodes, create_parent=True),
            "edges": TaggedOutputLocalTarget(self.output_edges, create_parent=True),
        }
