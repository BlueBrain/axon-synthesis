"""Create morphologies from the Steiner Tree solutions."""
import logging
from pathlib import Path

import luigi_tools
import pandas as pd
from luigi_tools.parameter import OptionalStrParameter
from luigi_tools.parameter import PathParameter
from neurom import load_morphology
from morphio import PointLevel
from morphio import SectionType

from PCSF.steiner_tree import SteinerTree

logger = logging.getLogger(__name__)


class SteinerMorphologies(luigi_tools.task.WorkflowTask):
    nodes_path = OptionalStrParameter(description="Path to the nodes CSV file.", default=None)
    edges_path = OptionalStrParameter(description="Path to the edges CSV file.", default=None)
    output_dir = PathParameter(
        description="Output folder for figures.",
        default="steiner_morphologies",
    )

    def requires(self):
        return SteinerTree()

    def run(self):
        nodes = pd.read_csv(self.nodes_path or self.input()["nodes"].path)
        edges = pd.read_csv(self.edges_path or self.input()["edges"].path)
        self.output().mkdir(is_dir=True)

        node_groups = nodes.groupby("morph_file")
        edge_groups = edges.groupby("morph_file")

        group_names = node_groups.groups.keys()
        assert set(group_names) == set(
            edge_groups.groups.keys()
        ), "The nodes and edges have different 'morph_file' entries"

        for group_name in group_names:
            group_nodes = node_groups.get_group(group_name)
            group_edges = edge_groups.get_group(group_name)
            in_solution_edges = group_edges.loc[group_edges["is_solution"]]

            logger.debug(f"{group_name}: {len(group_nodes)} nodes and {len(group_edges)} edges")

            # Load the biological neuron
            morph = load_morphology(group_name)

            # Remove all neurites and keep only soma
            root_sections = [i for i in morph.root_sections]
            for i in root_sections:
                morph.delete_section(i)

            # Create the synthesized axon
            active_sections = []
            already_added = []
            for row in in_solution_edges.loc[in_solution_edges["from"] == 0].iterrows():
                already_added.append(row[0])
                active_sections.append(
                    (
                        morph.append_root_section(
                            PointLevel(
                                [
                                    row[1][["x_from", "y_from", "z_from"]].values,
                                    row[1][["x_to", "y_to", "z_to"]].values,
                                ], [0, 0]),
                            SectionType.axon,
                        ),
                        row[1]["to"]
                    )
                )

            while active_sections:
                current_section, target = active_sections.pop()
                for row in in_solution_edges.loc[in_solution_edges["from"] == target].iterrows():
                    already_added.append(row[0])
                    active_sections.append(
                        (
                            current_section.append_section(
                                PointLevel(
                                    [
                                        row[1][["x_from", "y_from", "z_from"]].values,
                                        row[1][["x_to", "y_to", "z_to"]].values,
                                    ], [0, 0]),
                                SectionType.axon,
                            ),
                            row[1]["to"]
                        )
                    )
                for row in in_solution_edges.loc[in_solution_edges["to"] == target].iterrows():
                    already_added.append(row[0])
                    active_sections.append(
                        (
                            current_section.append_section(
                                PointLevel(
                                    [
                                        row[1][["x_to", "y_to", "z_to"]].values,
                                        row[1][["x_from", "y_from", "z_from"]].values,
                                    ], [0, 0]),
                                SectionType.axon,
                            ),
                            row[1]["from"]
                        )
                    )
                in_solution_edges = group_edges.loc[
                    (group_edges["is_solution"]) & (~group_edges.index.isin(already_added))
                ]

            # Merge consecutive sections that are not separated by a bifurcation
            morph.remove_unifurcations()

            # Export the morphology
            morph_name = Path(group_name).name
            morph_path = str((self.output().pathlib_path / morph_name).with_suffix(".asc"))
            morph.write(morph_path)

            logger.info(f"{morph_name}: exported to {morph_path}")

    def output(self):
        return luigi_tools.target.OutputLocalTarget(self.output_dir)
