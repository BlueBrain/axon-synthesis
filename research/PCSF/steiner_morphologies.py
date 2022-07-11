"""Create morphologies from the Steiner Tree solutions."""
import logging
from pathlib import Path

import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from luigi_tools.parameter import OptionalStrParameter
from luigi_tools.parameter import PathParameter
from morphio import PointLevel
from morphio import SectionType
from morphio.mut import Morphology as MutableMorphology
from neurom import load_morphology
from neurom.core import Morphology

from PCSF.create_graph import CreateGraph
from PCSF.steiner_tree import SteinerTree

logger = logging.getLogger(__name__)


class SteinerMorphologies(luigi_tools.task.WorkflowTask):
    nodes_path = OptionalStrParameter(description="Path to the nodes CSV file.", default=None)
    edges_path = OptionalStrParameter(description="Path to the edges CSV file.", default=None)
    somata_path = OptionalStrParameter(description="Path to the somata CSV file.", default=None)
    smoothing = OptionalStrParameter(
        description="Path to the edges CSV file.",
        default=None,
    )
    output_dir = PathParameter(
        description="Output folder for figures.",
        default="steiner_morphologies",
    )

    def requires(self):
        return {
            "steiner_tree": SteinerTree(),
            "terminals": CreateGraph(),
        }

    def run(self):
        nodes = pd.read_csv(self.nodes_path or self.input()["steiner_tree"]["nodes"].path)
        edges = pd.read_csv(self.edges_path or self.input()["steiner_tree"]["edges"].path)

        # import pdb
        # pdb.set_trace()
        somata = pd.read_csv(self.somata_path or self.input()["terminals"]["input_terminals"].path)
        soma_centers = somata.loc[somata["axon_id"] == -1].copy()

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
            morph = MutableMorphology()
            morph.soma.points = soma_centers.loc[soma_centers["morph_file"] == group_name, ["x", "y", "z"]].values
            morph.soma.diameters = [2]  # So the radius is 1
            morph = Morphology(morph)

            # Create the synthesized axon
            active_sections = []
            already_added = []
            roots = in_solution_edges.loc[in_solution_edges["from"] == 0]
            root_point = np.array(roots[["x_from", "y_from", "z_from"]].values[0])
            root_section_vec = root_point - morph.soma.center
            root_section_point = (
                morph.soma.center
                + root_section_vec / np.linalg.norm(root_section_vec) * max(
                    1,
                    min(morph.soma.radius, np.linalg.norm(root_section_vec) - 1)
                )
            )
            root_section = morph.append_root_section(
                PointLevel(
                    [
                        root_section_point,
                        root_point,
                    ],
                    [0, 0],
                ),
                SectionType.axon,
            )

            for row in in_solution_edges.loc[in_solution_edges["from"] == 0].iterrows():
                already_added.append(row[0])
                active_sections.append(
                    (
                        root_section.append_section(
                            PointLevel(
                                [
                                    row[1][["x_from", "y_from", "z_from"]].values,
                                    row[1][["x_to", "y_to", "z_to"]].values,
                                ],
                                [0, 0],
                            ),
                            SectionType.axon,
                        ),
                        row[1]["to"],
                    )
                )

            while active_sections:
                current_section, target = active_sections.pop()
                in_solution_edges = group_edges.loc[
                    (group_edges["is_solution"]) & (~group_edges.index.isin(already_added))
                ]
                for row in in_solution_edges.loc[in_solution_edges["from"] == target].iterrows():
                    already_added.append(row[0])
                    active_sections.append(
                        (
                            current_section.append_section(
                                PointLevel(
                                    [
                                        row[1][["x_from", "y_from", "z_from"]].values,
                                        row[1][["x_to", "y_to", "z_to"]].values,
                                    ],
                                    [0, 0],
                                ),
                                SectionType.axon,
                            ),
                            row[1]["to"],
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
                                    ],
                                    [0, 0],
                                ),
                                SectionType.axon,
                            ),
                            row[1]["from"],
                        )
                    )

            # Merge consecutive sections that are not separated by a bifurcation
            # TODO: Move it at the very end of the process
            # morph.remove_unifurcations()

            # Export the morphology
            morph_name = Path(str(group_name)).name
            morph_path = str((self.output().pathlib_path / morph_name).with_suffix(".asc"))
            morph.write(morph_path)

            logger.info(f"{morph_name}: exported to {morph_path}")

    def output(self):
        return TaggedOutputLocalTarget(self.output_dir, create_parent=True)
