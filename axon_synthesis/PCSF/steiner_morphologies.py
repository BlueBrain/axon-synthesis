"""Create morphologies from the Steiner Tree solutions."""
import logging
from pathlib import Path

import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from luigi.parameter import OptionalStrParameter
from luigi.parameter import PathParameter
from morphio import PointLevel
from morphio import SectionType
from morphio.mut import Morphology as MutableMorphology
from neurom.core import Morphology

from axon_synthesis.PCSF.create_graph import CreateGraph
from axon_synthesis.PCSF.steiner_tree import SteinerTree

logger = logging.getLogger(__name__)


class SteinerMorphologies(luigi_tools.task.WorkflowTask):
    """Task to create morphologies from Steiner solutions."""

    nodes_path = OptionalStrParameter(description="Path to the nodes CSV file.", default=None)
    edges_path = OptionalStrParameter(description="Path to the edges CSV file.", default=None)
    somata_path = OptionalStrParameter(description="Path to the somata CSV file.", default=None)
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

        somata = pd.read_csv(self.somata_path or self.input()["terminals"]["input_terminals"].path)
        soma_centers = somata.loc[somata["axon_id"] == -1].copy()

        self.output()["morphologies"].mkdir(is_dir=True)

        node_groups = nodes.groupby("morph_file")
        edge_groups = edges.groupby("morph_file")

        group_names = node_groups.groups.keys()
        assert set(group_names) == set(
            edge_groups.groups.keys()
        ), "The nodes and edges have different 'morph_file' entries"

        # Create an empty column for future file locations
        nodes["steiner_morph_file"] = None

        morph_paths = []

        for group_name in group_names:
            group_nodes = node_groups.get_group(group_name)
            group_edges = edge_groups.get_group(group_name)
            in_solution_nodes = group_nodes.loc[group_nodes["is_solution"]]
            in_solution_edges = group_edges.loc[group_edges["is_solution"]]

            logger.debug(
                "%s: %s on %s nodes in solution and %s on %s edges in solution",
                group_name,
                len(in_solution_nodes),
                len(group_nodes),
                len(in_solution_edges),
                len(group_edges),
            )

            # Load the biological neuron
            morph = MutableMorphology()
            morph.soma.points = soma_centers.loc[
                soma_centers["morph_file"] == group_name, ["x", "y", "z"]
            ].values
            morph.soma.diameters = [2]  # So the radius is 1
            morph = Morphology(morph)

            # Create the synthesized axon
            active_sections = []
            already_added = []
            roots = in_solution_edges.loc[in_solution_edges["from"] == 0]
            root_point = np.array(roots[["x_from", "y_from", "z_from"]].values[0])
            root_section_vec = root_point - morph.soma.center
            root_section_point = morph.soma.center + root_section_vec / np.linalg.norm(
                root_section_vec
            ) * max(1, min(morph.soma.radius, np.linalg.norm(root_section_vec) - 1))
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

            # At this point we do not merge consecutive sections that are not separated by a
            # bifurcation, we do it only at the very end of the process. This is to keep section
            # IDs synced with point IDs.

            # Export the morphology
            morph_name = Path(str(group_name)).name
            morph_path = str(
                (self.output()["morphologies"].pathlib_path / morph_name).with_suffix(".asc")
            )
            morph.write(morph_path)

            logger.info("%s: exported to %s", morph_name, morph_path)

            morph_paths.append((str(group_name), morph_path))

            # Set the path of the new morph in the node DF
            nodes.loc[nodes["morph_file"] == group_name, "steiner_morph_file"] = morph_path

        # Export the node DF
        nodes.to_csv(self.output()["nodes"].path, index=False)

        # Export the morph path DF
        pd.DataFrame(morph_paths, columns=["morph_file", "steiner_morph_file"]).to_csv(
            self.output()["morphology_paths"].path, index=False
        )

    def output(self):
        return {
            "nodes": TaggedOutputLocalTarget(
                self.output_dir / "steiner_morph_nodes.csv", create_parent=True
            ),
            "morphologies": TaggedOutputLocalTarget(self.output_dir, create_parent=True),
            "morphology_paths": TaggedOutputLocalTarget(
                self.output_dir / "steiner_morph_paths.csv", create_parent=True
            ),
        }
