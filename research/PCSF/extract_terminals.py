"""Extract the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import logging

import luigi
import luigi_tools
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from neurom import load_morphology

from create_dataset import RepairDataset
from utils import get_axons

logger = logging.getLogger(__name__)


class ExtractTerminals(luigi_tools.task.WorkflowTask):
    morph_dir = luigi_tools.parameter.OptionalPathParameter(
        description="Folder containing the input morphologies.",
        default=None,
    )
    output_dataset = luigi.Parameter(
        description="Output dataset file",
        default="input_terminals.csv",
    )

    def requires(self):
        return RepairDataset()

    def run(self):
        morph_dir = self.morph_dir or self.input().pathlib_path

        pts = []
        for morph_path in morph_dir.iterdir():

            morph = load_morphology(morph_path)

            # Add soma center as terminal
            pts.append([morph_path, -1, -1, -1] + morph.soma.center.tolist())

            axons = get_axons(morph)

            nb_axons = len(axons)
            if nb_axons != 1:
                logger_func = logger.warning
            else:
                logger_func = logger.debug

            logger_func(f"{morph_path}: {nb_axons} axon(s) found")

            for axon_id, axon in enumerate(axons):
                # Add root point
                pts.append(
                    [morph_path, axon_id, 0, axon.root_node.id]
                    + axon.root_node.points[0][:3].tolist()
                )

                # Add terminal points
                terminal_id = 1
                for section in axon.iter_sections():
                    if not section.children:
                        pts.append(
                            [morph_path, axon_id, terminal_id, section.id]
                            + section.points[-1][:3].tolist()
                        )
                        terminal_id += 1

        dataset = pd.DataFrame(
            pts, columns=["morph_file", "axon_id", "terminal_id", "section_id", "x", "y", "z"]
        )

        dataset.sort_values(["morph_file", "axon_id", "terminal_id"], inplace=True)
        dataset.to_csv(self.output().path, index=False)

    def output(self):
        return TaggedOutputLocalTarget(self.output_dataset, create_parent=True)
