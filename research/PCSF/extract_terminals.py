"""Extract the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import logging
from pathlib import Path

import luigi
import luigi_tools
import neurom
import pandas as pd
from morphology_processing_workflow.tasks.workflows import Curate
from neurom import load_neuron

from create_dataset import CreateDatasetForRepair

logger = logging.getLogger(__name__)


class ExtractTerminals(luigi_tools.task.WorkflowTask):
    morph_dir = luigi.Parameter(description="Folder containing the input morphologies.")
    output_dataset = luigi.Parameter(description="Output dataset file", default="terminals.csv")

    def requires(self):
        dataset = CreateDatasetForRepair()
        return [dataset, Curate(dataset_df=dataset.output().path)]

    def run(self):
        morph_dir = Path(
            self.morph_dir or self.input()[1]["data"].get_default_prefix() / "CheckNeurites/data/"
        )
        dataset_file = Path(self.output().path)
        dataset_file.parent.mkdir(parents=True, exist_ok=True)

        pts = []
        for morph_path in morph_dir.iterdir():

            neuron = load_neuron(morph_path)

            axons = [i for i in neuron.neurites if i.type == neurom.NeuriteType.axon]

            nb_axons = len(axons)
            if nb_axons != 1:
                logger_func = logger.warning
            else:
                logger_func = logger.debug

            logger_func(f"{morph_path}: {nb_axons} axon(s) found")

            for axon_id, axon in enumerate(axons):
                # Add root point
                pts.append([morph_path, axon_id, 0] + axon.root_node.points[0][:3].tolist())

                # Add terminal points
                terminal_id = 1
                for section in axon.iter_sections():
                    if not section.children:
                        pts.append(
                            [morph_path, axon_id, terminal_id] + section.points[-1][:3].tolist()
                        )
                        terminal_id += 1

        dataset = pd.DataFrame(pts, columns=["morph_file", "axon_id", "terminal_id", "x", "y", "z"])

        dataset.to_csv(dataset_file, index=False)

    def output(self):
        return luigi_tools.target.OutputLocalTarget(self.output_dataset)
