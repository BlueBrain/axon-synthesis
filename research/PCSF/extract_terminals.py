"""Extract the terminal points of a morphology so that a Steiner Tree can be computed on them."""
from pathlib import Path

import luigi
import luigi_tools
import neurom
import numpy as np
import pandas as pd
from neurom import load_neuron


class ExtractTerminals(luigi_tools.task.WorkflowTask):
    morph_dir = luigi.Parameter(description="Folder containing the input morphologies.")
    output_dataset = luigi.Parameter(description="Output dataset file", default="terminals.csv")

    def run(self):
        morph_dir = Path(self.morph_dir)
        dataset_file = Path(self.output().path)
        dataset_file.parent.mkdir(parents=True, exist_ok=True)

        pts = []
        for morph_path in morph_dir.iterdir():

            neuron = load_neuron(morph_path)

            axons = [i for i in neuron.neurites if i.type == neurom.NeuriteType.axon]

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
        return luigi.LocalTarget(self.output_dataset)
