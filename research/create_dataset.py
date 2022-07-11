"""Create the dataset that can be used for the Curate workflow from MPoW.

The workflow should be called using the luigi.cfg file from this directory and
"morphology-processing-workflow==0.0.5".
"""
from pathlib import Path

import luigi
import luigi_tools
import pandas as pd


class CreateDatasetForRepair(luigi_tools.task.WorkflowTask):
    morph_dir = luigi.Parameter(description="Folder containing the input morphologies.")
    output_dataset = luigi.Parameter(description="Output dataset file", default="dataset.csv")

    def run(self):
        morph_dir = Path(self.morph_dir)
        dataset_file = Path(self.output().path)
        dataset_file.parent.mkdir(parents=True, exist_ok=True)

        dataset = pd.DataFrame(columns=["morph_path", "mtype"])
        dataset.index.name = "morph_name"

        for morph in morph_dir.iterdir():
            if morph.suffix.lower() in [".asc", ".h5", ".swc"]:
                dataset.loc[morph.stem, "morph_path"] = morph
                dataset.loc[morph.stem, "mtype"] = "UNKOWN"
        dataset.sort_index(inplace=True)
        dataset.reset_index().to_csv(dataset_file, index=False)
        return dataset

    def output(self):
        return luigi.LocalTarget(self.output_dataset)
