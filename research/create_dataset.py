"""Create the dataset that can be used for the Curate workflow from MPoW.

The workflow should be called using the luigi.cfg file from this directory and
"morphology-workflows==0.2.0".
"""
from pathlib import Path

import luigi
import luigi_tools
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from morphology_workflows.tasks.workflows import Curate

from config import Config


class CreateDatasetForRepair(luigi_tools.task.WorkflowTask):
    morph_dir = luigi.Parameter(description="Folder containing the input morphologies.")
    output_dataset = luigi.Parameter(description="Output dataset file", default="dataset.csv")

    def run(self):
        morph_dir = Path(self.morph_dir)

        dataset = pd.DataFrame(columns=["morph_path", "mtype"])
        dataset.index.name = "morph_name"

        for morph in morph_dir.iterdir():
            if morph.suffix.lower() in [".asc", ".h5", ".swc"]:
                dataset.loc[morph.stem, "morph_path"] = morph
                dataset.loc[morph.stem, "mtype"] = "UNKOWN"
        dataset.sort_index(inplace=True)
        dataset.reset_index().to_csv(self.output().path, index=False)
        return dataset

    def output(self):
        return TaggedOutputLocalTarget(self.output_dataset, create_parent=True)


class RepairDataset(luigi_tools.task.WorkflowTask):
    def requires(self):
        dataset = CreateDatasetForRepair()
        return dataset

    def run(self):
        repair = yield Curate(dataset_df=self.input().path, result_path=Config().output_dir.absolute())

    def output(self):
        return TaggedOutputLocalTarget("Resample/data/")


class RawDataset(luigi_tools.task.WorkflowWrapperTask):
    def requires(self):
        return RepairDataset()

    def output(self):
        return TaggedOutputLocalTarget(
            self.input().pathlib_path.resolve().parent.parent / "Collect/data/"
        )
