"""Create the dataset that can be used for the Curate workflow from MPoW.

The workflow should be called using the luigi.cfg file from this directory and
"morphology-processing-workflow==0.0.5".
"""
from pathlib import Path

import luigi
import luigi_tools
import pandas as pd
from morphology_processing_workflow.tasks.workflows import Curate


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
        return luigi_tools.target.OutputLocalTarget(self.output_dataset, create_parent=True)


class RepairDataset(luigi_tools.task.WorkflowWrapperTask):

    def requires(self):
        dataset = CreateDatasetForRepair()
        return [dataset, Curate(dataset_df=dataset.output().path)]

    def output(self):
        return luigi_tools.target.OutputLocalTarget(
            self.input()[1]["data"].pathlib_path.resolve().parent.parent / "Resample/data/"
        )
