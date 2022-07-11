"""Create the dataset that can be used for the Curate workflow from MPoW.

The workflow should be called using the luigi.cfg file from this directory and
"morphology-workflows==0.2.0".
"""
from pathlib import Path

import luigi
import luigi_tools
import pandas as pd
from config import Config
from data_validation_framework.target import TaggedOutputLocalTarget
from morphology_workflows.tasks.workflows import Curate

from white_matter_recipe import fetch as fetch_wmr


class CreateDatasetForRepair(luigi_tools.task.WorkflowTask):
    morph_dir = luigi.Parameter(description="Folder containing the input morphologies.")
    output_dataset = luigi.Parameter(
        description="Output dataset file", default="dataset.csv"
    )

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
        repair = yield Curate(
            dataset_df=self.input().path, result_path=Config().output_dir.absolute()
        )

    def output(self):
        return TaggedOutputLocalTarget("Resample/data/")


class RawDataset(luigi_tools.task.WorkflowWrapperTask):
    def requires(self):
        return RepairDataset()

    def output(self):
        return TaggedOutputLocalTarget(
            self.input().pathlib_path.resolve().parent.parent / "Collect/data/"
        )


class FetchWhiteMatterRecipe(luigi_tools.task.WorkflowTask):
    """Task to fetch the White Matter Recipe file from a repository."""
    url = luigi.Parameter(
        default=None,
        description=":str: Url of the repository.",
    )
    file_path = luigi.OptionalParameter(
        description=":str: Path of the file in the repository to fetch.",
        default="white_matter_FULL_RECIPE_v1p20.yaml",
    )
    version = luigi.OptionalParameter(
        description=":str: Version of the repository to checkout (use HEAD if not given).",
        default=None,
    )

    def run(self):
        target = self.output()
        if not target.pathlib_path.exists():
            # Note: this check should be useless because luigi calls the run() method only if the
            # target does not exist, but we keep it for safety.
            fetch_wmr(
                url=self.url,
                file_path=self.file_path,
                version=self.version,
                output_path=target.path,
            )

    def output(self):
        return TaggedOutputLocalTarget(Config().white_matter_file)
