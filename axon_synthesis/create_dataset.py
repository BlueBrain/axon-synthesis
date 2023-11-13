"""Create the dataset that can be used for the Curate workflow from MPoW.

The workflow should be called using the luigi.cfg file from this directory and
"morphology-workflows==0.2.0".
"""
# import json
from pathlib import Path

import luigi
import luigi_tools
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from morphology_workflows.tasks.workflows import Curate

# from axon_synthesis.atlas import load as load_atlas
from axon_synthesis.config import Config


class CreateDatasetForRepair(luigi_tools.task.WorkflowTask):
    """Task to create the input dataset."""

    morph_dir = luigi.Parameter(description="Folder containing the input morphologies.")
    output_dataset = luigi.Parameter(description="Output dataset file", default="dataset.csv")

    def run(self):
        morph_dir = Path(self.morph_dir)

        dataset = pd.DataFrame(columns=["morph_path", "mtype"])
        dataset.index.name = "morph_name"

        for morph in morph_dir.iterdir():
            if morph.suffix.lower() in [".asc", ".h5", ".swc"]:
                dataset.loc[morph.stem, "morph_path"] = morph
                dataset.loc[morph.stem, "mtype"] = "UNKNOWN"
        dataset.sort_index(inplace=True)
        dataset.reset_index().to_csv(self.output().path, index=False)
        return dataset

    def output(self):
        return TaggedOutputLocalTarget(self.output_dataset, create_parent=True)


class RepairDataset(luigi_tools.task.WorkflowTask):
    """Task to repair the input dataset."""

    def requires(self):
        dataset = CreateDatasetForRepair()
        return dataset

    def run(self):
        yield Curate(dataset_df=self.input().path, result_path=Config().output_dir.absolute())

    def output(self):
        return TaggedOutputLocalTarget("Resample/data/")


class RawDataset(luigi_tools.task.WorkflowWrapperTask):
    """Task to get the input dataset."""

    def requires(self):
        return RepairDataset()

    def output(self):
        return TaggedOutputLocalTarget(
            self.input().pathlib_path.resolve().parent.parent / "Collect/data/"
        )


# class WMROutputLocalTarget(TaggedOutputLocalTarget):
#     """Target for white matter recipe outputs."""

#     __prefix = "white_matter_recipe"  # pylint: disable=unused-private-member


# class FetchWhiteMatterRecipe(luigi_tools.task.WorkflowTask):
#     """Task to fetch the White Matter Recipe file from a repository."""

#     url = luigi.Parameter(
#         default=None,
#         description=":str: Url of the repository.",
#     )
#     file_path = luigi.OptionalParameter(
#         description=":str: Path of the file in the repository to fetch.",
#         default="white_matter_FULL_RECIPE_v1p20.yaml",
#     )
#     version = luigi.OptionalParameter(
#         description=":str: Version of the repository to checkout (use HEAD if not given).",
#         default=None,
#     )
#     sub_region_separator = luigi.Parameter(
#         description="Separator use between region and subregion names to build the acronym.",
#         default="",
#     )
#     subregion_uppercase = luigi.BoolParameter(
#         description=("If set to True, the subregion names are uppercased."),
#         default=False,
#         parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
#     )
#     subregion_remove_prefix = luigi.BoolParameter(
#         description=(
#             "If set to True, only the layer numbers are extracted from the subregion names."
#         ),
#         default=False,
#         parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
#     )

#     def process(
#         self,
#         target_path,
#         atlas_path,
#         atlas_region_filename,
#         atlas_hierarchy_filename,
#     ):
#         if not target_path.exists():
#             # Note: this check should be useless because luigi calls the run() method only if the
#             # target does not exist, but we keep it for safety.
#             fetch_wmr(
#                 url=self.url,
#                 file_path=self.file_path,
#                 version=self.version,
#                 output_path=target_path.resolve(),
#             )

#         # Get atlas data
#         _, brain_regions, region_map = load_atlas(
#             atlas_path,
#             atlas_region_filename,
#             atlas_hierarchy_filename,
#         )

#         # Get the white matter recipe
#         wm_recipe = load_wmr(target_path)

#         # Process the white matter recipe
#         return process_wmr(
#             wm_recipe,
#             region_map,
#             brain_regions,
#             self.subregion_uppercase,
#             self.subregion_remove_prefix,
#             self.sub_region_separator,
#         )

#     def run(self):
#         target = self.output()["WMR"]
#         config = Config()

#         wmr_data = self.process(
#             target.pathlib_path,
#             config.atlas_path.resolve(),
#             config.atlas_region_filename,
#             config.atlas_hierarchy_filename,
#         )

#         self.save(wmr_data)

#     def save(self, wmr_data):
#         """"""
#         wm_populations = wmr_data["wm_populations"]
#         wm_projections = wmr_data["wm_projections"]
#         wm_targets = wmr_data["wm_targets"]
#         wm_fractions = wmr_data["wm_fractions"]
#         wm_interaction_strengths = wmr_data["wm_interaction_strengths"]
#         projection_targets = wmr_data["projection_targets"]
#         wm_layer_profiles = wmr_data["wm_layer_profiles"]
#         region_data = wmr_data["region_data"]

#         # Export the population DataFrame
#         wm_populations = cols_to_json(wm_populations, ["atlas_region", "filters"])
#         wm_populations.to_csv(self.output()["wm_populations"].path, index=False)

#         # Export the projection DataFrame
#         wm_projections = cols_to_json(
#             wm_projections, ["mapping_coordinate_system", "targets", "atlas_region", "filters"]
#         )
#         wm_projections.to_csv(self.output()["wm_projections"].path, index=False)

#         # Export the targets DataFrame
#         wm_targets = cols_to_json(wm_targets, ["target"])
#         wm_targets.to_csv(self.output()["wm_targets"].path, index=False)

#         # Export the projection DataFrame
#         projection_targets = cols_to_json(
#             projection_targets,
#             ["targets", "atlas_region", "filters", "target", "topographical_mapping"],
#         )
#         projection_targets.to_csv(self.output()["wm_projection_targets"].path, index=False)

#         # Export the fractions
#         with self.output()["wm_fractions"].pathlib_path.open("w", encoding="utf-8") as f:
#             json.dump(wm_fractions, f, indent=4, sort_keys=True)

#         # Export the interaction strengths
#         with self.output()["wm_interaction_strengths"].pathlib_path.open(
#             "w", encoding="utf-8"
#         ) as f:
#             json.dump(
#                 {k: v.to_dict("index") for k, v in wm_interaction_strengths.items()},
#                 f,
#                 indent=4,
#                 sort_keys=True,
#             )

#         # Export the layer profiles
#         layer_profiles = cols_to_json(wm_layer_profiles, ["layers"])
#         layer_profiles.to_csv(self.output()["wm_layer_profiles"].path, index=False)

#         # Export the region data
#         region_data.to_csv(self.output()["region_data"].path, index=False)

#     def output(self):
#         return {
#             "WMR": WMROutputLocalTarget(Config().white_matter_file),
#             "wm_populations": WMROutputLocalTarget(
#                 "white_matter_population.csv", create_parent=True
#             ),
#             "wm_projections": WMROutputLocalTarget(
#                 "white_matter_projections.csv", create_parent=True
#             ),
#             "wm_projection_targets": WMROutputLocalTarget(
#                 "white_matter_projection_targets.csv", create_parent=True
#             ),
#             "wm_fractions": WMROutputLocalTarget("white_matter_fractions.json", create_parent=True),
#             "wm_targets": WMROutputLocalTarget("white_matter_targets.csv", create_parent=True),
#             "wm_interaction_strengths": WMROutputLocalTarget(
#                 "white_matter_interaction_strengths.json", create_parent=True
#             ),
#             "wm_layer_profiles": WMROutputLocalTarget("layer_profiles.csv", create_parent=True),
#             "region_data": WMROutputLocalTarget("region_data.csv", create_parent=True),
#         }
