"""Helpers for white matter recipe."""
import json
import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np
import pandas as pd
import yaml
from git import Repo
from scipy.spatial.distance import squareform

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.utils import cols_from_json
from axon_synthesis.utils import cols_to_json
from axon_synthesis.utils import fill_diag

LOGGER = logging.getLogger(__name__)


def fetch(
    url,
    output_path,
    file_path="white_matter_FULL_RECIPE_v1p20.yaml",
    version_reference=None,
):
    """Fetch the White Natter Recipe file from an internal repository."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "tmp_repo"
        Repo.clone_from(url, dest)
        if version_reference is not None:
            r = Repo(dest)
            r.git.checkout(version_reference)
        shutil.copy(dest / file_path, output_path)
    if version_reference is None:
        version_reference = "latest"
    LOGGER.info(
        (
            "Fetched the White Matter Recipe using the '%s' file from the '%s' repository at "
            "version '%s' to the file '%s'"
        ),
        file_path,
        url,
        version_reference,
        output_path,
    )


def load(white_matter_file: Union[str, Path]):
    """Load the white matter recipe from YAML file."""
    white_matter_file = Path(white_matter_file)
    LOGGER.debug("Loading white matter recipe file from: %s", white_matter_file)
    with white_matter_file.open("r", encoding="utf-8") as f:
        wm_recipe = yaml.load(f, Loader=yaml.SafeLoader)

    return wm_recipe


def get_atlas_region_id(region_map, pop_row, col_name, second_col_name=None):
    """Get the ID of an atlas region."""

    def get_ids(region_map, pop_row, col_name):
        if not pop_row.isnull()[col_name]:
            acronym = pop_row[col_name]
            ids = region_map.find(acronym, attr="acronym")
        else:
            acronym = None
            ids = []
        return ids, acronym

    ids, acronym = get_ids(region_map, pop_row, col_name)
    if len(ids) == 0 and second_col_name is not None:
        ids, new_acronym = get_ids(region_map, pop_row, second_col_name)
        if len(ids) == 1 and acronym is not None:
            LOGGER.warning(
                "Could not find any ID for %s in the region map but found one for %s",
                acronym,
                new_acronym,
            )
    else:
        new_acronym = None

    if len(ids) > 1:
        raise ValueError(
            f"Found several IDs for the acronym '{acronym or new_acronym}' in the region "
            f"map: {sorted(ids)}"
        )
    if len(ids) == 0:
        raise ValueError(f"Could not find the acronym '{acronym or new_acronym}' in the region map")
    return ids.pop()


class WhiteMatterRecipe:
    """Class to store the White Matter Recipe data."""

    filename = {
        "populations": "populations.csv",
        "projections": "projections.csv",
        "targets": "targets.csv",
        "fractions": "fractions.json",
        "interaction_strengths": "interaction_strengths.json",
        "projection_targets": "projection_targets.csv",
        "layer_profiles": "layer_profiles.csv",
        "region_data": "region_data.csv",
    }

    def __init__(
        self,
        populations,
        projections,
        targets,
        fractions,
        interaction_strengths,
        projection_targets,
        layer_profiles,
        region_data,
    ):
        self.populations = populations
        self.projections = projections
        self.targets = targets
        self.fractions = fractions
        self.interaction_strengths = interaction_strengths
        self.projection_targets = projection_targets
        self.layer_profiles = layer_profiles
        self.region_data = region_data

    def save(self, path):
        """Save the White Matter Recipe into the given directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Export the population DataFrame
        populations = cols_to_json(self.populations, ["atlas_region", "filters"])
        populations.to_csv(path / self.filename["populations"], index=False)

        # Export the projection DataFrame
        projections = cols_to_json(
            self.projections, ["mapping_coordinate_system", "targets", "atlas_region", "filters"]
        )
        projections.to_csv(path / self.filename["projections"], index=False)

        # Export the targets DataFrame
        targets = cols_to_json(self.targets, ["target"])
        targets.to_csv(path / self.filename["targets"], index=False)

        # Export the projection DataFrame
        projection_targets = cols_to_json(
            self.projection_targets,
            ["targets", "atlas_region", "filters", "target", "topographical_mapping"],
        )
        projection_targets.to_csv(path / self.filename["projection_targets"], index=False)

        # Export the fractions
        with (path / self.filename["fractions"]).open("w", encoding="utf-8") as f:
            json.dump(self.fractions, f, indent=4, sort_keys=True)

        # Export the interaction strengths
        with (path / self.filename["interaction_strengths"]).open("w", encoding="utf-8") as f:
            json.dump(
                {k: v.to_dict("index") for k, v in self.interaction_strengths.items()},
                f,
                indent=4,
                sort_keys=True,
            )

        # Export the layer profiles
        layer_profiles = cols_to_json(self.layer_profiles, ["layers"])
        layer_profiles.to_csv(path / self.filename["layer_profiles"], index=False)

        # Export the region data
        self.region_data.to_csv(path / self.filename["region_data"], index=False)

    @classmethod
    def load(cls, path):
        """Load the White Matter Recipe from the given directory."""
        populations = pd.read_csv(path / cls.filename["populations"])
        populations = cols_from_json(populations, ["atlas_region", "filters"])

        projections = pd.read_csv(path / cls.filename["projections"])
        projections = cols_from_json(
            projections, ["mapping_coordinate_system", "targets", "atlas_region", "filters"]
        )

        targets = pd.read_csv(path / cls.filename["targets"])
        targets = cols_from_json(targets, ["target"])

        with (path / cls.filename["fractions"]).open("r", encoding="utf-8") as f:
            fractions = json.load(f)

        with (path / cls.filename["interaction_strengths"]).open("r", encoding="utf-8") as f:
            interaction_strengths = json.load(f)

        projection_targets = pd.read_csv(path / cls.filename["projection_targets"])
        projection_targets = cols_from_json(
            projection_targets,
            ["targets", "atlas_region", "filters", "target", "topographical_mapping"],
        )

        layer_profiles = pd.read_csv(path / cls.filename["layer_profiles"])
        layer_profiles = cols_from_json(layer_profiles, ["layers"])

        region_data = pd.read_csv(path / cls.filename["region_data"])

        return cls(
            populations,
            projections,
            targets,
            fractions,
            interaction_strengths,
            projection_targets,
            layer_profiles,
            region_data,
        )

    @classmethod
    def from_raw_WMR(
        cls,
        path: str | os.PathLike,
        atlas: AtlasHelper,
        subregion_uppercase: bool,
        subregion_remove_prefix: bool,
        sub_region_separator: str,
    ):
        """Process the white matter recipe."""
        # pylint: disable=too-many-statements
        LOGGER.info("Loading and processing the white matter recipe YAML file '%s'", path)
        region_map = atlas.region_map
        brain_regions = atlas.brain_regions
        wmr = load(path)

        # Get populations
        LOGGER.debug("Extracting populations from white matter recipe")
        wm_populations = pd.DataFrame.from_records(wmr["populations"])
        wm_populations_sub = wm_populations.loc[
            wm_populations["atlas_region"].apply(lambda x: isinstance(x, list)),
            "atlas_region",
        ]
        if not wm_populations_sub.empty:
            wm_populations_sub = (
                wm_populations_sub.apply(pd.Series)
                .stack()
                .dropna()
                .rename("atlas_region_split")
                .reset_index(level=1, drop=True)
            )
            wm_populations = wm_populations.join(wm_populations_sub, how="left")
            wm_populations["atlas_region_split"].fillna(
                wm_populations["atlas_region"], inplace=True
            )
            wm_populations.drop(columns=["atlas_region"], inplace=True)
            wm_populations.rename(columns={"atlas_region_split": "atlas_region"}, inplace=True)
        wm_populations.rename(columns={"name": "pop_raw_name"}, inplace=True)
        wm_populations["region_acronym"] = wm_populations["atlas_region"].apply(
            lambda row: row["name"]
        )
        wm_populations_sub = (
            wm_populations["atlas_region"]
            .apply(lambda row: pd.Series(row.get("subregions", []), dtype=object))
            .stack()
            .dropna()
            .rename("sub_region")
            .reset_index(level=1, drop=True)
        )
        wm_populations = wm_populations.join(wm_populations_sub, how="left")

        # Get subregion names
        wm_populations["formatted_subregion"] = wm_populations["sub_region"]
        if subregion_uppercase:
            wm_populations["formatted_subregion"] = wm_populations[
                "formatted_subregion"
            ].str.upper()
        if subregion_remove_prefix:
            wm_populations["formatted_subregion"] = wm_populations[
                "formatted_subregion"
            ].str.extract(r"(\d+.*)")
        wm_populations["subregion_acronym"] = (
            wm_populations["region_acronym"]
            + sub_region_separator
            + wm_populations["formatted_subregion"]
        )

        # Get atlas subregion IDs
        wm_populations["atlas_region_id"] = wm_populations.apply(
            lambda row: get_atlas_region_id(region_map, row, "subregion_acronym", "region_acronym"),
            axis=1,
        )

        # Compute the volume of each region
        region_ids, region_counts = np.unique(brain_regions.raw, return_counts=True)
        region_data = pd.DataFrame({"atlas_region_id": region_ids, "count": region_counts})

        # Get upper-level regions from the populations
        region_data = region_data.merge(
            wm_populations[["atlas_region_id"]], on="atlas_region_id", how="outer"
        )
        region_data.drop_duplicates(inplace=True)

        # Compute the volumes of upper-level regions
        # TODO: Check if the attr should be 'id' or 'atlas_id'
        region_data["nb_voxels"] = region_data["atlas_region_id"].apply(
            lambda row: region_counts[
                np.argwhere(
                    np.isin(
                        region_ids,
                        [
                            region_map.get(i, "atlas_id")
                            for i in region_map.find(row, attr="atlas_id", with_descendants=True)
                        ],
                    )
                )
            ].sum()
        )
        region_data["atlas_region_volume"] = region_data["nb_voxels"] * brain_regions.voxel_volume
        region_data.drop(columns=["count"], inplace=True)

        # Join region data to population
        wm_populations = wm_populations.merge(region_data, on="atlas_region_id", how="left")

        # Compute volume fractions of sub-regions
        region_ids = wm_populations.apply(
            lambda row: get_atlas_region_id(region_map, row, "region_acronym"), axis=1
        ).rename("region_id")
        pop_frac = wm_populations.join(region_ids)
        pop_frac = pop_frac.merge(
            region_data,
            left_on="region_id",
            right_on="atlas_region_id",
            how="left",
            suffixes=("", "_total"),
        )
        wm_populations["sub_region_volume_frac"] = (
            (pop_frac["atlas_region_volume"] / pop_frac["atlas_region_volume_total"])
            .fillna(1)
            .clip(0, 1)
        )

        # Get layer_profiles
        LOGGER.debug("Extracting layer profiles from white matter recipe")
        wm_layer_profiles = pd.DataFrame.from_records(wmr["layer_profiles"])
        layer_profiles = (
            wm_layer_profiles["relative_densities"]
            .apply(pd.Series)
            .stack()
            .rename("layer_profile")
            .reset_index(level=1)
            .rename(columns={"level_1": "layer_profile_num"})
        )
        wm_layer_profiles = wm_layer_profiles.join(layer_profiles).set_index(
            "layer_profile_num", append=True
        )
        wm_layer_profiles["layers"] = wm_layer_profiles["layer_profile"].apply(
            lambda row: row.get("layers", None)
        )
        wm_layer_profiles["value"] = wm_layer_profiles["layer_profile"].apply(
            lambda row: row.get("value", None)
        )
        wm_layer_profiles.drop(columns=["relative_densities", "layer_profile"], inplace=True)
        wm_layer_profiles = wm_layer_profiles.join(
            wm_layer_profiles["layers"]
            .apply(pd.Series)
            .stack()
            .rename("layer")
            .reset_index(level=2)
            .rename(columns={"level_2": "layer_index"})
        )
        wm_layer_profiles["formatted_layer"] = wm_layer_profiles["layer"].str.extract("l(.*)")
        wm_layer_profiles["formatted_layer"].fillna(wm_layer_profiles["layer"], inplace=True)

        # Get projections
        LOGGER.debug("Extracting projections from white matter recipe")
        wm_projections = pd.DataFrame.from_records(wmr["projections"])
        if wm_projections["source"].duplicated().any():
            raise ValueError(
                "Found several equal sources in the 'projections' entry: "
                f"{sorted(wm_projections.loc[wm_projections['a'].duplicated(), 'a'].tolist())}"
            )

        # Map projections
        wm_projections = wm_projections.merge(
            wm_populations, left_on="source", right_on="pop_raw_name", how="left"
        )

        wm_targets = (
            wm_projections["targets"]
            .apply(pd.Series)
            .stack()
            .rename("target")
            .reset_index(level=1)
            .rename(columns={"level_1": "target_num"})
        )
        wm_projection_targets = wm_projections.join(wm_targets).set_index("target_num", append=True)
        wm_projection_targets["target_population_name"] = wm_projection_targets["target"].apply(
            lambda row: row["population"]
        )
        wm_projection_targets["target_projection_name"] = wm_projection_targets["target"].apply(
            lambda row: row["projection_name"]
        )
        wm_projection_targets["target_density"] = wm_projection_targets["target"].apply(
            lambda row: row["density"]
        )
        wm_projection_targets["topographical_mapping"] = wm_projection_targets["target"].apply(
            lambda row: row["presynaptic_mapping"]
        )
        wm_projection_targets["target_layer_profiles"] = wm_projection_targets["target"].apply(
            lambda row: row["target_layer_profiles"]
        )
        wm_projection_targets.index.rename("proj_index", level=0, inplace=True)

        # Get target sub regions
        region_map_df = region_map.as_dataframe()
        region_map_df = (
            region_map_df.reset_index()
            .merge(
                region_map_df[["acronym"]].reset_index(),
                left_on="parent_id",
                right_on="id",
                suffixes=("", "_parent"),
                how="left",
            )
            .set_index("id")
        )
        sub_region_acronyms = region_map_df.groupby(["acronym_parent"])["acronym"].apply(list)
        sub_region_acronyms.index.rename("region_acronym", inplace=True)
        sub_region_acronyms.rename("subregion_acronyms", inplace=True)
        # sub_region_atlas_ids = region_map_df.groupby(["acronym_parent"])["atlas_id"].apply(list)
        # sub_region_atlas_ids.index.rename("region_acronym", inplace=True)
        # sub_region_atlas_ids.rename("subregion_atlas_ids", inplace=True)

        # Join layer profiles

        # Stack sub-regions

        target_pop_name = (
            wm_projection_targets["target_population_name"]
            .str.extract("(.*)(_ALL_LAYERS)?")[0]
            .rename("target_pop_name")
            .to_frame()
        )
        wm_projection_targets["target_region"] = (
            target_pop_name.reset_index()
            .merge(
                wm_populations[["pop_raw_name", "region_acronym"]],
                left_on="target_pop_name",
                right_on="pop_raw_name",
            )
            .set_index(["proj_index", "target_num"])["region_acronym"]
            .sort_index()
        )
        wm_projection_targets["target_region"].fillna(
            wm_projection_targets["target_population_name"], inplace=True
        )
        wm_projection_targets = (
            wm_projection_targets.reset_index()
            .merge(
                region_map_df[["acronym", "atlas_id"]],
                left_on="target_region",
                right_on="acronym",
                how="left",
            )
            .set_index(["proj_index", "target_num"])
            .rename(columns={"atlas_id": "target_region_atlas_id"})
            .drop(columns=["acronym"])
        )
        wm_projection_targets = wm_projection_targets.merge(
            sub_region_acronyms, left_on="target_region", right_index=True, how="left"
        )

        selected_sub_region_acronyms = (
            wm_projection_targets["subregion_acronyms"]
            .apply(pd.Series)
            .stack()
            .rename("subregion_acronym")
            .reset_index(level=2)
            .rename(columns={"level_2": "subregion_num"})
        )
        wm_projection_targets = wm_projection_targets.join(
            selected_sub_region_acronyms, rsuffix="_target"
        ).rename(columns={"subregion_acronym_target": "target_subregion_acronym"})
        wm_projection_targets = (
            wm_projection_targets.reset_index()
            .merge(
                region_map_df[["acronym", "atlas_id"]],
                left_on="target_subregion_acronym",
                right_on="acronym",
                how="left",
            )
            .set_index(["proj_index", "target_num"])
            .rename(columns={"atlas_id": "target_subregion_atlas_id"})
            .drop(columns=["acronym"])
        )

        # selected_sub_region_acronyms.index.rename("index", level=0, inplace=True)

        # Drop target sub-regions that are not listed in the populations
        wm_projection_targets = wm_projection_targets.loc[
            wm_projection_targets["target_subregion_acronym"].isin(
                wm_populations["subregion_acronym"].drop_duplicates()
            )
        ]

        # Join layer profiles
        wm_projection_targets = wm_projection_targets.merge(
            wm_populations[
                ["subregion_acronym", "sub_region", "formatted_subregion", "sub_region_volume_frac"]
            ].drop_duplicates(),
            left_on="target_subregion_acronym",
            right_on="subregion_acronym",
            how="left",
            suffixes=("", "_target"),
        )
        wm_projection_targets["target_layer_profile_name"] = wm_projection_targets.apply(
            lambda row: row["target_layer_profiles"][0].get("name"), axis=1
        )
        # wm_projection_targets.drop(columns=["targets", "target_layer_profiles"], inplace=True)
        wm_projection_targets = (
            wm_projection_targets.reset_index()
            .merge(
                wm_layer_profiles[["name", "layer", "formatted_layer", "value"]].rename(
                    columns={"value": "target_layer_profile_density"}
                ),
                left_on=["target_layer_profile_name", "sub_region_target"],
                right_on=["name", "layer"],
                how="left",
            )
            .set_index("index")
        )
        wm_projection_targets["target_layer_profile_density"].fillna(1, inplace=True)
        wm_projection_targets["target_layer_profile_prob"] = (
            wm_projection_targets["target_layer_profile_density"]
            * wm_projection_targets["sub_region_volume_frac_target"]
        )

        normalization_factor = (
            wm_projection_targets.groupby(
                ["source", "sub_region", "target_projection_name", "target_region"]
            )["target_layer_profile_prob"]
            .sum()
            .rename("target_layer_profile_norm_factor")
        )
        wm_projection_targets = wm_projection_targets.merge(
            normalization_factor,
            left_on=["source", "sub_region", "target_projection_name", "target_region"],
            right_index=True,
            how="left",
        )
        wm_projection_targets.fillna({"target_layer_profile_norm_factor": 1})
        wm_projection_targets["target_layer_profile_region_prob"] = (
            wm_projection_targets["target_layer_profile_prob"]
            / wm_projection_targets["target_layer_profile_norm_factor"]
        )
        # wm_projection_targets["partial_strength"] = (
        #     wm_projection_targets["strength"]
        #     * wm_projection_targets["target_layer_profile_norm_factor"]
        # )

        # Ignore the sub-regions not listed in layer profiles

        # wm_projection_targets["target_region"].fillna("target_population_name", inplace=True)

        # target_all_layers_mask = wm_projection_targets["target_population_name"].str.endswith(
        #     "_ALL_LAYERS"
        # )
        # wm_projection_targets["target_sub_regions"] = None
        # wm_projection_targets.loc[target_all_layers_mask, "target_sub_regions"]

        # Use region ID when there is no sub-region
        wm_projection_targets["target_atlas_id"] = wm_projection_targets[
            "target_subregion_atlas_id"
        ].fillna(wm_projection_targets["target_region_atlas_id"])
        wm_projection_targets.fillna({"target_layer_profile_region_prob": 1}, inplace=True)
        wm_projection_targets["has_atlas_id"] = ~wm_projection_targets[
            "target_subregion_atlas_id"
        ].isnull()

        # # Compute normalization factors

        # # Compute final probabilities

        # wm_projection_targets.merge(
        #     wm_populations[
        #         [
        #             "pop_raw_name",
        #             "region_acronym",
        #             "sub_region",
        #             "formatted_subregion",
        #             "subregion_acronym",
        #             "atlas_region_id",
        #             "atlas_region_volume",
        #             "sub_region_volume_frac",
        #         ]
        #     ],
        #     left_on="target_population_name",
        #     right_on="pop_raw_name",
        #     how="left",
        #     suffixes=("_src", "_trgt"),
        # )

        # wm_projection_targets["layer_profile_name"] = wm_projection_targets.apply(
        #     lambda row: row["target_layer_profiles"][0].get("name"), axis=1
        # )
        # wm_projection_targets.drop(columns=["targets", "target_layer_profiles"], inplace=True)
        # wm_projection_targets = (
        #     wm_projection_targets.reset_index()
        #     .merge(
        #         wm_layer_profiles[["name", "layer", "formatted_layer", "value"]].rename(
        #             columns={"value": "layer_profile_density"}
        #         ),
        #         left_on=["layer_profile_name", "sub_region"],
        #         right_on=["name", "layer"],
        #         how="left",
        #     )
        #     .set_index(["level_0", "target_num"])
        # )
        # wm_projection_targets["layer_profile_density"].fillna(1, inplace=True)

        # wm_projection_targets["layer_profile_prob"] = (
        #     wm_projection_targets["layer_profile_density"]
        #     * wm_projection_targets["sub_region_volume_frac"]
        # )

        # print()
        # wm_projection_targets.groupby("region_acronym")["layer_profile_prob"].sum()

        # wm_projection_targets.drop(columns=["atlas_region", "name", "layer"], inplace=True)

        # for i in target.get("target_layer_profiles", []):
        #     layer_fraction = i.get("fraction", 1.0)
        #     layer_name = i["name"]

        # Get fractions
        LOGGER.debug("Extracting fractions from white matter recipe")
        wm_fractions = {i["population"]: i["fractions"] for i in wmr["p-types"]}

        # Get interaction_mat and strengths
        LOGGER.debug("Extracting interaction matrix from white matter recipe")
        wm_interaction_mat = {
            i["population"]: i["interaction_mat"] for i in wmr["p-types"] if "interaction_mat" in i
        }

        wm_interaction_strengths = {
            k: pd.DataFrame(
                fill_diag(squareform(v["strengths"]), 1),
                columns=wm_interaction_mat[k]["projections"],
                index=wm_interaction_mat[k]["projections"],
            )
            for k, v in wm_interaction_mat.items()
        }

        return cls(
            wm_populations,
            wm_projections,
            wm_targets,
            wm_fractions,
            wm_interaction_strengths,
            wm_projection_targets,
            wm_layer_profiles,
            region_data,
        )
