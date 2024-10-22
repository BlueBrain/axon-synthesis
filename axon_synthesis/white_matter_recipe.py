"""Helpers for white matter recipe."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import json
import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import numpy as np
import pandas as pd
import yaml
from attrs import asdict
from attrs import define
from attrs import field
from git import Repo
from scipy.spatial.distance import squareform

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.base_path_builder import BasePathBuilder
from axon_synthesis.constants import WMR_ATLAS_ID
from axon_synthesis.typing import FileType
from axon_synthesis.typing import Self
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


def load(white_matter_file: FileType) -> dict:
    """Load the white matter recipe from YAML file."""
    white_matter_file = Path(white_matter_file)
    LOGGER.debug("Loading white matter recipe file from: %s", white_matter_file)
    with white_matter_file.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def get_atlas_region_id(region_map, pop_row, col_name, second_col_name=None):
    """Get the ID of an atlas region."""

    def get_ids(region_map, pop_row, col_name) -> tuple[list, str]:
        if not pop_row.isna()[col_name]:
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
        msg = (
            f"Found several IDs for the acronym '{acronym or new_acronym}' in the region "
            f"map: {sorted(ids)}"
        )
        raise ValueError(msg)
    if len(ids) == 0:
        msg = f"Could not find the acronym '{acronym or new_acronym}' in the region map"
        raise ValueError(msg)
    return ids.pop()


@define
class WmrConfig:
    """Class to store the WhiteMatterRecipe configuration."""

    path: Path = field(converter=Path)
    subregion_uppercase: bool
    subregion_remove_prefix: bool
    sub_region_separator: str

    def to_dict(self) -> dict:
        """Return all attribute values into a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a new AtlasConfig object from a dictionary."""
        return cls(
            data["path"],
            data["subregion_uppercase"],
            data["subregion_remove_prefix"],
            data["sub_region_separator"],
        )


class WhiteMatterRecipe(BasePathBuilder):
    """Class to store the White Matter Recipe data."""

    _filenames: ClassVar[dict] = {
        "FRACTIONS_FILENAME": "fractions.json",
        "INTERACTION_STRENGTHS_FILENAME": "interaction_strengths.json",
        "LAYER_PROFILES_FILENAME": "layer_profiles.csv",
        "POPULATIONS_FILENAME": "populations.csv",
        "PROJECTION_TARGETS_FILENAME": "projection_targets.csv",
        "PROJECTIONS_FILENAME": "projections.csv",
        "REGION_DATA_FILENAME": "region_data.csv",
        "TARGETS_FILENAME": "targets.csv",
    }

    def __init__(
        self,
        path,
        *,
        load=True,
    ):  # pylint: disable=redefined-outer-name
        """Create a new WhiteMatterRecipe object.

        Args:
            path: The base path used to build the relative paths.
            load: If set to False the internal data are not automatically loaded.
        """
        super().__init__(path)

        self.populations: pd.DataFrame | None = None
        self.projections: pd.DataFrame | None = None
        self.targets: pd.DataFrame | None = None
        self.fractions: dict | None = None
        self.interaction_strengths: dict | None = None
        self.projection_targets: pd.DataFrame | None = None
        self.layer_profiles: pd.DataFrame | None = None
        self.region_data: pd.DataFrame | None = None

        if load:
            self.assert_exists()
            self.load()

    def save(self):
        """Save the White Matter Recipe into the given directory."""
        self.path.mkdir(parents=True, exist_ok=True)

        # Export the population DataFrame
        if self.populations is not None:
            populations = cols_to_json(self.populations, ["atlas_region", "filters"])
            populations.to_csv(self.POPULATIONS_FILENAME, index=False)

        # Export the projection DataFrame
        if self.projections is not None:
            projections = cols_to_json(
                self.projections,
                ["mapping_coordinate_system", "targets", "atlas_region", "filters"],
            )
            projections.to_csv(self.PROJECTIONS_FILENAME, index=False)

        # Export the targets DataFrame
        if self.targets is not None:
            targets = cols_to_json(self.targets, ["target"])
            targets.to_csv(self.TARGETS_FILENAME, index=False)

        # Export the projection DataFrame
        if self.projection_targets is not None:
            projection_targets = cols_to_json(
                self.projection_targets,
                ["targets", "atlas_region", "filters", "target", "topographical_mapping"],
            )
            projection_targets.to_csv(self.PROJECTION_TARGETS_FILENAME, index=False)

        # Export the fractions
        if self.fractions is not None:
            with (self.FRACTIONS_FILENAME).open("w", encoding="utf-8") as f:
                json.dump(self.fractions, f, indent=4, sort_keys=True)

        # Export the interaction strengths
        if self.interaction_strengths is not None:
            with (self.INTERACTION_STRENGTHS_FILENAME).open("w", encoding="utf-8") as f:
                json.dump(
                    {k: v.to_dict("index") for k, v in self.interaction_strengths.items()},
                    f,
                    indent=4,
                    sort_keys=True,
                )

        # Export the layer profiles
        if self.layer_profiles is not None:
            layer_profiles = cols_to_json(self.layer_profiles, ["layers"])
            layer_profiles.to_csv(self.LAYER_PROFILES_FILENAME, index=False)

        # Export the region data
        if self.region_data is not None:
            self.region_data.to_csv(self.REGION_DATA_FILENAME, index=False)

    def load(self):
        """Load the White Matter Recipe from the associated directory."""
        LOGGER.info(
            "Loading the White Matter Recipe from '%s'",
            self.path,
        )
        populations = pd.read_csv(self.POPULATIONS_FILENAME)
        self.populations = cols_from_json(populations, ["atlas_region", "filters"])

        projections = pd.read_csv(self.PROJECTIONS_FILENAME)
        self.projections = cols_from_json(
            projections,
            ["mapping_coordinate_system", "targets", "atlas_region", "filters"],
        )

        targets = pd.read_csv(self.TARGETS_FILENAME)
        self.targets = cols_from_json(targets, ["target"])

        with self.FRACTIONS_FILENAME.open("r", encoding="utf-8") as f:
            self.fractions = json.load(f)

        with self.INTERACTION_STRENGTHS_FILENAME.open("r", encoding="utf-8") as f:
            self.interaction_strengths = json.load(f)

        projection_targets = pd.read_csv(self.PROJECTION_TARGETS_FILENAME)
        self.projection_targets = cols_from_json(
            projection_targets,
            ["targets", "atlas_region", "filters", "target", "topographical_mapping"],
        )

        layer_profiles = pd.read_csv(self.LAYER_PROFILES_FILENAME)
        self.layer_profiles = cols_from_json(layer_profiles, ["layers"])

        self.region_data = pd.read_csv(self.REGION_DATA_FILENAME)

    def compute_probabilities(self, atlas: AtlasHelper):
        """Compute projection probabilities from the White Matter Recipe."""
        LOGGER.info("Computing the probabilities from the white matter")

        if self.projection_targets is None:
            msg = (
                "The 'projection_targets' DataFrame is not loaded yet so it is not possible to "
                "compute the probabilities"
            )
            raise RuntimeError(msg)

        projection_targets = self.projection_targets.loc[
            ~self.projection_targets["target_region_atlas_id"].isna()
        ]
        projection_targets = projection_targets.fillna(
            {"target_subregion_atlas_id": projection_targets["target_region_atlas_id"]},
        ).astype({"target_region_atlas_id": int, "target_subregion_atlas_id": int})

        # Get the projection matrix
        projection_matrix = (
            pd.DataFrame.from_records(self.fractions)  # type: ignore[call-overload]
            .stack()
            .rename("target_projection_strength")
            .reset_index()
            .rename(columns={"level_0": "target_projection_name", "level_1": "pop_raw_name"})
        )

        all_targets = projection_targets.merge(
            projection_matrix,
            on=["pop_raw_name", "target_projection_name"],
            how="left",
        )
        all_targets["probability"] = (
            all_targets["target_projection_strength"]
            * all_targets["target_layer_profile_region_prob"]
        )
        all_targets["subregion_acronym"] = all_targets.fillna(
            {"subregion_acronym": all_targets["region_acronym"]}, inplace=True
        )
        all_targets = all_targets.rename(
            columns={"atlas_region_id": "source_brain_region_id", "pop_raw_name": "population_id"}
        )

        all_targets["target_brain_region_acronym"] = all_targets["target_subregion_acronym"].fillna(
            all_targets["target_region"]
        )
        all_targets["target_brain_region_id"] = all_targets["target_brain_region_acronym"].apply(
            lambda x: atlas.get_region_ids(x, with_descendants=False)[0][0]
        )

        population_probabilities = (
            all_targets[["source_brain_region_id", "population_id"]]
            .drop_duplicates()
            .rename(columns={"source_brain_region_id": "brain_region_id"})
        )
        population_probabilities = population_probabilities.merge(
            (1 / population_probabilities.groupby(["brain_region_id"]).size()).rename(  # type: ignore[call-overload]
                "probability"
            ),
            left_on="brain_region_id",
            right_index=True,
        )

        projection_probabilities = (
            all_targets[
                [
                    "source_brain_region_id",
                    "population_id",
                    "target_brain_region_id",
                    "target_projection_name",
                    "probability",
                ]
            ]
            .drop_duplicates()
            .rename(
                columns={
                    "population_id": "source_population_id",
                    "target_projection_name": "target_population_id",
                }
            )
        )
        projection_probabilities = projection_probabilities.loc[
            projection_probabilities["probability"] > 0
        ]

        return population_probabilities, projection_probabilities

    def load_from_raw_wmr(  # noqa: PLR0915
        self,
        config: WmrConfig,
        atlas: AtlasHelper,
    ):
        """Process the white matter recipe."""
        # pylint: disable=too-many-statements
        LOGGER.info("Loading and processing the white matter recipe YAML file '%s'", config.path)
        region_map = atlas.region_map
        brain_regions = atlas.brain_regions
        wmr = load(config.path)

        # Get populations
        LOGGER.debug("Extracting populations from white matter recipe")
        wm_populations = pd.DataFrame.from_records(wmr["populations"])
        wm_populations_sub = wm_populations.loc[
            wm_populations["atlas_region"].apply(lambda x: isinstance(x, list)),
            "atlas_region",
        ]
        if not wm_populations_sub.empty:
            wm_populations_sub = (
                wm_populations_sub.apply(pd.Series)  # type: ignore[call-overload]
                .stack()
                .dropna()
                .rename("atlas_region_split")
                .reset_index(level=1, drop=True)
            )
            wm_populations = wm_populations.join(wm_populations_sub, how="left")
            wm_populations["atlas_region_split"].fillna(
                wm_populations["atlas_region"],
                inplace=True,
            )
            wm_populations.drop(columns=["atlas_region"], inplace=True)
            wm_populations.rename(columns={"atlas_region_split": "atlas_region"}, inplace=True)
        wm_populations.rename(columns={"name": "pop_raw_name"}, inplace=True)
        wm_populations["region_acronym"] = wm_populations["atlas_region"].apply(
            lambda row: row["name"],
        )
        wm_populations_sub = (
            wm_populations["atlas_region"]  # type: ignore[call-overload]
            .apply(lambda row: pd.Series(row.get("subregions", []), dtype=object))
            .stack()
            .dropna()
            .rename("sub_region")
            .reset_index(level=1, drop=True)
        )
        wm_populations = wm_populations.join(wm_populations_sub, how="left")

        # Get subregion names
        wm_populations["formatted_subregion"] = wm_populations["sub_region"]
        if config.subregion_uppercase:
            wm_populations["formatted_subregion"] = wm_populations[
                "formatted_subregion"
            ].str.upper()
        if config.subregion_remove_prefix:
            wm_populations["formatted_subregion"] = wm_populations[
                "formatted_subregion"
            ].str.extract(r"(\d+.*)")
        wm_populations["subregion_acronym"] = (
            wm_populations["region_acronym"]
            + config.sub_region_separator
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
            wm_populations[["atlas_region_id"]],
            on="atlas_region_id",
            how="outer",
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
                            region_map.get(i, WMR_ATLAS_ID)
                            for i in region_map.find(row, attr=WMR_ATLAS_ID, with_descendants=True)
                        ],
                    ),
                )
            ].sum(),
        )
        region_data["atlas_region_volume"] = region_data["nb_voxels"] * brain_regions.voxel_volume
        region_data.drop(columns=["count"], inplace=True)

        # Join region data to population
        wm_populations = wm_populations.merge(region_data, on="atlas_region_id", how="left")

        # Compute volume fractions of sub-regions
        region_ids = wm_populations.apply(
            lambda row: get_atlas_region_id(region_map, row, "region_acronym"),
            axis=1,
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
            wm_layer_profiles["relative_densities"]  # type: ignore[call-overload]
            .apply(pd.Series)
            .stack()
            .rename("layer_profile")
            .reset_index(level=1)
            .rename(columns={"level_1": "layer_profile_num"})
        )
        wm_layer_profiles = wm_layer_profiles.join(layer_profiles).set_index(
            "layer_profile_num",
            append=True,
        )
        wm_layer_profiles["layers"] = wm_layer_profiles["layer_profile"].apply(
            lambda row: row.get("layers", None),
        )
        wm_layer_profiles["value"] = wm_layer_profiles["layer_profile"].apply(
            lambda row: row.get("value", None),
        )
        wm_layer_profiles.drop(columns=["relative_densities", "layer_profile"], inplace=True)
        wm_layer_profiles = wm_layer_profiles.join(
            wm_layer_profiles["layers"]  # type: ignore[call-overload]
            .apply(pd.Series)
            .stack()
            .rename("layer")
            .reset_index(level=2)
            .rename(columns={"level_2": "layer_index"}),
        )
        wm_layer_profiles["formatted_layer"] = wm_layer_profiles["layer"].str.extract("l(.*)")
        wm_layer_profiles.fillna({"formatted_layer": wm_layer_profiles["layer"]}, inplace=True)

        # Get projections
        LOGGER.debug("Extracting projections from white matter recipe")
        wm_projections = pd.DataFrame.from_records(wmr["projections"])
        if wm_projections["source"].duplicated().any():
            msg = (
                "Found several equal sources in the 'projections' entry: "
                f"{sorted(wm_projections.loc[wm_projections['a'].duplicated(), 'a'].tolist())}",
            )
            raise ValueError(msg)

        # Map projections
        wm_projections = wm_projections.merge(
            wm_populations,
            left_on="source",
            right_on="pop_raw_name",
            how="left",
        )

        wm_targets = (
            wm_projections["targets"]  # type: ignore[call-overload]
            .apply(pd.Series)
            .stack()
            .rename("target")
            .reset_index(level=1)
            .rename(columns={"level_1": "target_num"})
        )
        wm_projection_targets = wm_projections.join(wm_targets).set_index("target_num", append=True)
        wm_projection_targets["target_population_name"] = wm_projection_targets["target"].apply(
            lambda row: row["population"],
        )
        wm_projection_targets["target_projection_name"] = wm_projection_targets["target"].apply(
            lambda row: row["projection_name"],
        )
        wm_projection_targets["target_density"] = wm_projection_targets["target"].apply(
            lambda row: row["density"],
        )
        wm_projection_targets["topographical_mapping"] = wm_projection_targets["target"].apply(
            lambda row: row["presynaptic_mapping"],
        )
        wm_projection_targets["target_layer_profiles"] = wm_projection_targets["target"].apply(
            lambda row: row["target_layer_profiles"],
        )
        wm_projection_targets.index.rename("proj_index", level=0, inplace=True)  # type: ignore[call-arg,call-overload]

        # Get target sub regions
        region_map_df = region_map.as_dataframe()
        if "atlas_id" not in region_map_df.columns:
            region_map_df["atlas_id"] = region_map_df.index.to_numpy()
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
        wm_projection_targets.fillna(
            {"target_region": wm_projection_targets["target_population_name"]},
            inplace=True,
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
            sub_region_acronyms,
            left_on="target_region",
            right_index=True,
            how="left",
        )

        selected_sub_region_acronyms = (
            wm_projection_targets["subregion_acronyms"]  # type: ignore[call-overload]
            .apply(pd.Series)
            .stack()
            .rename("subregion_acronym")
            .reset_index(level=2)
            .rename(columns={"level_2": "subregion_num"})
        )
        wm_projection_targets = wm_projection_targets.join(
            selected_sub_region_acronyms,
            rsuffix="_target",
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
                wm_populations["subregion_acronym"].drop_duplicates(),
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
            lambda row: row["target_layer_profiles"][0].get("name"),
            axis=1,
        )
        # wm_projection_targets.drop(columns=["targets", "target_layer_profiles"], inplace=True)
        wm_projection_targets = (
            wm_projection_targets.reset_index()
            .merge(
                wm_layer_profiles[["name", "layer", "formatted_layer", "value"]].rename(
                    columns={"value": "target_layer_profile_density"},
                ),
                left_on=["target_layer_profile_name", "sub_region_target"],
                right_on=["name", "layer"],
                how="left",
            )
            .set_index("index")
        )
        wm_projection_targets.fillna({"target_layer_profile_density": 1}, inplace=True)
        wm_projection_targets["target_layer_profile_prob"] = (
            wm_projection_targets["target_layer_profile_density"]
            * wm_projection_targets["sub_region_volume_frac_target"]
        )

        normalization_factor = (
            wm_projection_targets.groupby(
                ["source", "sub_region", "target_projection_name", "target_region"],
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
        wm_projection_targets.fillna({"target_layer_profile_norm_factor": 1}, inplace=True)
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
        ].isna()

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

        self.populations = wm_populations
        self.projections = wm_projections
        self.targets = wm_targets
        self.fractions = wm_fractions
        self.interaction_strengths = wm_interaction_strengths
        self.projection_targets = wm_projection_targets
        self.layer_profiles = wm_layer_profiles
        self.region_data = region_data
