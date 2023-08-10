"""Find the target points of the input morphologies."""
import logging
from itertools import chain

import h5py
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget

from axon_synthesis.atlas import load as load_atlas
from axon_synthesis.config import Config
from axon_synthesis.create_dataset import FetchWhiteMatterRecipe
from axon_synthesis.white_matter_recipe import load_WMR_data

logger = logging.getLogger(__name__)


def _is_in(test_elements, brain_regions):
    res = np.zeros_like(brain_regions, dtype=bool)
    for i in test_elements:
        res |= brain_regions == i
    return res


class PrepareAtlas(luigi_tools.task.WorkflowTask):
    """Task to find the target points used for axon synthesis."""

    # Attributes that are populated in the run() method
    wm_projection_targets = None
    region_data = None

    def requires(self):
        return {
            "WMR": FetchWhiteMatterRecipe(),
        }

    def load_WMR(self):
        """Get the white matter recipe data."""
        data = load_WMR_data(
            wm_projection_targets_path=self.input()["WMR"]["wm_projection_targets"].pathlib_path,
            region_data_path=self.input()["WMR"]["region_data"].pathlib_path,
        )
        self.wm_projection_targets = data["wm_projection_targets"]
        self.region_data = data["region_data"]

    def run(self):
        config = Config()

        # Get atlas data
        _, brain_regions, region_map = load_atlas(
            str(config.atlas_path),
            config.atlas_region_filename,
            config.atlas_hierarchy_filename,
        )

        # Get the white matter recipe data
        self.load_WMR()

        projection_targets = self.wm_projection_targets.loc[
            ~self.wm_projection_targets["target_region_atlas_id"].isnull()
        ]
        projection_targets = projection_targets.fillna(
            {"target_subregion_atlas_id": projection_targets["target_region_atlas_id"]}
        ).astype({"target_region_atlas_id": int, "target_subregion_atlas_id": int})
        projection_targets = projection_targets[
            [
                "pop_raw_name",
                "target_density",
                "target_population_name",
                "target_projection_name",
                "target_region",
                "target_subregion_acronym",
                "target_subregion_atlas_id",
                "target_layer_profile_region_prob",
            ]
        ]

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
        region_map_df["self_and_descendants"] = region_map_df.index.to_series().apply(
            lambda row: tuple(sorted(region_map.find(row, attr="id", with_descendants=True)))
        )

        # region_atlas_ids = (
        #     region_map_df.dropna(subset=["atlas_id"])
        #     .astype(dtype={"atlas_id": int})
        #     .groupby("atlas_id")
        #     .groups
        # )
        self_and_descendants = (
            region_map_df["self_and_descendants"]
            .apply(pd.Series)
            .stack()
            .dropna()
            .astype(int)
            .rename("self_and_descendants")
        )

        atlas_id_mapping = pd.merge(
            self_and_descendants,
            region_map_df[["atlas_id"]],
            left_on="self_and_descendants",
            right_index=True,
            how="left",
        )
        atlas_id_mapping.dropna().astype(int).reset_index().groupby("id")["atlas_id"].apply(
            lambda row: tuple(set(row))
        )
        region_map_df["self_and_descendants_atlas_ids"] = (
            atlas_id_mapping.dropna()
            .astype(int)
            .reset_index()
            .groupby("id")["atlas_id"]
            .apply(lambda row: tuple(set(row)))
        )
        region_map_df["self_and_descendants_atlas_ids"].fillna(
            {i: tuple() for i in region_map_df.index}, inplace=True
        )
        region_map_df.sort_values("atlas_id", inplace=True)

        with h5py.File(self.output().path, "w") as f:
            for atlas_id, self_and_descendants_atlas_ids in (
                region_map_df.loc[
                    ~region_map_df["atlas_id"].isnull(),
                    ["atlas_id", "self_and_descendants_atlas_ids"],
                ]
                .astype({"atlas_id": int})
                .drop_duplicates(subset=["atlas_id"])
                .to_records(index=False)
            ):
                logger.debug("Create mask for %s", atlas_id)
                mask = _is_in(self_and_descendants_atlas_ids, brain_regions.raw)
                if not mask.any():
                    raw_ids = sorted(
                        chain(
                            *region_map_df.loc[
                                region_map_df["atlas_id"] == atlas_id, "self_and_descendants"
                            ].tolist()
                        )
                    )
                    mask = _is_in(raw_ids, brain_regions.raw)
                    logger.warning(
                        (
                            "No voxel found for atlas ID %s, "
                            "found %s voxels using the following raw IDs: %s"
                        ),
                        self_and_descendants_atlas_ids,
                        mask.sum(),
                        raw_ids,
                    )
                coords = np.argwhere(mask)
                f.create_dataset(str(atlas_id), data=coords, compression="gzip", compression_opts=9)

        logger.info("Masks written in %s", self.output().path)

    def output(self):
        return TaggedOutputLocalTarget("atlas_region_masks.h5", create_parent=True)
