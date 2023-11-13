"""Helpers for atlas."""
import logging
import operator
from itertools import chain
from pathlib import Path

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import h5py
import numpy as np
import pandas as pd
from voxcell import RegionMap
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas

from axon_synthesis.typing import FileType
from axon_synthesis.typing import LayerNamesType

LOGGER = logging.getLogger(__name__)


def _is_in(test_elements, brain_regions):
    res = np.zeros_like(brain_regions, dtype=bool)
    for i in test_elements:
        res |= brain_regions == i
    return res


class AtlasHelper:
    """Atlas helper."""

    def __init__(
        self,
        atlas: Atlas,
        brain_regions: VoxelData,
        region_map: RegionMap,
        layers_names: LayerNamesType = None,
    ):
        """The AtlasHelper constructor.

        Args:
            atlas: The atlas.
            brain_regions: The brain regions.
            region_map: The brain region hierarchy.
            layer_names: The list of layer names.
        """
        self.atlas = atlas
        self.brain_regions = brain_regions
        self.region_map = region_map
        self.layers = layers_names if layers_names is not None else list(range(1, 7))
        self.top_layer = atlas.load_data(f"[PH]{self.layers[0]}")

        # TODO: Compute the depth for specific layers of each region (like in region-grower)
        self.depths = VoxelData.reduce(operator.sub, [self.pia_coord, atlas.load_data("[PH]y")])

    @classmethod
    def load(
        cls,
        atlas_path: FileType,
        atlas_region_filename: FileType,
        atlas_hierarchy_filename: FileType,
        layers_names: LayerNamesType = None,
        # atlas_flatmap_filename: str = None,
    ) -> Self:
        """Read Atlas data from directory."""
        # Get atlas data
        LOGGER.info("Loading atlas from: %s", atlas_path)
        atlas = Atlas.open(atlas_path)

        atlas_region_filename = Path(atlas_region_filename).with_suffix(".nrrd")
        LOGGER.debug("Loading brain regions from the atlas using: %s", atlas_region_filename.name)
        brain_regions = atlas.load_data(atlas_region_filename.stem)

        atlas_hierarchy_filename = Path(atlas_hierarchy_filename).with_suffix(".json").name
        LOGGER.debug("Loading region map from the atlas using: %s", atlas_hierarchy_filename)
        region_map = atlas.load_region_map(atlas_hierarchy_filename)

        # if config.atlas_flatmap_filename is None:
        #     # Create the flatmap of the atlas
        #     LOGGER.debug("Building flatmap")
        #     one_layer_flatmap = np.mgrid[
        #         : brain_regions.raw.shape[2],
        #         : brain_regions.raw.shape[0],
        #     ].T[:, :, ::-1]
        #     flatmap = VoxelData(
        #         np.stack([one_layer_flatmap] * brain_regions.raw.shape[1], axis=1),
        #         voxel_dimensions=brain_regions.voxel_dimensions,
        #     )
        # else:
        #     # Load the flatmap of the atlas
        #     flatmap = atlas.load_data(config.atlas_flatmap_filename)

        # if self.debug_flatmap:
        #     LOGGER.debug(f"Saving flatmap to: {self.output()['flatmap'].path}")
        #     flatmap.save_nrrd(self.output()["flatmap"].path, encoding="raw")

        return cls(atlas, brain_regions, region_map, layers_names)

    @property
    def pia_coord(self) -> VoxelData:
        """Return an atlas of the pia coordinate along the principal axis."""
        return self.top_layer.with_data(self.top_layer.raw[..., 1])

    def compute_region_masks(self, output_path: FileType):
        """Compute all region masks."""
        Path(output_path).mkdir(parents=True, exist_ok=True)

        region_map_df = self.region_map.as_dataframe()
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
            lambda row: tuple(sorted(self.region_map.find(row, attr="id", with_descendants=True)))
        )

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

        # TODO: Maybe we can keep all the masks in memory? It's just a set of lists of ints.
        with h5py.File(output_path, "w") as f:
            for atlas_id, self_and_descendants_atlas_ids in (
                region_map_df.loc[
                    ~region_map_df["atlas_id"].isnull(),
                    ["atlas_id", "self_and_descendants_atlas_ids"],
                ]
                .astype({"atlas_id": int})
                .drop_duplicates(subset=["atlas_id"])
                .to_records(index=False)
            ):
                LOGGER.debug("Create mask for %s", atlas_id)
                mask = _is_in(self_and_descendants_atlas_ids, self.brain_regions.raw)
                if not mask.any():
                    raw_ids = sorted(
                        chain(
                            *region_map_df.loc[
                                region_map_df["atlas_id"] == atlas_id, "self_and_descendants"
                            ].tolist()
                        )
                    )
                    mask = _is_in(raw_ids, self.brain_regions.raw)
                    LOGGER.warning(
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

        LOGGER.info("Masks written in %s", output_path)
