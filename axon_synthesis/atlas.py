"""Helpers for atlas."""
import logging
import operator
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from attrs import asdict
from attrs import define
from attrs import field
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas

from axon_synthesis.typing import FileType
from axon_synthesis.typing import LayerNamesType
from axon_synthesis.typing import Self

LOGGER = logging.getLogger(__name__)


def _is_in(test_elements: list, brain_regions: np.ndarray) -> np.ndarray:
    res = np.zeros_like(brain_regions, dtype=bool)
    for i in test_elements:
        res |= brain_regions == i
    return res


@define
class AtlasConfig:
    """Class to store the Atlas configuration.

    Attributes:
        path: The path to the directory containing the atlas.
        region_filename: The name of the file containing the brain regions.
        hierarchy_filename: The name of the file containing the brain region hierarchy.
        layer_names: The list of layer names.
    """

    path: Path = field(converter=Path)
    region_filename: Path = field(converter=Path)
    hierarchy_filename: Path = field(converter=Path)
    # flatmap_filename: Path = field(converter=Path)
    layer_names: LayerNamesType

    def to_dict(self) -> dict:
        """Return all attribute values into a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a new AtlasConfig object from a dictionary."""
        return cls(
            data["path"],
            data["region_filename"],
            data["hierarchy_filename"],
            # data["flatmap_filename"],
            data.get("layer_names", None),
        )


class AtlasHelper:
    """Atlas helper."""

    def __init__(
        self: Self,
        config: AtlasConfig,
    ):
        """Create a new BasePathBuilder object.

        Args:
            config: The configuration used to load the atlas.
        """
        self.config = config

        # Get atlas data
        LOGGER.info("Loading atlas from: %s", self.config.path)
        self.atlas = Atlas.open(str(self.config.path.resolve()))

        LOGGER.debug(
            "Loading brain regions from the atlas using: %s", self.config.region_filename.name
        )
        self.brain_regions = self.atlas.load_data(self.config.region_filename.stem)

        LOGGER.debug("Loading region map from the atlas using: %s", self.config.hierarchy_filename)
        self.region_map = self.atlas.load_region_map(self.config.hierarchy_filename.name)

        self.layers = (
            self.config.layer_names if self.config.layer_names is not None else list(range(1, 7))
        )
        self.top_layer = self.atlas.load_data(f"[PH]{self.layers[0]}")

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

        # TODO: Compute the depth for specific layers of each region (like in region-grower)
        self.depths = VoxelData.reduce(
            operator.sub, [self.pia_coord, self.atlas.load_data("[PH]y")]
        )

    @property
    def pia_coord(self) -> VoxelData:
        """Return an atlas of the pia coordinate along the principal axis."""
        return self.top_layer.with_data(self.top_layer.raw[..., 1])

    def compute_region_masks(self, output_path: FileType):
        """Compute all region masks."""
        LOGGER.info("Computing brain region masks")
        output_path = Path(output_path)

        if output_path.exists():
            LOGGER.info(
                "The brain region mask is not computed because it already exists in '%s'",
                output_path,
            )
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

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
            lambda row: tuple(sorted(self.region_map.find(row, attr="id", with_descendants=True))),
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
            lambda row: tuple(set(row)),
        )
        region_map_df["self_and_descendants_atlas_ids"] = (
            atlas_id_mapping.dropna()
            .astype(int)
            .reset_index()
            .groupby("id")["atlas_id"]
            .apply(lambda row: tuple(set(row)))
        )
        region_map_df["self_and_descendants_atlas_ids"].fillna(
            {i: () for i in region_map_df.index},
            inplace=True,
        )
        region_map_df.sort_values("atlas_id", inplace=True)

        # TODO: Maybe we can keep all the masks in memory? It's just a set of lists of ints.
        with h5py.File(output_path, "w") as f:
            for atlas_id, self_and_descendants_atlas_ids in (
                region_map_df.loc[
                    ~region_map_df["atlas_id"].isna(),
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
                                region_map_df["atlas_id"] == atlas_id,
                                "self_and_descendants",
                            ].tolist(),
                        ),
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

        LOGGER.info("Masks exported to %s", output_path)
