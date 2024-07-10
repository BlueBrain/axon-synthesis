"""Helpers for atlas."""
import contextlib
import copy
import json
import logging
import operator
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from attrs import asdict
from attrs import define
from attrs import field
from voxcell import OrientationField
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas
from voxcell.voxel_data import ValueToIndexVoxels

from axon_synthesis.typing import ArrayLike
from axon_synthesis.typing import FileType
from axon_synthesis.typing import LayerNamesType
from axon_synthesis.typing import RegionIdsType
from axon_synthesis.typing import SeedType
from axon_synthesis.typing import Self

LOGGER = logging.getLogger(__name__)


def _is_in(test_elements: list, brain_regions: ArrayLike) -> np.ndarray:
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
        layer_names: The list of layer names.
    """

    path: Path = field(converter=Path)
    region_filename: Path = field(converter=Path)
    # flatmap_filename: Path = field(converter=Path)
    layer_names: LayerNamesType | None = field(default=None)
    load_region_map: bool = field(default=False)

    def to_dict(self) -> dict:
        """Return all attribute values into a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a new AtlasConfig object from a dictionary."""
        return cls(
            data["path"],
            data["region_filename"],
            # data["flatmap_filename"],
            data.get("layer_names", None),
        )


class AtlasHelper:
    """Atlas helper."""

    def __init__(
        self: Self,
        config: AtlasConfig,
        logger=None,
    ):
        """Create a new BasePathBuilder object.

        Args:
            config: The configuration used to load the atlas.
            logger: An optional logger.
        """
        if logger is None:
            logger = LOGGER
        self.config = config

        # Get atlas data
        logger.info("Loading atlas with the following config: %s", self.config)
        atlas = Atlas.open(str(self.config.path.resolve()))

        logger.debug(
            "Loading brain regions from the atlas using: %s", self.config.region_filename.name
        )
        self.brain_regions = atlas.load_data(self.config.region_filename.stem)
        self.orientations = atlas.load_data("orientation", cls=OrientationField)

        if self.config.load_region_map:
            logger.debug("Loading region map from the atlas")
            self.region_map = atlas.load_region_map()
            self._build_region_map_df_level()
        else:
            self.region_map = None

        self.layers = (
            self.config.layer_names if self.config.layer_names is not None else list(range(1, 7))
        )
        self.top_layer = atlas.load_data(f"[PH]{self.layers[0]}")

        # if config.atlas_flatmap_filename is None:
        #     # Create the flatmap of the atlas
        #     logger.debug("Building flatmap")
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
        #     logger.debug(f"Saving flatmap to: {self.output()['flatmap'].path}")
        #     flatmap.save_nrrd(self.output()["flatmap"].path, encoding="raw")

        # TODO: Compute the depth for specific layers of each region (like in region-grower)
        self.y = atlas.load_data("[PH]y")
        self.depths = VoxelData.reduce(operator.sub, [self.pia_coord, self.y])

    def save(self, path):
        """Export the atlas to the given directory."""
        path = Path(path)
        self.brain_regions.save_nrrd(str(path / "brain_regions.nrrd"))
        self.orientations.save_nrrd(str(path / "orientations.nrrd"))
        self.top_layer.save_nrrd(str(path / f"[PH]{self.layers[0]}.nrrd"))
        self.y.save_nrrd(str(path / "[PH]y.nrrd"))
        with (path / "hierarchy.json").open(mode="w", encoding="utf-8") as f:
            json.dump(self.region_map.as_dict(), f, indent=4)

    def copy(self):
        """Return a copy of the current atlas."""
        return copy.deepcopy(self)

    @property
    def region_map(self):
        """Return the region map."""
        return self._region_map

    @region_map.setter
    def region_map(self, value) -> None:
        """Setter for the region map."""
        self._region_map = value
        if value is None:
            self._region_map_df = None
        else:
            self._region_map_df = value.as_dataframe()
            self._build_region_map_df_level()

    @property
    def region_map_df(self):
        """Return the DF representation of the region map."""
        return self._region_map_df

    @cached_property
    def pia_coord(self) -> VoxelData:
        """Return an atlas of the pia coordinate along the principal axis."""
        return self.top_layer.with_data(self.top_layer.raw[..., 1])

    def compute_region_masks(self, output_path: FileType):
        """Compute all region masks."""
        if self.region_map is None:
            msg = "The region_map is not loaded so the region masks can't be computed"
            raise RuntimeError(msg)
        LOGGER.info("Computing brain region masks")
        output_path = Path(output_path)

        if output_path.exists():
            LOGGER.info(
                "The brain region mask is not computed because it already exists in '%s'",
                output_path,
            )
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        region_map_df = (
            self.region_map_df.reset_index()
            .merge(
                self.region_map_df[["acronym"]].reset_index(),
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

        # TODO: Maybe we can keep all the masks in memory? It's just a set of lists of ints.
        with h5py.File(output_path, "w") as f:
            index = ValueToIndexVoxels(self.brain_regions.raw)
            for current_id, self_and_descendants_atlas_ids in region_map_df[
                ["self_and_descendants"]
            ].to_records(index=True):
                LOGGER.debug("Create mask for %s", current_id)
                coords = index.value_to_indices(self_and_descendants_atlas_ids)
                if len(coords) == 0:
                    LOGGER.warning(
                        "No voxel found for atlas ID %s using the following descendants: %s",
                        current_id,
                        self_and_descendants_atlas_ids,
                    )
                f.create_dataset(
                    str(current_id), data=coords, compression="gzip", compression_opts=1
                )

        LOGGER.info("Masks exported to %s", output_path)

    def get_region_ids(
        self,
        brain_region_names: RegionIdsType,
        *,
        with_descendants: bool = True,
    ) -> tuple[list[str | int], list[str | int]]:
        """Find brain region IDs from their names, acronyms or direct IDs.

        Args:
            brain_region_names: The names of the brain regions to get IDs.
            with_descendants: If set to True, all the descendants are included.
        """
        if self.region_map is None:
            msg = "The region_map is not loaded so it's not possible to get region IDs"
            raise RuntimeError(msg)

        if isinstance(brain_region_names, int | str):
            brain_region_names = [brain_region_names]
        missing_ids = []
        brain_region_ids = []

        for i in brain_region_names:
            if isinstance(i, str):
                with contextlib.suppress(ValueError):
                    i = int(i)  # noqa: PLW2901

            new_ids = []

            if isinstance(i, int):
                new_ids.extend(
                    list(self.region_map.find(i, attr="id", with_descendants=with_descendants))
                )
            else:
                new_ids.extend(
                    list(self.region_map.find(i, attr="name", with_descendants=with_descendants))
                )
                new_ids.extend(
                    list(
                        self.region_map.find(i, attr="acronym", with_descendants=with_descendants)
                    ),
                )

            if not new_ids:
                missing_ids.append(i)
            else:
                brain_region_ids.extend(new_ids)

        return sorted(set(brain_region_ids)), sorted(set(missing_ids))

    def get_region_voxels(
        self,
        brain_region_names: RegionIdsType,
        *,
        inverse: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[str | int]]:
        """Get the coordinates of the voxels located in the given regions from the atlas."""
        brain_region_ids: list[str | int]
        missing_ids: list[str | int]
        brain_region_ids, missing_ids = self.get_region_ids(brain_region_names)

        brain_region_mask = np.isin(
            self.brain_regions.raw, list(set(brain_region_ids).union(missing_ids))
        )
        if inverse:
            brain_region_mask = ~brain_region_mask

        brain_regions_coords = np.argwhere(brain_region_mask)

        return brain_regions_coords, missing_ids

    def get_random_voxel_shifts(self, size, *, rng: SeedType = None):
        """Pick random shifts from voxel centers according to the voxel sizes."""
        rng = np.random.default_rng(rng)
        half_voxels = self.brain_regions.voxel_dimensions / 2
        return rng.uniform(-half_voxels, half_voxels, size=(size, 3))

    def get_region_points(
        self,
        brain_region_names: RegionIdsType,
        *,
        size: int | None = None,
        random_shifts: bool = False,
        inverse: bool = False,
        rng: SeedType = None,
    ) -> np.ndarray | tuple[np.ndarray, list[str | int]]:
        """Extract region points from the atlas.

        If 'random_shifts' is set to False (the default), the voxel centers are returned. If
        'random_shifts' is set to True, one random point inside each of these voxels (chosen using
        the given Random Number Generator) are returned.

        Args:
            brain_region_names: The name of the regions to consider.
            size: The number of points to return (they are chosen randomly along the possible ones).
            random_shifts: If True, a random shift is applied to return a random point inside the
                voxel.
            inverse: If True, choose points that are NOT located in the given regions.
            rng: The random number generator (can be an int used as seed or a numpy Generator).
        """
        if size is not None and size <= 0:
            msg = "The 'size' argument must be a positive integer."
            raise ValueError(msg)

        brain_regions_coords, missing_ids = self.get_region_voxels(
            brain_region_names, inverse=inverse
        )

        voxel_points = self.brain_regions.indices_to_positions(
            brain_regions_coords + [0.5, 0.5, 0.5]  # noqa: RUF005
        )

        if size is not None:
            # Pick the given number of voxels (the voxels are picked before applying shifts so the
            # duplicated voxels have different shifts)
            rng = np.random.default_rng(rng)
            voxel_points = rng.choice(voxel_points, size)

        if random_shifts:
            # Pick a random point inside the voxel
            rng = np.random.default_rng(rng)
            voxel_points += self.get_random_voxel_shifts(len(voxel_points), rng=rng)

        return voxel_points, missing_ids

    def _build_region_map_df_level(self) -> None:
        """Check if region_map_df contains the st_level column and build it if it's missing."""
        if "st_level" in self.region_map_df.columns:
            return
        current_level = -1
        self.region_map_df["st_level"] = current_level
        current_level_ids = self.region_map_df.loc[
            ~self.region_map_df["parent_id"].isin(self.region_map_df.index)
        ]
        while not current_level_ids.empty:
            current_level += 1
            self.region_map_df.loc[current_level_ids.index, "st_level"] = current_level
            current_level_ids = self.region_map_df.loc[
                self.region_map_df["parent_id"].isin(current_level_ids.index)
            ]

    @cached_property
    def brain_regions_and_descendants(self) -> pd.DataFrame:
        """Return the brain regions and their descendants from the hierarchy."""
        if self.region_map is None:
            msg = "The region_map is not loaded so the descendants can't be computed"
            raise RuntimeError(msg)
        return (
            self.region_map_df.index.to_series()
            .apply(
                lambda row: tuple(
                    sorted(self.region_map.find(row, attr="id", with_descendants=True))
                )
            )
            .apply(pd.Series)
            .stack()
            .dropna()
            .astype(int)
            .rename("youth_id")
            .to_frame()
            .merge(self.region_map_df[["st_level"]], left_on="youth_id", right_index=True)
            .reset_index()
            .drop(columns="level_1")
            .sort_values(["id", "st_level"], ascending=[True, False])
            .reset_index(drop=True)
        )

    @cached_property
    def brain_regions_and_ascendants(self) -> pd.DataFrame:
        """Return the brain regions and their ascendants from the hierarchy."""
        if self.region_map is None:
            msg = "The region_map is not loaded so the ascendants can't be computed"
            raise RuntimeError(msg)
        return (
            self.region_map_df.index.to_series()
            .apply(
                lambda row: tuple(sorted(self.region_map.get(row, attr="id", with_ascendants=True)))
            )
            .apply(pd.Series)
            .stack()
            .dropna()
            .astype(int)
            .rename("elder_id")
            .to_frame()
            .merge(self.region_map_df[["st_level"]], left_on="elder_id", right_index=True)
            .reset_index()
            .drop(columns="level_1")
            .sort_values(["id", "st_level"], ascending=[True, False])
            .reset_index(drop=True)
        )
