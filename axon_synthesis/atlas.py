"""Helpers for atlas."""
import logging
from functools import lru_cache

# import numpy as np
# from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas

logger = logging.getLogger(__name__)


@lru_cache
def load(atlas_path: str, atlas_region_filename: str, atlas_hierarchy_filename: str):
    """Read Atlas data from directory."""
    # Get atlas data
    logger.info("Loading atlas from: %s", atlas_path)
    atlas = Atlas.open(atlas_path)

    logger.debug("Loading brain regions from the atlas using: %s", atlas_region_filename)
    brain_regions = atlas.load_data(atlas_region_filename)

    logger.debug("Loading region map from the atlas using: %s", atlas_hierarchy_filename)
    region_map = atlas.load_region_map(atlas_hierarchy_filename)

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

    return atlas, brain_regions, region_map
