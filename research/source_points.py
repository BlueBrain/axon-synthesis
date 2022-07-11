"""Create the source points from the atlas."""
import logging

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from voxcell.nexus.voxelbrain import Atlas

from config import Config

logger = logging.getLogger(__name__)


def _fill_diag(mat, val=1):
    np.fill_diagonal(mat, val)
    return mat


class CreateSourcePoints(luigi_tools.task.WorkflowTask):
    output_dataset = luigi.Parameter(
        description="Output dataset file", default="terminals.csv"
    )
    nb_points = luigi.IntParameter(
        description="The number of random points generated.",
        default=10,
    )
    source_regions = luigi_tools.parameter.OptionalListParameter(
        description="The region used to generate the source points.",
        default=None,
    )
    seed = luigi.IntParameter(
        description="The seed used to generate random points.",
        default=0,
    )

    def run(self):
        # Get config
        config = Config()

        # Get atlas data
        atlas = Atlas.open(str(config.atlas_path))
        logger.debug("Loading brain regions from the atlas")

        logger.debug(
            f"Loading brain regions from the atlas using: {config.atlas_region_filename}"
        )
        brain_regions = atlas.load_data(config.atlas_region_filename)

        logger.debug(
            f"Loading region map from the atlas using: {config.atlas_hierarchy_filename}"
        )
        region_map = atlas.load_region_map(config.atlas_hierarchy_filename)

        rng = np.random.default_rng(self.seed)

        if self.source_regions:
            missing_ids = []
            region_ids = []
            for i in self.source_regions:
                if isinstance(i, int):
                    region_ids.append(i)
                    continue
                new_ids = []
                for j in region_map.find(i, attr="name", with_descendants=True):
                    new_ids.append(j)
                for j in region_map.find(i, attr="acronym", with_descendants=True):
                    new_ids.append(j)
                if not new_ids:
                    missing_ids.append(i)
                else:
                    region_ids.extend(new_ids)

            if missing_ids:
                logger.warning("Could not find the following regions in the atlas: %s", missing_ids)

            potential_voxels = np.argwhere(np.isin(brain_regions.raw, region_ids))
        else:
            potential_voxels = np.argwhere(brain_regions.raw != 0)

        # Get random voxels where the brain region value is not 0
        voxels = rng.choice(potential_voxels, self.nb_points)

        # Compute coordinates of these voxels and add a random component up to the voxel size
        coords = brain_regions.indices_to_positions(voxels)
        coords += np.vstack(
            [
                rng.uniform(
                    -0.5 * np.abs(brain_regions.voxel_dimensions[i]),
                    0.5 * np.abs(brain_regions.voxel_dimensions[i]),
                    size=self.nb_points,
                )
                for i in range(3)
            ]
        ).T

        dataset = pd.DataFrame(coords, columns=["x", "y", "z"])
        dataset.reset_index(inplace=True)
        dataset.rename(columns={"index": "morph_file"}, inplace=True)
        dataset["axon_id"] = 0
        dataset["terminal_id"] = -1
        dataset["section_id"] = -1

        dataset[
            ["morph_file", "axon_id", "terminal_id", "section_id", "x", "y", "z"]
        ].to_csv(self.output()["terminals"].path, index=False)

    def output(self):
        return {
            "terminals": TaggedOutputLocalTarget(
                self.output_dataset, create_parent=True
            ),
        }
