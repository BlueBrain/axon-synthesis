"""Create the source points from the atlas."""
import logging

import luigi
import luigi_tools
import numpy as np
import pandas as pd
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
    seed = luigi.IntParameter(
        description="The seed used to generate random points.",
        default=0,
    )

    def run(self):
        # Get atlas data
        atlas = Atlas.open(str(Config().atlas_path))
        logger.debug("Loading brain regions from the atlas")
        brain_regions = atlas.load_data(Config().atlas_region_filename)

        rng = np.random.default_rng(self.seed)

        # Get random voxels where the brain region value is not 0
        voxels = rng.choice(np.argwhere(brain_regions.raw != 0), self.nb_points)

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
            "terminals": luigi_tools.target.OutputLocalTarget(
                self.output_dataset, create_parent=True
            ),
        }
