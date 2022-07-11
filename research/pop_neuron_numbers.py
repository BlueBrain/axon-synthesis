"""Create number of neurons in each population."""
import logging
import sys

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from atlas import load as load_atlas
from config import Config
from data_validation_framework.target import TaggedOutputLocalTarget
from source_points import CreateSourcePoints
from target_points import FindTargetPoints
from white_matter_recipe import load as load_wmr
from white_matter_recipe import process as process_wmr

logger = logging.getLogger(__name__)


def _fill_diag(mat, val=1):
    np.fill_diagonal(mat, val)
    return mat


class PopNeuronNumbersOutputLocalTarget(TaggedOutputLocalTarget):
    __prefix = "pop_neuron_numbers"


class PickPopulationNeuronNumbers(luigi_tools.task.WorkflowTask):
    populations = luigi_tools.parameter.OptionalPathParameter(
        description="Path to the populations CSV file.",
        default=None,
        exists=True,
    )
    output_dataset = luigi.Parameter(
        description="Output dataset file", default="population_neuron_numbers.csv"
    )
    seed = luigi.IntParameter(
        description="The seed used to generate random points.",
        default=0,
    )
    neuron_density = luigi.NumericalParameter(
        description=(
            "The density of neurons in the atlas (we suppose here that this density is uniform). "
            "This density should be given in number of neurons by cube atlas-unit (usually "
            "micrometer)."
        ),
        var_type=float,
        min_value=0,
        max_value=sys.float_info.max,
        default=1e-2,
        left_op=luigi.parameter.operator.lt,
    )
    debug = luigi.BoolParameter(
        description=("If set to True, the debug mode is enabled."),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
    )

    def requires(self):
        return FindTargetPoints()

    def run(self):
        config = Config()

        populations = pd.read_csv(self.input()["wm_populations"].path)

        # Get atlas data
        _, brain_regions, region_map = load_atlas(
            str(config.atlas_path),
            config.atlas_region_filename,
            config.atlas_hierarchy_filename,
        )

        # Compute the volume of each region
        region_ids, region_counts = np.unique(brain_regions.raw, return_counts=True)

        populations["atlas_region_volume"] = (
            populations["atlas_region_id"].apply(
                lambda row: region_counts[
                    np.argwhere(
                        np.isin(
                            region_ids,
                            list(
                                region_map.find(row, attr="id", with_descendants=True)
                            ),
                        )
                    )
                ].sum()
            )
            * brain_regions.voxel_volume
        )

        # Compute the number of neurons in each population
        populations["pop_neuron_numbers"] = (
            populations["atlas_region_volume"] * self.neuron_density
        ).astype(int)

        # Format the population numbers
        populations.loc[populations["pop_neuron_numbers"] < 2, "pop_neuron_numbers"] = 2

        # Export the results
        populations[
            [
                "pop_raw_name",
                "atlas_region_id",
                "atlas_region_volume",
                "pop_neuron_numbers",
            ]
        ].to_csv(self.output()["population_numbers"].path, index=False)

    def output(self):
        targets = {
            "population_numbers": PopNeuronNumbersOutputLocalTarget(
                self.output_dataset, create_parent=True
            ),
        }
        return targets
