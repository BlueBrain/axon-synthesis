"""Create number of neurons in each population."""
import logging
import sys

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget

from axon_synthesis.atlas import load as load_atlas
from axon_synthesis.config import Config
from axon_synthesis.target_points import FindTargetPoints

logger = logging.getLogger(__name__)


class PopNeuronNumbersOutputLocalTarget(TaggedOutputLocalTarget):
    """Target for tuft outputs."""

    __prefix = "pop_neuron_numbers"  # pylint: disable=unused-private-member


class PickPopulationNeuronNumbers(luigi_tools.task.WorkflowTask):
    """Task to compute the number of neurons in each population."""

    populations = luigi.parameter.OptionalPathParameter(
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
                            list(region_map.find(row, attr="id", with_descendants=True)),
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
