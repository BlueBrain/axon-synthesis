"""Package to create inputs."""
import logging

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.input_creation import pop_neuron_numbers
from axon_synthesis.input_creation.clustering import cluster_morphologies
from axon_synthesis.input_creation.inputs import Inputs
from axon_synthesis.typing import FileType
from axon_synthesis.white_matter_recipe import WmrConfig

LOGGER = logging.getLogger(__name__)


def create_inputs(
    morphology_path: FileType,
    wmr_config: WmrConfig,
    atlas_config: AtlasConfig,
    neuron_density,
    clustering_parameters,
    output_dir: FileType,
    *,
    nb_workers=1,
    debug=False,
):
    """Create all inputs required to synthesize long-range axons."""
    inputs = Inputs(output_dir, morphology_path)
    inputs.create_root()

    # Load the Atlas
    inputs.load_atlas(atlas_config)

    # Process the White Matter Recipe
    inputs.load_wmr(wmr_config)

    # Pre-compute atlas data
    inputs.atlas.compute_region_masks(inputs.BRAIN_REGIONS_MASK_FILENAME)

    # Compute the expected number of neurons in each brain region
    pop_neuron_numbers.compute(
        inputs.wmr.populations,
        neuron_density,
        inputs.POP_NEURON_NUMBERS_FILENAME,
    )

    # Define the tufts and main trunks in input morphologies and compute the properties of the long
    # range trunk and the tufts of each morphology
    inputs.clustering_data = cluster_morphologies(
        inputs.atlas,
        inputs.wmr,
        inputs.MORPHOLOGY_DIRNAME,
        clustering_parameters,
        inputs.CLUSTERING_DIRNAME,
        debug=debug,
        nb_workers=nb_workers,
    )

    # Export the input metadata
    inputs.save_metadata()

    return inputs
