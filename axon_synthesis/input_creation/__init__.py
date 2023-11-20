"""Package to create inputs."""
import logging

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.input_creation import pop_neuron_numbers
from axon_synthesis.input_creation.clustering import cluster_morphologies
from axon_synthesis.input_creation.inputs import Inputs
from axon_synthesis.typing import FileType
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe
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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the Atlas
    inputs.load_atlas(atlas_config)

    # Process the White Matter Recipe
    if WhiteMatterRecipe.exists(inputs.WMR_DIRNAME):
        LOGGER.info(
            "Loading the White Matter Recipe from '%s' since it already exists", inputs.WMR_DIRNAME
        )
        inputs.load_wmr()
    else:
        inputs.wmr = WhiteMatterRecipe.from_raw_wmr(
            wmr_config,
            inputs.atlas,
        )
        inputs.wmr.save(inputs.WMR_DIRNAME)

    # Pre-compute atlas data
    if not inputs.BRAIN_REGIONS_MASK_FILENAME.exists():
        inputs.atlas.compute_region_masks(inputs.BRAIN_REGIONS_MASK_FILENAME)
    else:
        LOGGER.info(
            "The brain region mask is not computed because it already exists in '%s'",
            inputs.BRAIN_REGIONS_MASK_FILENAME,
        )

    # Compute the expected number of neurons in each brain region
    if not inputs.POP_NEURON_NUMBERS_FILENAME.exists():
        pop_neuron_numbers.compute(
            inputs.wmr.populations,
            neuron_density,
            inputs.POP_NEURON_NUMBERS_FILENAME,
        )
    else:
        LOGGER.info(
            "The population neuron numbers are not computed because they already exist in '%s'",
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
