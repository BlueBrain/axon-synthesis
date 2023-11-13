"""Package to create inputs."""
import logging

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.input_creation import pop_neuron_numbers
from axon_synthesis.input_creation.clustering import cluster_morphologies
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe

LOGGER = logging.getLogger(__name__)


def create_inputs(
    morphology_path,
    wmr_path,
    wmr_subregion_uppercase,
    wmr_subregion_remove_prefix,
    wmr_sub_region_separator,
    atlas_path,
    atlas_region_filename,
    atlas_hierarchy_filename,
    atlas_layer_names,
    neuron_density,
    clustering_parameters,
    output_dir,
    nb_workers=1,
    debug=False,
):
    """Create all inputs required to synthesize long-range axons."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the Atlas
    atlas = AtlasHelper.load(
        atlas_path,
        atlas_region_filename,
        atlas_hierarchy_filename,
        atlas_layer_names,
    )

    # Process the White Matter Recipe
    wmr_dir = output_dir / "WhiteMatterRecipe"
    if wmr_dir.exists() and all(
        (wmr_dir / i).exists() for i in WhiteMatterRecipe.filename.values()
    ):
        LOGGER.info("Loading the White Matter Recipe from '%s' since it already exists", wmr_dir)
        wmr = WhiteMatterRecipe.load(wmr_dir)
    else:
        wmr = WhiteMatterRecipe.from_raw_WMR(
            wmr_path,
            atlas,
            wmr_subregion_uppercase,
            wmr_subregion_remove_prefix,
            wmr_sub_region_separator,
        )
        wmr.save(wmr_dir)

    # Pre-compute atlas data
    brain_regions_mask_file = output_dir / "region_masks.h5"
    if not brain_regions_mask_file.exists():
        atlas.compute_region_masks(brain_regions_mask_file)
    else:
        LOGGER.info(
            "The brain region mask is not computed because it already exists in '%s'",
            brain_regions_mask_file,
        )

    # Compute the expected number of neurons in each brain region
    pop_neuron_numbers_file = output_dir / "neuron_density.csv"
    if not pop_neuron_numbers_file.exists():
        pop_neuron_numbers.compute(
            atlas,
            wmr,
            neuron_density,
            pop_neuron_numbers_file,
        )
    else:
        LOGGER.info(
            "The population neuron numbers are not computed because they already exist in '%s'",
            pop_neuron_numbers_file,
        )

    # Define the tufts and main trunks in input morphologies and compute the properties of the long
    # range trunk and the tufts of each morphology
    clustered_morph_paths, trunk_morph_paths, tuft_morph_paths = cluster_morphologies(
        atlas,
        wmr,
        morphology_path,
        clustering_parameters,
        output_dir,
        debug=debug,
        nb_workers=nb_workers,
    )

    return clustered_morph_paths, trunk_morph_paths, tuft_morph_paths
