"""Package to create inputs."""
from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.input_creation import pop_neuron_numbers
from axon_synthesis.input_creation.clustering import cluster_morphologies
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe


def create_inputs(
    morphology_path,
    WMR_path,
    WMR_subregion_uppercase,
    WMR_subregion_remove_prefix,
    WMR_sub_region_separator,
    atlas_path,
    atlas_region_file_name,
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
        atlas_region_file_name,
        atlas_hierarchy_filename,
        atlas_layer_names,
    )

    # Process the White Matter Recipe
    wmr = WhiteMatterRecipe.from_raw_wmr(
        WMR_path,
        atlas,
        WMR_subregion_uppercase,
        WMR_subregion_remove_prefix,
        WMR_sub_region_separator,
    )
    wmr.save(output_dir / "WhiteMatterRecipe")

    # Pre-compute atlas data
    atlas.compute_region_masks(output_dir)

    # Compute the expected number of neurons in each brain region
    pop_neuron_numbers.compute(
        atlas,
        wmr,
        neuron_density,
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
