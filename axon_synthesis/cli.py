"""Command Line Interface for the axon_synthesis package."""
import functools
import json
import logging
import os
import sys
from pathlib import Path

import click
from click_option_group import optgroup
from configobj import ConfigObj

from axon_synthesis import input_creation
from axon_synthesis.utils import setup_logger
from axon_synthesis.white_matter_recipe import fetch


def configure(ctx, param, filename):
    """Set parameter default values according to a given configuration file."""
    # pylint: disable=unused-argument
    if filename is None:
        return

    # Load the config file
    cfg = ConfigObj(filename)

    # Get current default values
    defaults = cfg.dict()

    def format_value(data, name):
        return {f"{name}_{key}": value for key, value in data.items()}

    for key, value in defaults.items():
        if isinstance(value, dict):
            to_add = {}
            to_remove = []
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    to_add.update(format_value(subvalue, subkey))
                    to_remove.append(subkey)
            value.update(to_add)
            for i in to_remove:
                del value[i]

    ctx.default_map = defaults


class GlobalConfig:
    """Class to store global configuration."""

    def __init__(self, debug=False):
        self.debug = debug


class ListParam(click.ParamType):
    """A `click` parameter to process parameters given as JSON arrays."""

    name = "list"

    def convert(self, value, param, ctx):
        """Convert a given value."""
        try:
            if not isinstance(value, list):
                value = json.loads(value)
        except json.JSONDecodeError:
            self.fail(f"{value!r} is not a valid JSON array", param, ctx)

        return value


def atlas_options(func):
    """Decorate a click command to add Atlas-specific options."""

    @optgroup.group("Atlas parameters", help="Parameters used to read and prepare the atlas")
    @optgroup.option(
        "--atlas-path",
        type=click.Path(exists=True, file_okay=False),
        required=True,
        help="Path to the Atlas folder",
    )
    @optgroup.option(
        "--atlas-region-filename",
        type=str,
        required=True,
        help="Name of NRRD file containing the brain regions in the Atlas folder",
    )
    @optgroup.option(
        "--atlas-hierarchy-filename",
        type=str,
        required=True,
        help="Name of file containing the brain region hierarchy in the Atlas folder",
    )
    @optgroup.option(
        "--atlas-layer-names",
        type=ListParam(),
        help=(
            "Names of the layers given as a JSON list (the atlas folder must contain a file name "
            "'[PH]<layer_name>.nrrd' for each given layer)"
        ),
    )
    @functools.wraps(func)
    def wrapper_atlas_options(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper_atlas_options


seed_option = click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=None,
    help="The random seed.",
)


@click.group()
@click.version_option()
@click.option(
    "-c",
    "--config",
    type=click.Path(dir_okay=False, exists=True),
    callback=configure,
    is_eager=True,
    expose_value=False,
    show_default=True,
    help="Read option defaults from the specified CFG file.",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
    help="The logger level.",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    help="Trigger the debug mode.",
)
@click.pass_context
def main(ctx, *args, **kwargs):
    """A tool for axon-synthesis management."""
    debug = kwargs.get("debug", False)
    log_level = kwargs.get("log_level", "info")
    if kwargs.get("debug", False):
        log_level = "debug"
    ctx.ensure_object(GlobalConfig)
    ctx.obj.debug = debug
    setup_logger(log_level)
    LOGGER = logging.getLogger()
    LOGGER.info("Running the following command: %s", " ".join(sys.argv))
    LOGGER.info("From the following folder: %s", os.getcwd())


@main.command(short_help="Fetch the White Matter Recipe file from a given repository")
@click.option(
    "--url",
    type=str,
    required=True,
    help="The URL of the repository that contains the target file",
)
@click.option(
    "--file-path",
    type=str,
    default="white_matter_FULL_RECIPE_v1p20.yaml",
    help="The path of the target file in the target repository",
)
@click.option(
    "--version-reference",
    type=str,
    help=(
        "The version that should be used in the repository (can be a tag, a commit hash or a "
        "branch name)"
    ),
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="The path to the destination file",
)
def fetch_white_matter_recipe(**kwargs):
    """Command to fetch the White Matter Recipe from a git repository."""
    fetch(**kwargs)


@main.command(
    short_help=(
        "Generate all the parameters from the Atlas, the White Matter Recipe and the input "
        "morphologies"
    )
)
@click.option(
    "--morphology-path",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the folder containing the input morphologies",
)
@atlas_options
@optgroup.group(
    "White Matter Recipe parameters",
    help="Parameters used to load and process the White Matter Recipe file",
)
@optgroup.option(
    "--wmr-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the White Matter Recipe file",
)
@optgroup.option(
    "--wmr-subregion-uppercase",
    is_flag=True,
    default=False,
    help="",
)
@optgroup.option(
    "--wmr-subregion-keep-prefix",
    "wmr_subregion_remove_prefix",
    flag_value=True,
    default=True,
    help="",
)
@optgroup.option(
    "--wmr-subregion-remove-prefix",
    "wmr_subregion_remove_prefix",
    flag_value=False,
    help="",
)
@optgroup.option(
    "--wmr-sub-region-separator",
    type=str,
    default="",
    help="",
)
@click.option(
    "--clustering-parameters",
    type=ListParam(),
    required=True,
    help="Parameters used for the clustering algorithm",
)
@click.option(
    "--neuron-density",
    type=click.FloatRange(min=0, min_open=True),
    default=1e-2,
    help=(
        "The density of neurons in the atlas (we suppose here that this density is uniform). "
        "This density should be given in number of neurons by cube atlas-unit (usually "
        "micrometer)."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory",
)
@click.pass_obj
def create_inputs(global_config, **kwargs):
    """The command to create inputs."""
    kwargs["debug"] = global_config.debug
    input_creation.create_inputs(**kwargs)


# [Config]
# ; white_matter_file = /gpfs/bbp.cscs.ch/project/proj83/scratch/home/reimann/rat_wm_recipe_tr_ll_ul_un_n_m_subtract_adjacents.yaml
# white_matter_file = white_matter_FULL_RECIPE_v1p20.yaml
# ; white_matter_file = /gpfs/bbp.cscs.ch/project/proj83/scratch/home/reimann/white_matter_FULL_RECIPE_v1p20.yaml
# atlas_path = /gpfs/bbp.cscs.ch/project/proj82/entities/atlas/ThalNCX/20201019/
# ; atlas_path = /gpfs/bbp.cscs.ch/project/proj83/data/atlas/S1/MEAN/P14-MEAN

# # Fetch and process the white matter recipe
# [FetchWhiteMatterRecipe]
# ; url = git@bbpgitlab.epfl.ch:conn/configs/long-range-connectivity-configs.git
# url = /home/adrien/Work/BBP/codes/long-range-connectivity-configs
# file_path = white_matter_FULL_RECIPE_v1p20.yaml
# # version =  # Set a tag here if you don't want to use the latest version
# subregion_remove_prefix = true

# # DiscoverRawData
# [CreateDatasetForRepair]
# morph_dir = /gpfs/bbp.cscs.ch/project/proj81/InputData/Morphologies/Neurons/Mouse/SSCx/WholeBrain/mouselight_isocortex_ASCII_Files_lite
# output_dataset = dataset_for_repair.csv

# # Extract and cluster terminal points
# [ExtractTerminals]
# output_dataset = input_terminals.csv

# [ClusterTerminals]
# clustering_parameters = [{"method": "sphere_parents", "sphere_radius": 500, "max_path_distance": 1500}, {"method": "sphere_parents", "sphere_radius": 300, "max_path_distance": 1000}, {"method": "sphere_parents", "sphere_radius": 100, "max_path_distance": 300}]
# plot_debug = True


# @main.command(short_help="Synthesize the axons for the given morphologies")
# @click.option(
#     "--morphologies",
#     type=click.Path(exists=True, file_okay=False),
#     required=True,
#     help="Path to the folder containing the input morphologies",
# )
# @click.option(
#     "--grafting_points",
#     type=click.Path(exists=True, dir_okay=False),
#     required=False,
#     help=(
#         "Path to the HDF5 file containing the section IDs where the axons should be grafted in "
#         "the input morphologies (axons are grafted to the soma if not provided)"
#     ),
# )
# @optgroup.group("Graph creation parameters", help="Parameters used to build the graph")
# [CreateGraph]
# intermediate_number = 10
# min_intermediate_distance = 500
# min_random_point_distance = 500
# random_point_bbox_buffer = 500
# plot_debug = False
# use_ancestors = False
# use_depth_penalty = True
# depth_penalty_sigma = 0.25
# depth_penalty_amplitude = 10
# favored_regions = ["fiber tracts"]
# @optgroup.group("Graph creation parameters", help="Parameters used to build the graph")
# [PostProcessSteinerMorphologies]
# plot_debug = True
# [AddTufts]
# plot_debug = True
# use_smooth_trunks = True
# def synthesize(**kwargs):
#     pass


# [CreateSourcePoints]
# nb_points = 50
# seed = 0
# source_regions = ["Isocortex"]
# def create_dummy_morphologies(**kwargs):
#     """Create some dummy morphologies that can be used as axon synthesis inputs."""
#     pass


if __name__ == "__main__":  # pragma: no cover
    """The main entry point."""
    main()
