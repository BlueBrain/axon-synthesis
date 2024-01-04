"""Command Line Interface for the axon_synthesis package."""
import functools
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import click
from click_option_group import optgroup
from configobj import ConfigObj

from axon_synthesis import input_creation
from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.synthesis import synthesize_axons
from axon_synthesis.utils import setup_logger
from axon_synthesis.white_matter_recipe import WmrConfig
from axon_synthesis.white_matter_recipe import fetch


def configure(ctx: click.Context, _, filename: None | str):
    """Set parameter default values according to a given configuration file."""
    if filename is None:
        return

    # Load the config file
    cfg = ConfigObj(filename)

    # Get current default values
    defaults = cfg.dict()

    # Copy global arguments to all sub commands
    global_values = defaults.get("global", None)
    if global_values:
        for subcommand in ctx.command.list_commands(ctx):
            if subcommand not in defaults:
                defaults[subcommand] = {}
            defaults[subcommand].update(global_values)

    def format_value(data: dict, name: str) -> str:
        return {f"{name}_{key}": value for key, value in data.items()}

    # Flatten sub-sections
    for value in defaults.values():
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

    def __init__(self, *, debug=False, seed=None):
        """The GlobalConfig constructor."""
        self.debug = debug
        self.seed = seed


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


class DictParam(click.ParamType):
    """A `click` parameter to process parameters given as JSON objects."""

    name = "dict"

    def convert(self, value, param, ctx):
        """Convert a given value."""
        try:
            if not isinstance(value, dict):
                value = json.loads(value)
        except json.JSONDecodeError:
            self.fail(f"{value!r} is not a valid JSON object", param, ctx)

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
        "--atlas-layer-names",
        type=ListParam(),
        help=(
            "Names of the layers given as a JSON list (the atlas folder must contain a file name "
            "'[PH]<layer_name>.nrrd' for each given layer)"
        ),
    )
    @functools.wraps(func)
    def wrapper_atlas_options(*args, **kwargs) -> Callable:
        return func(*args, **kwargs)

    return wrapper_atlas_options


def atlas_kwargs_to_config(kwargs) -> AtlasConfig:
    """Extract the atlas arguments from given kwargs to create an AtlasConfig object."""
    return AtlasConfig(
        kwargs.pop("atlas_path"),
        kwargs.pop("atlas_region_filename"),
        kwargs.pop("atlas_layer_names", None),
    )


def wmr_kwargs_to_config(kwargs) -> WmrConfig:
    """Extract the atlas arguments from given kwargs to create an AtlasConfig object."""
    return WmrConfig(
        kwargs.pop("wmr_path"),
        kwargs.pop("wmr_subregion_uppercase"),
        kwargs.pop("wmr_subregion_remove_prefix"),
        kwargs.pop("wmr_sub_region_separator"),
    )


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
    "-d/-nd",
    "--debug/--no-debug",
    default=False,
    help="Trigger the debug mode.",
)
@seed_option
@click.pass_context
def main(ctx, *args, **kwargs):
    """A tool for axon-synthesis management."""
    debug = kwargs.get("debug", False)
    seed = kwargs.get("seed", None)
    log_level = kwargs.get("log_level", "info")
    if kwargs.get("debug", False):
        log_level = "debug"

    ctx.ensure_object(GlobalConfig)
    ctx.obj.debug = debug
    ctx.obj.seed = seed
    setup_logger(log_level)

    logger = logging.getLogger()
    logger.info("Running the following command: %s", " ".join(sys.argv))
    logger.info("From the following folder: %s", Path.cwd())


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
    ),
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
    type=DictParam(),
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
    "--bouton-density",
    type=click.FloatRange(min=0, min_open=True),
    default=0.2,
    help="The density of boutons along the axons (we suppose here that this density is uniform).",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory",
)
@click.pass_obj
def create_inputs(global_config: GlobalConfig, **kwargs):
    """The command to create inputs."""
    kwargs["debug"] = global_config.debug
    kwargs["rng"] = global_config.seed
    kwargs["atlas_config"] = atlas_kwargs_to_config(kwargs)
    kwargs["wmr_config"] = wmr_kwargs_to_config(kwargs)
    input_creation.create_inputs(**kwargs)


@main.command(short_help="Synthesize the axons for the given morphologies")
@click.option(
    "--morphology-path",
    type=click.Path(exists=True, file_okay=False),
    # required=True,
    help="Path to the folder containing the input morphologies",
)
@click.option(
    "--morphology-ext",
    type=str,
    # required=True,
    help="The extension used to filter the input morphologies",
)
@click.option(
    "--morphology-data-file",
    type=click.Path(exists=True, dir_okay=False),
    # required=True,
    help="The MVD3 file containing morphology data.",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="The directory containing the inputs.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="The directory where the outputs will be stored.",
)
@atlas_options
# @click.option(
#     "--morphologies",
#     type=click.Path(exists=True, file_okay=False),
#     required=True,
#     help="Path to the folder containing the input morphologies",
# )
@click.option(
    "-r/-nr",
    "--rebuild-existing-axons/--no-rebuild-existing-axons",
    default=False,
    help="Force rebuilding existing axons.",
)
@click.option(
    "--axon-grafting-points-file",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help=(
        "Path to the HDF5 file containing the section IDs where the axons should be grafted in "
        "the input morphologies (axons are grafted to the soma if not provided)."
    ),
)
# @optgroup.group("Graph creation parameters", help="Parameters used to build the graph")
@click.pass_obj
def synthesize(global_config: GlobalConfig, **kwargs):
    """The command to synthesize axons."""
    kwargs["debug"] = global_config.debug
    kwargs["rng"] = global_config.seed
    kwargs["atlas_config"] = atlas_kwargs_to_config(kwargs)
    synthesize_axons(**kwargs)


if __name__ == "__main__":  # pragma: no cover
    """The main entry point."""
    main()
