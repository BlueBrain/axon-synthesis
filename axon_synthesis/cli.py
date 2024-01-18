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
from axon_synthesis.synthesis.main_trunk.create_graph import CreateGraphConfig
from axon_synthesis.utils import setup_logger
from axon_synthesis.validation.mimic import mimic_axons
from axon_synthesis.white_matter_recipe import WmrConfig
from axon_synthesis.white_matter_recipe import fetch


def _format_value(data: dict, name: str) -> str:
    return {f"{name}_{key}": value for key, value in data.items()}


def _flatten_command_subsections(ctx, command_group, command_defaults):
    for command_name in command_group.list_commands(ctx):
        command = command_group.get_command(ctx, command_name)
        if isinstance(command, click.core.Group):
            _flatten_command_subsections(ctx, command, command_defaults[command_name])
        else:
            to_add = {}
            to_remove = []
            for subkey, subvalue in command_defaults[command_name].items():
                if isinstance(subvalue, dict):
                    to_add.update(_format_value(subvalue, subkey))
                    to_remove.append(subkey)
            command_defaults[command_name].update(to_add)
            for i in to_remove:
                del command_defaults[command_name][i]


def _process_command(ctx, command, defaults, global_values):
    for subcommand_name in command.list_commands(ctx):
        subcommand = command.get_command(ctx, subcommand_name)
        if subcommand_name not in defaults:
            defaults[subcommand_name] = {}
        if isinstance(subcommand, click.core.Group):
            _process_command(ctx, subcommand, defaults[subcommand_name], global_values)
            continue
        defaults[subcommand_name].update(global_values)


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
        _process_command(ctx, ctx.command, defaults, global_values)

    # Flatten sub-sections
    _flatten_command_subsections(ctx, ctx.command, defaults)

    ctx.default_map = defaults


class GlobalConfig:
    """Class to store global configuration."""

    def __init__(self, *, debug=False, seed=None):
        """The GlobalConfig constructor."""
        self.debug = debug
        self.seed = seed

    def to_config(self, config):
        """Copy internal attributes in the given dictionary."""
        config["debug"] = self.debug
        config["rng"] = self.seed


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


def atlas_kwargs_to_config(config) -> AtlasConfig:
    """Extract the atlas arguments from given config to create an AtlasConfig object."""
    config["atlas_config"] = AtlasConfig(
        config.pop("atlas_path"),
        config.pop("atlas_region_filename"),
        config.pop("atlas_layer_names", None),
    )


def create_graph_options(func):
    """Decorate a click command to add Atlas-specific options."""

    @optgroup.group(
        "Create Graph parameters",
        help="Parameters used to create the graph on which the Steiner Tree is computed",
    )
    @optgroup.option(
        "--create-graph-intermediate-number",
        type=click.IntRange(min=0),
        help="The number of intermediate points added before Voronoï process",
    )
    @optgroup.option(
        "--create-graph-min-intermediate-distance",
        type=click.FloatRange(min=0),
        help="The min distance between two successive intermediate points",
    )
    @optgroup.option(
        "--create-graph-min-random-point-distance",
        type=click.FloatRange(min=0),
        help="The min distance used to add random points",
    )
    @optgroup.option(
        "--create-graph-random-point-bbox-buffer",
        type=click.FloatRange(min=0),
        help="The distance used to add a buffer around the bbox of the points",
    )
    @optgroup.option(
        "--create-graph-voronoi-steps",
        type=click.IntRange(min=1),
        help="The number of Voronoi steps",
    )
    @optgroup.option(
        "--create-graph-duplicate_precision",
        type=click.FloatRange(min=0),
        help="The precision used to detect duplicated points",
    )
    @optgroup.option(
        "--create-graph-use-orientation-penalty/--no-create-graph-use-orientation-penalty",
        default=None,
        help="If set to True, a penalty is added to edges whose direction is not radial",
    )
    @optgroup.option(
        "--create-graph-orientation-penalty-exponent",
        type=click.FloatRange(min=0),
        help="The exponent used for the orientation penalty",
    )
    @optgroup.option(
        "--create-graph-orientation-penalty-amplitude",
        type=click.FloatRange(min=0, min_open=True),
        help="The amplitude used for the orientation penalty",
    )
    @optgroup.option(
        "--create-graph-use-depth-penalty/--no-create-graph-use-depth-penalty",
        default=None,
        help=(
            "If set to True, a penalty is added to edges whose direction is not parallel to the "
            "iso-depth curves"
        ),
    )
    @optgroup.option(
        "--create-graph-depth-penalty-sigma",
        type=click.FloatRange(min=0, min_open=True),
        help="The sigma used for depth penalty",
    )
    @optgroup.option(
        "--create-graph-depth-penalty-amplitude",
        type=click.FloatRange(min=0, min_open=True),
        help="The amplitude of the depth penalty",
    )
    @optgroup.option(
        "--create-graph-favored-regions",
        type=ListParam(),
        help="The list of brain regions in which edge weights are divided by the favoring factor",
    )
    @optgroup.option(
        "--create-graph-favoring-sigma",
        type=click.FloatRange(min=0, min_open=True),
        help="The sigma used to favor the given regions",
    )
    @optgroup.option(
        "--create-graph-favoring-amplitude",
        type=click.FloatRange(min=0, min_open=True),
        help="The amplitude used to favor the given regions",
    )
    @optgroup.option(
        "--create-graph-use-terminal-penalty/--create-graph-no-use-terminal-penalty",
        default=None,
        help="If enabled, a penalty is added to edges that are connected to a",
    )
    @functools.wraps(func)
    def wrapper_atlas_options(*args, **kwargs) -> Callable:
        return func(*args, **kwargs)

    return wrapper_atlas_options


def create_graph_kwargs_to_config(config) -> CreateGraphConfig:
    """Extract the atlas arguments from given config to create an AtlasConfig object."""
    kwargs = {
        "intermediate_number": config.pop("create_graph_intermediate_number", None),
        "min_intermediate_distance": config.pop("create_graph_min_intermediate_distance", None),
        "min_random_point_distance": config.pop("create_graph_min_random_point_distance", None),
        "random_point_bbox_buffer": config.pop("create_graph_random_point_bbox_buffer", None),
        "voronoi_steps": config.pop("create_graph_voronoi_steps", None),
        "duplicate_precision": config.pop("create_graph_duplicate_precision", None),
        "orientation_penalty_exponent": config.pop(
            "create_graph_orientation_penalty_exponent", None
        ),
        "orientation_penalty_amplitude": config.pop(
            "create_graph_orientation_penalty_amplitude", None
        ),
        "depth_penalty_sigma": config.pop("create_graph_depth_penalty_sigma", None),
        "depth_penalty_amplitude": config.pop("create_graph_depth_penalty_amplitude", None),
        "favored_regions": config.pop("create_graph_favored_regions", None),
        "favoring_sigma": config.pop("create_graph_favoring_sigma", None),
        "favoring_amplitude": config.pop("create_graph_favoring_amplitude", None),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if config.pop("create_graph_use_depth_penalty", False):
        kwargs["use_depth_penalty"] = True
    if config.pop("create_graph_use_orientation_penalty", False):
        kwargs["use_orientation_penalty"] = True
    if config.pop("create_graph_use_terminal_penalty", False):
        kwargs["use_terminal_penalty"] = True

    config["create_graph_config"] = CreateGraphConfig(**kwargs)


def wmr_kwargs_to_config(config) -> WmrConfig:
    """Extract the atlas arguments from given config to create an AtlasConfig object."""
    config["wmr_config"] = WmrConfig(
        config.pop("wmr_path"),
        config.pop("wmr_subregion_uppercase"),
        config.pop("wmr_subregion_remove_prefix"),
        config.pop("wmr_sub_region_separator"),
    )


seed_option = click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=None,
    help="The random seed.",
)

clustering_parameters_option = click.option(
    "--clustering-parameters",
    type=DictParam(),
    required=True,
    help="Parameters used for the clustering algorithm",
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
    "--morphology-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="The directory containing the input morphologies",
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
@clustering_parameters_option
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
    global_config.to_config(kwargs)
    atlas_kwargs_to_config(kwargs)
    wmr_kwargs_to_config(kwargs)
    input_creation.create_inputs(**kwargs)


@main.command(short_help="Synthesize the axons for the given morphologies")
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
@click.option(
    "--morphology-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="The directory containing the input morphologies",
)
@click.option(
    "--morphology-ext",
    type=str,
    required=True,
    help="The extension used to filter the input morphologies",
)
@click.option(
    "--morphology-data-file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="The MVD3 or SONATA file containing morphology data.",
)
@atlas_options
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
@create_graph_options
@click.pass_obj
def synthesize(global_config: GlobalConfig, **kwargs):
    """The command to synthesize axons."""
    global_config.to_config(kwargs)
    atlas_kwargs_to_config(kwargs)
    create_graph_kwargs_to_config(kwargs)
    synthesize_axons(**kwargs)


@main.group()
def validation():
    """Subset of commands used to validate axon synthesis."""


@validation.command(
    short_help="Synthesize axons mimicking the input ones.",
)
@click.option(
    "--morphology-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="The directory containing the input morphologies",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="The directory where the outputs will be stored.",
)
@clustering_parameters_option
@create_graph_options
@click.pass_obj
def mimic(global_config: GlobalConfig, *args, **kwargs):
    """The command to synthesize mimicking axons."""
    global_config.to_config(kwargs)
    create_graph_kwargs_to_config(kwargs)
    mimic_axons(**kwargs)


if __name__ == "__main__":  # pragma: no cover
    """The main entry point."""
    main()
