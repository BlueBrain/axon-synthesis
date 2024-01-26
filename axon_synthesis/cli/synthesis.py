"""Entries of the Command Line Interface dedicated to synthesis."""
import functools
from collections.abc import Callable

import click
from click_option_group import optgroup

from axon_synthesis.cli.common import atlas_kwargs_to_config
from axon_synthesis.cli.common import atlas_options
from axon_synthesis.cli.utils import GlobalConfig
from axon_synthesis.cli.utils import ListParam
from axon_synthesis.synthesis import synthesize_axons
from axon_synthesis.synthesis.main_trunk.create_graph import CreateGraphConfig


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
    def wrapper_create_graph_options(*args, **kwargs) -> Callable:
        return func(*args, **kwargs)

    return wrapper_create_graph_options


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


@click.command(short_help="Synthesize the axons for the given morphologies")
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
