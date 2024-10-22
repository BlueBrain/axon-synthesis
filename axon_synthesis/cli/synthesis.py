"""Entries of the Command Line Interface dedicated to synthesis."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
# LICENSE HEADER MANAGED BY add-license-header

import functools
from collections.abc import Callable

import click
from attrs import fields_dict
from click_option_group import optgroup

from axon_synthesis.cli.common import atlas_kwargs_to_config
from axon_synthesis.cli.common import atlas_options
from axon_synthesis.cli.common import parallel_kwargs_to_config
from axon_synthesis.cli.common import parallel_options
from axon_synthesis.cli.utils import GlobalConfig
from axon_synthesis.cli.utils import ListParam
from axon_synthesis.synthesis import SynthesisConfig
from axon_synthesis.synthesis import synthesize_axons
from axon_synthesis.synthesis.main_trunk.create_graph import CreateGraphConfig
from axon_synthesis.synthesis.main_trunk.post_process import PostProcessConfig
from axon_synthesis.synthesis.outputs import OutputConfig


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
        "--create-graph-use-orientation-penalty/--create-graph-no-use-orientation-penalty",
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
        "--create-graph-use-depth-penalty/--create-graph-no-use-depth-penalty",
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
        "--create-graph-preferred-regions",
        type=ListParam(),
        help="The list of brain regions in which edge weights are divided by the preferring factor",
    )
    @optgroup.option(
        "--create-graph-preferred-region-min-random-point-distance",
        type=click.FloatRange(min=0, min_open=True),
        help="The min distance used to pick random points in preferred regions",
    )
    @optgroup.option(
        "--create-graph-preferring-sigma",
        type=click.FloatRange(min=0, min_open=True),
        help="The sigma used to compute the preferring factor for the given regions",
    )
    @optgroup.option(
        "--create-graph-preferring-amplitude",
        type=click.FloatRange(min=0, min_open=True),
        help="The amplitude used to compute the preferring factor for the given regions",
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


def create_graph_kwargs_to_config(config) -> None:
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
        "preferred_regions": config.pop("create_graph_preferred_regions", None),
        "preferred_region_min_random_point_distance": config.pop(
            "create_graph_preferred_region_min_random_point_distance", None
        ),
        "preferring_sigma": config.pop("create_graph_preferring_sigma", None),
        "preferring_amplitude": config.pop("create_graph_preferring_amplitude", None),
        "use_depth_penalty": config.pop("create_graph_use_depth_penalty", None),
        "use_orientation_penalty": config.pop("create_graph_use_orientation_penalty", None),
        "use_terminal_penalty": config.pop("create_graph_use_terminal_penalty", None),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    config["create_graph_config"] = CreateGraphConfig(**kwargs)


def post_process_options(func):
    """Decorate a click command to add Atlas-specific options."""

    @optgroup.group(
        "Long-range trunk Post-Processing parameters",
        help="Parameters used to post-process the long-range trunk of the morphology",
    )
    @optgroup.option(
        "--post-processing-disable/--post-processing-enable",
        default=None,
        help="If disabled, the long-range trunk is not post-processed with random walk",
    )
    @optgroup.option(
        "--post-processing-history-path-length",
        type=click.FloatRange(min=0),
        help="The length used to compute the random walk history",
    )
    @optgroup.option(
        "--post-processing-default-history-path-length",
        type=click.FloatRange(min=0),
        help="The coefficient used to compute the history path length when it is not provided",
    )
    @optgroup.option(
        "--post-processing-global-target-coeff",
        type=click.FloatRange(min=0),
        help="The coefficient applied to the global target term",
    )
    @optgroup.option(
        "--post-processing-target-coeff",
        type=click.FloatRange(min=0),
        help="The coefficient applied to the next target term",
    )
    @optgroup.option(
        "--post-processing-random-coeff",
        type=click.FloatRange(min=0),
        help="The coefficient applied to the random term",
    )
    @optgroup.option(
        "--post-processing-history-coeff",
        type=click.FloatRange(min=0),
        help="The coefficient applied to the history term",
    )
    @optgroup.option(
        "--post-processing-length-coeff",
        type=click.FloatRange(min=0),
        help="The coefficient applied to step length",
    )
    @optgroup.option(
        "--post-processing-max-random-direction-picks",
        type=click.IntRange(min=1),
        help="The maximum number of random direction picks",
    )
    @functools.wraps(func)
    def wrapper_post_processing_options(*args, **kwargs) -> Callable:
        return func(*args, **kwargs)

    return wrapper_post_processing_options


def post_process_kwargs_to_config(config) -> None:
    """Extract the post-process arguments from given config to create a PostProcessConfig object."""
    kwargs = {
        "history_path_length": config.pop("post_processing_history_path_length", None),
        "default_history_path_length": config.pop(
            "post_processing_default_history_path_length", None
        ),
        "global_target_coeff": config.pop("post_processing_global_target_coeff", None),
        "target_coeff": config.pop("post_processing_target_coeff", None),
        "random_coeff": config.pop("post_processing_random_coeff", None),
        "history_coeff": config.pop("post_processing_history_coeff", None),
        "length_coeff": config.pop("post_processing_length_coeff", None),
        "max_random_direction_picks": config.pop(
            "post_processing_max_random_direction_picks", None
        ),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if config.pop("post_processing_disable", False):
        kwargs["skip"] = True

    config["post_process_config"] = PostProcessConfig(**kwargs)


def output_options(func):
    """Decorate a click command to add output-specific options."""

    @optgroup.group(
        "Output parameters",
        help="Parameters used to configure the outputs",
    )
    @optgroup.option(
        "--output-dir",
        type=click.Path(file_okay=False),
        required=True,
        help="The directory where the outputs will be stored.",
    )
    @optgroup.option(
        "--outputs-enable-final-figures/--outputs-disable-final-figures",
        default=None,
        help="If enabled, the final figures are exported",
    )
    @optgroup.option(
        "--outputs-enable-graph-creation-figures/--outputs-disable-graph-creation-figures",
        default=None,
        help="If enabled, the graph creation figures are exported",
    )
    @optgroup.option(
        "--outputs-enable-graph-creation-data/--outputs-disable-graph-creation-data",
        default=None,
        help="If enabled, the graph creation data are exported",
    )
    @optgroup.option(
        "--outputs-enable-main-trunk-figures/--outputs-disable-main-trunk-figures",
        default=None,
        help="If enabled, the main trunk figures are exported",
    )
    @optgroup.option(
        "--outputs-enable-main-trunk-morphologies/--outputs-disable-main-trunk-morphologies",
        default=None,
        help="If enabled, the main trunk morphologies are exported",
    )
    @optgroup.option(
        "--outputs-enable-morphologies/--outputs-disable-morphologies",
        default=None,
        help="If enabled, the morphologies with new axons are exported",
    )
    @optgroup.option(
        "--outputs-enable-morphologies-edges/--outputs-disable-morphologies-edges",
        default=None,
        help="If enabled, the morphologies with new axons are exported as edges",
    )
    @optgroup.option(
        "--outputs-enable-postprocess-trunk-figures/--outputs-disable-postprocess-trunk-figures",
        default=None,
        help="If enabled, the post-process trunk figures are exported",
    )
    @optgroup.option(
        "--outputs-enable-postprocess-trunk-morphologies/--outputs-disable-postprocess-trunk-morphologies",
        default=None,
        help="If enabled, the post-process trunk morphologies are exported",
    )
    @optgroup.option(
        "--outputs-enable-steiner-tree-solutions/--outputs-disable-steiner-tree-solutions",
        default=None,
        help="If enabled, the Steiner tree solutions are exported",
    )
    @optgroup.option(
        "--outputs-enable-steiner-tree-solution-figures/--outputs-disable-steiner-tree-solution-figures",
        default=None,
        help="If enabled, the Steiner tree solution figures are exported",
    )
    @optgroup.option(
        "--outputs-enable-target-point-figures/--outputs-disable-target-point-figures",
        default=None,
        help="If enabled, the target point figures are exported",
    )
    @optgroup.option(
        "--outputs-enable-target-points/--outputs-disable-target-points",
        default=None,
        help="If enabled, the target point data are exported",
    )
    @optgroup.option(
        "--outputs-enable-tuft-figures/--outputs-disable-tuft-figures",
        default=None,
        help="If enabled, the tuft figures are exported",
    )
    @optgroup.option(
        "--outputs-enable-tuft-morphologies/--outputs-disable-tuft-morphologies",
        default=None,
        help="If enabled, the tuft morphologies are exported",
    )
    @functools.wraps(func)
    def wrapper_output_options(*args, **kwargs) -> Callable:
        return func(*args, **kwargs)

    return wrapper_output_options


def outputs_kwargs_to_config(config) -> None:
    """Extract the atlas arguments from given config to create an AtlasConfig object."""
    kwargs = {"path": config.pop("output_dir")}

    for k in fields_dict(OutputConfig):  # pylint: disable=not-an-iterable
        name = "outputs_enable_" + k
        if config.pop(name, False):
            kwargs[k] = True

    config["output_config"] = OutputConfig(**kwargs)


def synthesis_options(func):
    """Decorate a click command to add synthesis-specific options."""

    @optgroup.group(
        "Synthesis parameters",
        help="Parameters used to configure the synthesis process",
    )
    @optgroup.option(
        "--input-dir",
        type=click.Path(exists=True, file_okay=False),
        required=True,
        help="The directory containing the inputs.",
    )
    @optgroup.option(
        "--morphology-dir",
        type=click.Path(exists=True, file_okay=False),
        required=True,
        help="The directory containing the input morphologies",
    )
    @optgroup.option(
        "--morphology-data-file",
        type=click.Path(exists=True, dir_okay=False),
        required=True,
        help="The MVD3 or SONATA file containing morphology data.",
    )
    @optgroup.option(
        "-r/-nr",
        "--rebuild-existing-axons/--no-rebuild-existing-axons",
        default=False,
        help="Force rebuilding existing axons.",
    )
    @optgroup.option(
        "--axon-grafting-points-file",
        type=click.Path(exists=True, dir_okay=False),
        required=False,
        help=(
            "Path to the HDF5 file containing the section IDs where the axons should be grafted in "
            "the input morphologies (axons are grafted to the soma if not provided)."
        ),
    )
    @optgroup.option(
        "--population-probabilities-file",
        type=click.Path(exists=True, dir_okay=False),
        required=False,
        help="Path to the file containing the population probabilities.",
    )
    @optgroup.option(
        "--projection-probabilities-file",
        type=click.Path(exists=True, dir_okay=False),
        required=False,
        help="Path to the file containing the projection probabilities.",
    )
    @optgroup.option(
        "--population-tuft-number-file",
        type=click.Path(exists=True, dir_okay=False),
        required=False,
        help="Path to the file containing the tuft number distribution for each target population.",
    )
    @optgroup.option(
        "--trunk-properties-file",
        type=click.Path(exists=True, dir_okay=False),
        required=False,
        help="Path to the file containing the trunk properties.",
    )
    @optgroup.option(
        "--tuft-properties-file",
        type=click.Path(exists=True, dir_okay=False),
        required=False,
        help="Path to the file containing the tuft barcodes given to NeuroTS.",
    )
    @optgroup.option(
        "--tuft-distributions-file",
        type=click.Path(exists=True, dir_okay=False),
        required=False,
        help="Path to the file containing the tuft distributions given to NeuroTS.",
    )
    @optgroup.option(
        "--tuft-parameters-file",
        type=click.Path(exists=True, dir_okay=False),
        required=False,
        help="Path to the file containing the tuft parameters given to NeuroTS.",
    )
    @optgroup.option(
        "--tuft-boundary-max-distance",
        type=click.FloatRange(min=0),
        required=False,
        help=(
            "Maximum distance used for the calculation of the attenuation component near the "
            "boundary."
        ),
    )
    @optgroup.option(
        "--tuft-boundary-scale-coeff",
        type=click.FloatRange(min=0),
        required=False,
        help="Coefficient used in the calculation of the attenuation component near the boundary.",
    )
    @optgroup.option(
        "--target-max-tries",
        type=click.IntRange(min=1),
        required=False,
        help="The maximum number of tries for picking target points.",
    )
    @functools.wraps(func)
    def wrapper_synthesis_options(*args, **kwargs) -> Callable:
        return func(*args, **kwargs)

    return wrapper_synthesis_options


def synthesis_kwargs_to_config(config) -> None:
    """Extract the synthesis arguments from given config to create an SynthesisConfig object."""
    kwargs = {}
    for k in fields_dict(SynthesisConfig):  # pylint: disable=not-an-iterable
        name = k
        value = config.pop(name, None)
        if value is not None:
            kwargs[k] = value

    config["config"] = SynthesisConfig(**kwargs)


@click.command(short_help="Synthesize the axons for the given morphologies")
@synthesis_options
@output_options
@atlas_options(required=True)
@create_graph_options
@post_process_options
@parallel_options
@click.pass_obj
def synthesize(global_config: GlobalConfig, **kwargs):
    """The command to synthesize axons."""
    global_config.to_config(kwargs)
    kwargs.pop("debug", None)
    outputs_kwargs_to_config(kwargs)
    synthesis_kwargs_to_config(kwargs)
    atlas_kwargs_to_config(kwargs)
    create_graph_kwargs_to_config(kwargs)
    post_process_kwargs_to_config(kwargs)
    parallel_kwargs_to_config(kwargs)
    synthesize_axons(**kwargs)
