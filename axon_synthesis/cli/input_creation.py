"""Entries of the Command Line Interface dedicated to the input creation."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
# LICENSE HEADER MANAGED BY add-license-header

from pathlib import Path

import click
from click_option_group import optgroup

from axon_synthesis import inputs
from axon_synthesis.cli.common import atlas_kwargs_to_config
from axon_synthesis.cli.common import atlas_options
from axon_synthesis.cli.common import parallel_kwargs_to_config
from axon_synthesis.cli.common import parallel_options
from axon_synthesis.cli.utils import DictParam
from axon_synthesis.cli.utils import GlobalConfig
from axon_synthesis.white_matter_recipe import WmrConfig
from axon_synthesis.white_matter_recipe import fetch


def wmr_kwargs_to_config(config) -> None:
    """Extract the atlas arguments from given config to create an AtlasConfig object."""
    wmr_path = config.pop("wmr_path", None)
    wmr_subregion_uppercase = config.pop("wmr_subregion_uppercase")
    wmr_subregion_remove_prefix = config.pop("wmr_subregion_remove_prefix")
    wmr_sub_region_separator = config.pop("wmr_sub_region_separator")
    if wmr_path is None:
        config["wmr_config"] = None
    else:
        config["wmr_config"] = WmrConfig(
            wmr_path,
            wmr_subregion_uppercase,
            wmr_subregion_remove_prefix,
            wmr_sub_region_separator,
        )


clustering_parameters_option = click.option(
    "--clustering-parameters",
    type=DictParam(),
    required=True,
    help="Parameters used for the clustering algorithm",
)


@click.command(short_help="Fetch the White Matter Recipe file from a given repository")
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


@click.command(
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
@atlas_options(required=False)
@optgroup.group(
    "White Matter Recipe parameters",
    help="Parameters used to load and process the White Matter Recipe file",
)
@optgroup.option(
    "--wmr-path",
    type=click.Path(exists=True, dir_okay=False),
    # required=False,
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
@parallel_options
@click.pass_obj
def create_inputs(global_config: GlobalConfig, **kwargs):
    """The command to create inputs."""
    global_config.to_config(kwargs)
    atlas_kwargs_to_config(kwargs)
    wmr_kwargs_to_config(kwargs)
    parallel_kwargs_to_config(kwargs)
    inputs.create.create_inputs(**kwargs)
