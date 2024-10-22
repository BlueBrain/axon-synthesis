"""Entries of the Command Line Interface dedicated to the scripts."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
# LICENSE HEADER MANAGED BY add-license-header

import logging

import click

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.cli.synthesis import atlas_kwargs_to_config
from axon_synthesis.cli.synthesis import atlas_options
from axon_synthesis.cli.utils import GlobalConfig
from axon_synthesis.cli.utils import ListParam
from axon_synthesis.utils import create_random_morphologies


@click.command(
    short_help="Create empty cells at random locations in the atlas.",
)
@click.option(
    "--nb-morphologies",
    type=click.IntRange(min=1),
    required=True,
    help="The number of empty morphologies to generate",
)
@click.option(
    "--brain-regions",
    type=ListParam(),
    help="The list of brain regions in which the empty morphologies should be located",
)
@click.option(
    "--output-morphology-dir",
    type=click.Path(file_okay=False),
    help="The directory where the morphologies will be exported",
)
@click.option(
    "--output-cell-collection",
    type=click.Path(dir_okay=False),
    help="The path to the output cell collection",
)
@click.option(
    "--morphology-prefix",
    type=str,
    help="The prefix of morphology names",
)
@atlas_options(required=True)
@click.pass_obj
def random_morphologies(global_config: GlobalConfig, *_args, **kwargs):
    """The command to create random empty morphologies."""
    global_config.to_config(kwargs)
    kwargs.pop("debug", None)
    atlas_kwargs_to_config(kwargs)
    kwargs["atlas_config"].load_region_map = True
    kwargs["atlas"] = AtlasHelper(kwargs.pop("atlas_config"))
    kwargs["logger"] = logging.getLogger(__name__)
    create_random_morphologies(**kwargs)
