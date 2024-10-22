"""Main entry point of the Command Line Interface for the axon_synthesis package."""

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
import sys
from pathlib import Path

import click

from axon_synthesis.cli import input_creation
from axon_synthesis.cli import scripts
from axon_synthesis.cli import synthesis
from axon_synthesis.cli import validation
from axon_synthesis.cli.utils import GlobalConfig
from axon_synthesis.cli.utils import configure
from axon_synthesis.utils import setup_logger

seed_option = click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=None,
    help="The random seed.",
)


@click.group(name="axon-synthesis")
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
def main(ctx, *_args, **kwargs):
    """A command line tool for axon-synthesis management."""
    debug = kwargs.get("debug", False)
    seed = kwargs.get("seed")
    log_level = kwargs.get("log_level", "info")
    if kwargs.get("debug", False):
        log_level = "debug"

    ctx.ensure_object(GlobalConfig)
    ctx.obj.debug = debug
    ctx.obj.seed = seed
    setup_logger(log_level)

    logger = logging.getLogger()
    logger.info("Running the following command: %s", " ".join(sys.argv))
    logger.info("Running from the following folder: %s", Path.cwd())


main.add_command(synthesis.synthesize)
main.add_command(input_creation.create_inputs)
main.add_command(input_creation.fetch_white_matter_recipe)


@main.group(name="validation")
def validation_group():
    """Subset of commands used to validate axon synthesis."""


validation_group.add_command(validation.mimic)


@main.group(name="scripts")
def scripts_group():
    """Subset of commands used to run some useful scripts for axon synthesis."""


scripts_group.add_command(scripts.random_morphologies)
