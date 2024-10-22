"""Entries of the Command Line Interface dedicated to the validation."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
# LICENSE HEADER MANAGED BY add-license-header

import click

from axon_synthesis.cli.common import parallel_kwargs_to_config
from axon_synthesis.cli.common import parallel_options
from axon_synthesis.cli.input_creation import clustering_parameters_option
from axon_synthesis.cli.synthesis import create_graph_kwargs_to_config
from axon_synthesis.cli.synthesis import create_graph_options
from axon_synthesis.cli.synthesis import output_options
from axon_synthesis.cli.synthesis import outputs_kwargs_to_config
from axon_synthesis.cli.synthesis import post_process_kwargs_to_config
from axon_synthesis.cli.synthesis import post_process_options
from axon_synthesis.cli.utils import GlobalConfig
from axon_synthesis.cli.utils import ListParam
from axon_synthesis.validation.mimic import mimic_axons


@click.command(
    short_help="Synthesize axons mimicking the input ones.",
)
@click.option(
    "--morphology-dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="The directory containing the input morphologies",
)
@click.option(
    "--workflows",
    type=ListParam(
        schema={
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "basic",
                    "preferred_regions",
                ],
            },
        },
    ),
    help="The mimic workflows to synthesize",
)
@click.option(
    "--voxel-dimensions",
    type=ListParam(
        schema={
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 1,
            },
        }
    ),
    help="The voxel dimensions used to build the dummy atlas in the 'preferred_regions' workflow",
)
@click.option(
    "--keep-tmp-atlas/--no-keep-tmp-atlas",
    default=False,
    help=("If set to True, the temporary atlas is not removed at the end of the process"),
)
@click.option(
    "--merge-results/--no-merge-results",
    default=True,
    help=("If set to True, the temporary atlas is not removed at the end of the process"),
)
@click.option(
    "--tuft-parameters-file",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Path to the file containing the tuft parameters given to NeuroTS.",
)
@output_options
@clustering_parameters_option
@create_graph_options
@post_process_options
@parallel_options
@click.pass_obj
def mimic(global_config: GlobalConfig, *_args, **kwargs):
    """The command to synthesize mimicking axons."""
    global_config.to_config(kwargs)
    create_graph_kwargs_to_config(kwargs)
    post_process_kwargs_to_config(kwargs)
    outputs_kwargs_to_config(kwargs)
    parallel_kwargs_to_config(kwargs)
    mimic_axons(**kwargs)
