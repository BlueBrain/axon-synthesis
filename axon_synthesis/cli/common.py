"""Common entries of the Command Line Interface."""

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
from click_option_group import optgroup

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.cli.utils import DictParam
from axon_synthesis.cli.utils import ListParam
from axon_synthesis.utils import ParallelConfig


def atlas_options(*, required=False):
    """Decorate a click command to add Atlas-specific options."""

    def inner_atlas_options(func) -> Callable:
        @optgroup.group("Atlas parameters", help="Parameters used to read and prepare the atlas")
        @optgroup.option(
            "--atlas-enable/--atlas-disable",
            default=True,
            help="Enable or disable the Atlas",
        )
        @optgroup.option(
            "--atlas-path",
            type=click.Path(exists=True, file_okay=False),
            required=required,
            help="Path to the Atlas folder",
        )
        @optgroup.option(
            "--atlas-region-filename",
            type=str,
            help="Name of NRRD file containing the brain regions in the Atlas folder",
        )
        @optgroup.option(
            "--atlas-layer-names",
            type=ListParam(),
            help=(
                "Names of the layers given as a JSON list (the atlas folder must contain a file "
                "name '[PH]<layer_name>.nrrd' for each given layer)"
            ),
        )
        @optgroup.option(
            "--atlas-outside-region-id",
            type=int,
            help="ID of the 'outside' brain region",
        )
        @optgroup.option(
            "--atlas-enable-boundary/--atlas-disable-boundary",
            default=None,
            help="Enable or disable the boundary conditions",
        )
        @functools.wraps(func)
        def wrapper_atlas_options(*args, **kwargs) -> Callable:
            return func(*args, **kwargs)

        return wrapper_atlas_options

    return inner_atlas_options


def atlas_kwargs_to_config(config) -> None:
    """Extract the atlas arguments from given config to create an AtlasConfig object."""
    kwargs = {
        k: v
        for k, v in {
            "path": config.pop("atlas_path", None),
            "region_filename": config.pop("atlas_region_filename", None),
            "layer_names": config.pop("atlas_layer_names", None),
            "outside_region_id": config.pop("atlas_outside_region_id", None),
            "use_boundary": config.pop("atlas_enable_boundary", None),
        }.items()
        if v is not None
    }
    if config.pop("atlas_enable", False):
        config["atlas_config"] = AtlasConfig(**kwargs)
    else:
        config["atlas_config"] = None


def parallel_options(func):
    """Decorate a click command to add parallel-specific options."""

    @optgroup.group(
        "Parallel computation parameters",
        help="Parameters used to configure the parallel computation",
    )
    @optgroup.option(
        "--nb-workers",
        type=click.IntRange(min=0),
        help="The number of workers",
    )
    @optgroup.option(
        "--dask-config",
        type=DictParam(),
        help="The dask configuration given as a JSON string.",
    )
    @optgroup.option(
        "--progress-bar/--no-progress-bar",
        default=None,
        help="If set to True, a progress bar is displayed during computation",
    )
    @optgroup.option(
        "--use-mpi/--no-use-mpi",
        default=None,
        help="If set to True, MPI is used for parallel computation",
    )
    @functools.wraps(func)
    def wrapper_parallel_options(*args, **kwargs) -> Callable:
        return func(*args, **kwargs)

    return wrapper_parallel_options


def parallel_kwargs_to_config(config) -> None:
    """Extract the parallel arguments from given config to create an ParallelConfig object."""
    kwargs = {
        "nb_processes": config.pop("nb_workers", None),
        "dask_config": config.pop("dask_config", None),
        "progress_bar": config.pop("progress_bar", None),
        "use_mpi": config.pop("use_mpi", None),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    config["parallel_config"] = ParallelConfig(**kwargs)
