"""Common entries of the Command Line Interface."""
import functools
from collections.abc import Callable

import click
from click_option_group import optgroup

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.cli.utils import ListParam


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
