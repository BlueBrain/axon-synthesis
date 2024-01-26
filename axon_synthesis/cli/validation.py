"""Entries of the Command Line Interface dedicated to the validation."""
import click

from axon_synthesis.cli.input_creation import clustering_parameters_option
from axon_synthesis.cli.synthesis import create_graph_kwargs_to_config
from axon_synthesis.cli.synthesis import create_graph_options
from axon_synthesis.cli.utils import GlobalConfig
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
