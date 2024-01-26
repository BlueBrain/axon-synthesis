"""Some utils for the CLI of axon-synthesis."""
import json

import click
from configobj import ConfigObj


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
