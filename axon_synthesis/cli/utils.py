"""Some utils for the CLI of axon-synthesis."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
# LICENSE HEADER MANAGED BY add-license-header

import json
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any

import click
from configobj import ConfigObj
from jsonschema import validate


def _format_value(data: dict, name: str) -> dict[str, Any]:
    return {f"{name}_{key}": value for key, value in data.items()}


def _recursive_merge(dict_1: MutableMapping, dict_2: MutableMapping) -> MutableMapping:
    """Merge two dictionaries recursively.

    The right one takes precedense in case of conflicting keys.
    """
    merged_dict = deepcopy(dict_1)
    for k, v in dict_2.items():
        if k in merged_dict and isinstance(v, MutableMapping):
            merged_dict[k] = _recursive_merge(merged_dict[k], v)
        else:
            merged_dict[k] = v
    return merged_dict


def _flatten_command_subsections(ctx, command_group, command_defaults) -> None:
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
            command_defaults[command_name] = _recursive_merge(
                to_add, command_defaults[command_name]
            )
            for i in to_remove:
                del command_defaults[command_name][i]


def _process_command(ctx, command, defaults, global_values) -> None:
    for subcommand_name in command.list_commands(ctx):
        subcommand = command.get_command(ctx, subcommand_name)
        if subcommand_name not in defaults:
            defaults[subcommand_name] = {}
        if isinstance(subcommand, click.core.Group):
            _process_command(ctx, subcommand, defaults[subcommand_name], global_values)
            continue
        defaults[subcommand_name] = _recursive_merge(global_values, defaults[subcommand_name])


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

    def __init__(self, *args, schema=None, **kwargs):
        """The ListParam constructor."""
        self.schema = schema
        super().__init__(*args, **kwargs)

    def convert(self, value, param, ctx):
        """Convert a given value."""
        try:
            if not isinstance(value, list):
                value = json.loads(value)
        except json.JSONDecodeError:
            self.fail(f"{value!r} is not a valid JSON array", param, ctx)
        if self.schema is not None:
            validate(value, schema=self.schema)
        return value


class DictParam(click.ParamType):
    """A `click` parameter to process parameters given as JSON objects."""

    name = "dict"

    def __init__(self, *args, schema=None, **kwargs):
        """The DictParam constructor."""
        self.schema = schema
        super().__init__(*args, **kwargs)

    def convert(self, value, param, ctx):
        """Convert a given value."""
        try:
            if not isinstance(value, dict):
                value = json.loads(value)
        except json.JSONDecodeError:
            self.fail(f"{value!r} is not a valid JSON object", param, ctx)
        if self.schema is not None:
            validate(value, schema=self.schema)

        return value
