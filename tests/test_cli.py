"""Tests for the axon_synthesis.cli module."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

from click import Context

import axon_synthesis.cli


def test_cli(cli_runner):
    """Test the CLI."""
    result = cli_runner.invoke(axon_synthesis.cli.main, ["--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage: ")

    ctx = Context(axon_synthesis.cli.main)

    for command in axon_synthesis.cli.main.list_commands(ctx):
        result = cli_runner.invoke(axon_synthesis.cli.main, [command, "--help"])
        assert result.exit_code == 0
        assert f"Usage: axon-synthesis {command}" in result.output

    for command in axon_synthesis.cli.validation_group.list_commands(ctx):
        result = cli_runner.invoke(axon_synthesis.cli.main, ["validation", command, "--help"])
        assert result.exit_code == 0
        assert f"Usage: axon-synthesis validation {command}" in result.output


def test_entry_point(script_runner):
    """Test the entry point."""
    ret = script_runner.run("axon-synthesis", "--version")
    assert ret.success
    assert ret.stdout.startswith("axon-synthesis, version ")
    assert ret.stderr == ""
