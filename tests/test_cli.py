"""Tests for the axon_synthesis.cli module."""
import axon_synthesis.cli


def test_cli(cli_runner):
    # pylint: disable=unused-argument
    """Test the CLI."""
    result = cli_runner.invoke(axon_synthesis.cli.main, ["--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage: ")

    for command in axon_synthesis.cli.main.list_commands(None):
        result = cli_runner.invoke(axon_synthesis.cli.main, [command, "--help"])
        assert result.exit_code == 0
        assert f"Usage: axon-synthesis {command}" in result.output

    for command in axon_synthesis.cli.validation_group.list_commands(None):
        result = cli_runner.invoke(axon_synthesis.cli.main, ["validation", command, "--help"])
        assert result.exit_code == 0
        assert f"Usage: axon-synthesis validation {command}" in result.output


def test_entry_point(script_runner):
    """Test the entry point."""
    ret = script_runner.run("axon-synthesis", "--version")
    assert ret.success
    assert ret.stdout.startswith("axon-synthesis, version ")
    assert ret.stderr == ""
