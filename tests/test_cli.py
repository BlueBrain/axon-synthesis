"""Tests for the axon_synthesis.cli module."""
import axon_synthesis.cli


def test_cli(cli_runner):
    # pylint: disable=unused-argument
    """Test the CLI."""
    result = cli_runner.invoke(
        axon_synthesis.cli.main,
        [
            "-x",
            1,
            "-y",
            2,
        ],
    )
    assert result.exit_code == 0
    assert result.output == "1 + 2 = 3\n"


def test_entry_point(script_runner):
    """Test the entry point."""
    ret = script_runner.run("axon-synthesis", "--version")
    assert ret.success
    assert ret.stdout.startswith("axon-synthesis, version ")
    assert ret.stderr == ""
