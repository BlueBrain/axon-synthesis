"""Tests suite for the axon-synthesis package."""
from pathlib import Path

TEST_ROOT = Path(__file__).parent
DATA = TEST_ROOT / "data"


class SharedData:
    """A class used to pass data from a test to another."""

    create_inputs_path = None
