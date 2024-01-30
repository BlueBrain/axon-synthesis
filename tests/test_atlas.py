"""Tests for the atlas processing functions."""
import pytest


@pytest.mark.xfail()
def test_atlas(atlas):
    """Test the atlas."""
    raise RuntimeError(atlas)
