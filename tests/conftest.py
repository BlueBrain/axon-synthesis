"""Configuration for the pytest test suite."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=redefined-outer-name
import logging
import shutil
from pathlib import Path

import dir_content_diff.comparators.pandas
import pytest
from voxcell.nexus.voxelbrain import Atlas

from . import DATA
from . import EXAMPLES
from . import TEST_ROOT
from .data_factories import generate_small_O1

logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True

dir_content_diff.comparators.pandas.register()


def pytest_addoption(parser):
    """Hook to add custom options to the CLI of pytest."""
    parser.addoption(
        "--interactive-plots",
        action="store_true",
        default=False,
        help="Trigger interactive plots in tests to check the results",
    )


@pytest.fixture
def interactive_plots(request):
    """The value given to the option for interactive plots."""
    return request.config.getoption("--interactive-plots")


def pytest_configure(config):
    """Add --check-untyped-defs option to the mypy plugin."""
    plugin = config.pluginmanager.getplugin("mypy")
    plugin.mypy_argv.append("--check-untyped-defs")


@pytest.fixture
def root_dir():
    """The root directory."""
    return Path(TEST_ROOT)


@pytest.fixture
def data_dir():
    """The data directory."""
    return Path(DATA)


@pytest.fixture
def example_dir():
    """The example directory."""
    return Path(EXAMPLES)


@pytest.fixture
def testing_dir(tmpdir, monkeypatch):
    """The testing directory."""
    monkeypatch.chdir(tmpdir)
    return Path(tmpdir)


@pytest.fixture
def out_dir(testing_dir):
    """The output directory."""
    path = testing_dir / "out"
    path.mkdir(parents=True)

    return path


@pytest.fixture(scope="session")
def atlas_path(tmpdir_factory):
    """Generate a small O1 atlas for the test session."""
    atlas_directory = tmpdir_factory.mktemp("atlas_small_O1")
    return generate_small_O1(atlas_directory)


@pytest.fixture
def atlas(atlas_path):
    """Load the small O1 atlas."""
    return Atlas.open(str(atlas_path))


@pytest.fixture
def brain_regions(atlas):
    """Load the brain regions of the small O1 atlas."""
    return atlas.load_data("brain_regions")


@pytest.fixture(scope="session")
def morphology_path(tmpdir_factory):
    """Generate a small O1 atlas for the test session."""
    morph_directory = Path(tmpdir_factory.mktemp("morphologies"))
    for i in (DATA / "input_morphologies").iterdir():
        shutil.copyfile(i, morph_directory / i.name)
    return morph_directory


@pytest.fixture
def wmr_path(out_dir):
    """Generate a white matter recipe file.

    This WMR is compatible with the small O1 atlas.
    """
    wmr_filepath = out_dir / "white_matter_recipe" / "white_matter_recipe_test.yaml"
    wmr_filepath.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "white_matter_recipe.yaml", wmr_filepath)
    return wmr_filepath


@pytest.fixture
def _tuft_inputs(testing_dir) -> None:
    """Copy inputs for tuft generation in the testing directory."""
    shutil.copyfile(DATA / "tuft_distributions.json", testing_dir / "tuft_distributions.json")
    shutil.copyfile(DATA / "tuft_parameters.json", testing_dir / "tuft_parameters.json")
