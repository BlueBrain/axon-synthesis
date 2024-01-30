"""Configuration for the pytest test suite."""
# pylint: disable=redefined-outer-name
import logging
import shutil
from pathlib import Path

import dir_content_diff.pandas
import pytest
from voxcell.nexus.voxelbrain import Atlas

from . import DATA
from . import EXAMPLES
from . import TEST_ROOT
from .data_factories import generate_small_O1

logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True

dir_content_diff.pandas.register()


def pytest_addoption(parser):
    """Hook to add custom options to the CLI of pytest."""
    parser.addoption(
        "--interactive-plots",
        action="store_true",
        default=False,
        help="Trigger interactive plots in tests to check the results",
    )


@pytest.fixture()
def interactive_plots(request):
    """The value given to the option for interactive plots."""
    return request.config.getoption("--interactive-plots")


@pytest.fixture()
def root_dir():
    """The root directory."""
    return Path(TEST_ROOT)


@pytest.fixture()
def data_dir():
    """The data directory."""
    return Path(DATA)


@pytest.fixture()
def example_dir():
    """The example directory."""
    return Path(EXAMPLES)


@pytest.fixture()
def testing_dir(tmpdir, monkeypatch):
    """The testing directory."""
    monkeypatch.chdir(tmpdir)
    return Path(tmpdir)


@pytest.fixture()
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


@pytest.fixture()
def atlas(atlas_path):
    """Load the small O1 atlas."""
    return Atlas.open(str(atlas_path))


@pytest.fixture()
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


@pytest.fixture()
def wmr_path(out_dir):
    """Generate a white matter recipe file.

    This WMR is compatible with the small O1 atlas.
    """
    wmr_filepath = out_dir / "white_matter_recipe" / "white_matter_recipe_test.yaml"
    wmr_filepath.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / "white_matter_recipe.yaml", wmr_filepath)
    return wmr_filepath


# def _set_luigi_cfg(testing_dir, atlas_path, morphology_path, wmr_path, luigi_cfg_file):
#     """Create a luigi configuration file with the proper paths."""
#     # Load the config
#     cfg = ConfigParser()
#     cfg.read(luigi_cfg_file)

#     # Set the paths
#     cfg["Config"]["atlas_path"] = str(atlas_path)
#     cfg["CreateDatasetForRepair"]["morph_dir"] = str(morphology_path)

#     # Export the config into the luigi.cfg file
#     luigi_cfg_path = testing_dir / "luigi.cfg"
#     with open(luigi_cfg_path, "w") as configfile:
#         cfg.write(configfile)

#     # Set current config in luigi
#     luigi_config = luigi.configuration.get_config()
#     luigi_config.read(luigi_cfg_path)

#     # Copy the logging config file used by luigi
#     shutil.copyfile(DATA / "logging.conf", testing_dir / "logging.conf")

#     return luigi_config, luigi_cfg_path


# @pytest.fixture()
# def luigi_cfg(testing_dir, atlas_path, morphology_path, wmr_path):
#     """Create a luigi configuration file with the proper paths."""
#     luigi_config, luigi_cfg_path = _set_luigi_cfg(
#         testing_dir,
#         atlas_path,
#         morphology_path,
#         wmr_path,
#         DATA / "luigi.cfg",
#     )

#     yield luigi_cfg_path

#     # Reset luigi config
#     luigi_config.clear()


# @pytest.fixture()
# def luigi_mimic_cfg(testing_dir, atlas_path, morphology_path, wmr_path):
#     """Create a luigi configuration file with the proper paths."""
#     luigi_config, luigi_cfg_path = _set_luigi_cfg(
#         testing_dir,
#         atlas_path,
#         morphology_path,
#         wmr_path,
#         DATA / "luigi_mimic.cfg",
#     )

#     yield luigi_cfg_path

#     # Reset luigi config
#     luigi_config.clear()


@pytest.fixture()
def _tuft_inputs(testing_dir):
    """Copy inputs for tuft generation in the testing directory."""
    shutil.copyfile(DATA / "tuft_distributions.json", testing_dir / "tuft_distributions.json")
    shutil.copyfile(DATA / "tuft_parameters.json", testing_dir / "tuft_parameters.json")
