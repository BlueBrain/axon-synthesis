"""Test the CreateInputs workflow."""
import shutil

import luigi
import pytest

from axon_synthesis import workflows

from . import SharedData


def test_create_inputs(out_dir, luigi_cfg):
    """Test the complete CreateInputs workflow."""
    assert luigi.build([workflows.CreateInputs()], local_scheduler=True)

    assert (out_dir / "terminals.csv").exists()
    assert (out_dir / "clustering" / "tuft_properties.json").exists()
    assert (out_dir / "clustering" / "clustered_terminals.csv").exists()
    for i in range(1, 4):
        assert (out_dir / "clustering" / "morphologies" / f"morph_000{i}.asc").exists()
    assert (out_dir / "white_matter_recipe" / "white_matter_fractions.json").exists()
    assert (out_dir / "white_matter_recipe" / "white_matter_interaction_strengths.json").exists()
    assert (out_dir / "white_matter_recipe" / "white_matter_population.csv").exists()
    assert (out_dir / "white_matter_recipe" / "white_matter_projections.csv").exists()
    assert (out_dir / "white_matter_recipe" / "white_matter_projection_targets.csv").exists()
    assert (out_dir / "white_matter_recipe" / "white_matter_recipe_test.yaml").exists()
    assert (out_dir / "white_matter_recipe" / "white_matter_targets.csv").exists()

    SharedData.create_inputs_path = out_dir


@pytest.fixture()
def create_inputs(out_dir):
    """Copy outputs from the CreateInputs workflows."""


@pytest.mark.depends(on=["test_create_inputs"])
def test_synthesis(out_dir, luigi_cfg, tuft_inputs):
    """Test the complete Synthesis workflow."""
    shutil.copytree(SharedData.create_inputs_path, out_dir, dirs_exist_ok=True)
    assert luigi.build([workflows.Synthesis()], local_scheduler=True)

    # assert (out_dir / "terminals.csv").exists()
    # assert (out_dir / "clustering" / "tuft_properties.json").exists()
    # assert (out_dir / "clustering" / "clustered_terminals.csv").exists()
    # for i in range(1, 4):
    #     assert (out_dir / "clustering" / "morphologies" / f"morph_000{i}.asc").exists()
    # assert (out_dir / "white_matter_recipe" / "white_matter_fractions.csv").exists()
    # assert (out_dir / "white_matter_recipe" / "white_matter_interaction_strengths.csv").exists()
    # assert (out_dir / "white_matter_recipe" / "white_matter_population.csv").exists()
    # assert (out_dir / "white_matter_recipe" / "white_matter_projections.csv").exists()
    # assert (out_dir / "white_matter_recipe" / "white_matter_projection_targets.csv").exists()
    # assert (out_dir / "white_matter_recipe" / "white_matter_recipe_test.yaml").exists()
    # assert (out_dir / "white_matter_recipe" / "white_matter_targets.csv").exists()


def test_synthesis_mimic(out_dir, luigi_mimic_cfg, tuft_inputs):
    """Test the complete Synthesis workflow."""
    assert luigi.build([workflows.Synthesis()], local_scheduler=True)
