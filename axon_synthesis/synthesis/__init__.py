"""Base of the synthesis modules."""
from pathlib import Path

import numpy as np
import pandas as pd
from voxcell.cell_collection import CellCollection

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.input_creation.inputs import Inputs
from axon_synthesis.synthesis.source_points import set_source_points
from axon_synthesis.synthesis.target_points import get_target_points
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType

_HDF_DEFAULT_GROUP = "axon_grafting_points"


def load_axon_grafting_points(path: FileType = None, key: str = _HDF_DEFAULT_GROUP):
    """Load the axon mapping from the given file."""
    cols = ["morphology", "grafting_section_id"]
    if path is not None:
        path = Path(path)
        if path.exists():
            mapping = pd.read_hdf(path, key)
            if set(cols).difference(mapping.columns):
                msg = f"The DataFrame loaded from '{path}' must contain the {cols} columns."
                raise ValueError(msg)
            return mapping
    return pd.DataFrame([], columns=cols)


def synthesize_axons(
    input_dir: FileType,
    output_dir: FileType,
    morphology_data_file: FileType,
    morphology_path: FileType,
    morphology_ext: str,
    axon_grafting_points_file: FileType = None,
    *,
    atlas_config: AtlasConfig = None,
    rebuild_existing_axons: bool = False,
    seed: SeedType = None,
    debug: bool = False,
):
    """Synthesize the long-range axons.

    Args:
        input_dir: The directory containing the inputs.
        output_dir: The directory containing the outputs.
        morphology_data_file: The path to the MVD3/sonata file.
        morphology_path: The directory containing the input morphologies.
        morphology_ext: The extension of the input morphologies.
        axon_grafting_points_file: The file containing the grafting points.
        atlas_config: The config used to load the Atlas.
        rebuild_existing_axons: Rebuild the axons if they already exist.
        seed: The random seed.
        debug: Trigger the Debug mode.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all inputs
    inputs = Inputs.load(input_dir, atlas_config=atlas_config)

    # Load the cell collection
    cells_df = CellCollection.load(morphology_data_file).as_dataframe()

    # Load the axon grafting points
    axon_grafting_points = load_axon_grafting_points(axon_grafting_points_file)

    # Get source points for all axons
    source_points = set_source_points(
        cells_df,
        inputs.atlas,
        morphology_path,
        morphology_ext,
        inputs.population_probabilities,
        axon_grafting_points,
        rebuild_existing_axons=rebuild_existing_axons,
    )

    # Find targets for all axons
    target_points = get_target_points(
        inputs.atlas,
        inputs.brain_regions_mask_file,
        source_points,
        inputs.projection_probabilities,
        rng=rng,
        output_path=output_dir / "target_points.csv" if debug else None,
    )

    # Create the graph for each axon

    # Build the Steiner Tree for each axon

    # Create the tufts for each axon
