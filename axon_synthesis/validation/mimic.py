"""Validation workflow that mimics inputs morphologies."""
from pathlib import Path

import numpy as np
import pandas as pd
from morph_tool.converter import convert
from morph_tool.utils import is_morphology
from voxcell.cell_collection import CellCollection

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.constants import COMMON_ANCESTOR_COORDS_COLS
from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import DEFAULT_POPULATION
from axon_synthesis.constants import TARGET_COORDS_COLS
from axon_synthesis.inputs.create import create_inputs
from axon_synthesis.synthesis import ParallelConfig
from axon_synthesis.synthesis import synthesize_axons
from axon_synthesis.synthesis.main_trunk.create_graph import CreateGraphConfig
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType


def create_cell_collection(
    morphology_dir, output_path: FileType | None = None, convert_to: FileType | None = None
) -> CellCollection:
    """Create a CellCollection object from a directory containing morphologies."""
    morphology_dir = Path(morphology_dir)
    morph_files = [i for i in morphology_dir.iterdir() if is_morphology(i)]

    if convert_to is not None:
        convert_to = Path(convert_to)
        convert_to.mkdir(parents=True, exist_ok=True)
        converted_files = []
        for file in morph_files:
            converted_file = (convert_to / file.stem).with_suffix(".h5")
            convert(file, converted_file, nrn_order=True)
            converted_files.append(converted_file)
        morph_files = converted_files

    morph_names = [i.stem for i in morph_files]
    morph_files = [str(i) for i in morph_files]
    if not morph_files:
        msg = f"No morphology file found in '{morphology_dir}'"
        raise RuntimeError(msg)
    cells_df = pd.DataFrame({"morphology": morph_names, "morph_file": morph_files})
    cells_df["mtype"] = DEFAULT_POPULATION
    cells_df["region"] = DEFAULT_POPULATION
    cells_df[COORDS_COLS] = 0
    cells_df["orientation"] = [np.eye(3)] * len(cells_df)
    cells_df = cells_df.sort_values("morphology", ignore_index=True)
    cells_df.index += 1
    cells = CellCollection.from_dataframe(cells_df)
    if output_path is not None:
        cells.save(output_path)
    return cells


def create_probabilities(cells_df, tuft_properties):
    """Create the population and projection probabilities."""
    # Create population probabilities with all probs = 1
    population_probabilities = cells_df[["population_id"]].copy()
    population_probabilities["brain_region_id"] = population_probabilities["population_id"]
    population_probabilities["probability"] = 1

    # Create projection probabilities with all probs = 1
    projection_probabilities = (
        cells_df[["morphology", "morph_file", "population_id"]]
        .rename(columns={"population_id": "source_population_id"})
        .merge(
            tuft_properties[["morphology", "axon_id", "tuft_id", *COMMON_ANCESTOR_COORDS_COLS]],
            on="morphology",
            how="left",
        )
    )
    projection_probabilities["source_brain_region_id"] = projection_probabilities[
        "source_population_id"
    ].copy()
    projection_probabilities["target_population_id"] = (
        projection_probabilities["source_brain_region_id"]
        + "_"
        + projection_probabilities["tuft_id"].astype(str)
    )
    projection_probabilities["target_brain_region_id"] = projection_probabilities[
        "target_population_id"
    ]
    projection_probabilities[TARGET_COORDS_COLS] = projection_probabilities[
        COMMON_ANCESTOR_COORDS_COLS
    ]
    projection_probabilities["probability"] = 1

    return population_probabilities, projection_probabilities


def mimic_axons(
    output_dir: FileType,
    morphology_dir: FileType,
    clustering_parameters: dict,
    *,
    atlas_config: AtlasConfig | None = None,
    create_graph_config: CreateGraphConfig | None = None,
    rng: SeedType = None,
    debug: bool = False,
    parallel_config: ParallelConfig | None = None,
):
    """Synthesize mimicking axons."""
    input_dir = Path(output_dir) / "inputs"
    synthesis_output_dir = Path(output_dir) / "synthesis"

    if len(clustering_parameters) != 1:
        msg = "The 'clustering_parameters' JSON object should contain exactly 1 entry."
        raise ValueError(msg)

    # Create raw inputs
    inputs = create_inputs(
        morphology_dir,
        input_dir,
        clustering_parameters,
        rng=rng,
        debug=debug,
    )

    # Modify the inputs for the mimic workflow

    # Convert the morphologies to h5 and create a CellCollection
    morphology_data_file = inputs.path / "circuit.h5"
    converted_morphologies_dir = inputs.path / "converted_morphologies"
    cells = create_cell_collection(morphology_dir, morphology_data_file, converted_morphologies_dir)
    cells_df = cells.as_dataframe()

    # Add population_id column to the cell collection with one unique pop for each cell-axon couple
    cells_df["population_id"] = np.arange(len(cells_df)).astype(str)

    # Build and export the probabilities
    inputs.population_probabilities, inputs.projection_probabilities = create_probabilities(
        cells_df, inputs.clustering_data.tuft_properties
    )
    inputs.export_probabilities()

    # Update tuft properties
    tuft_properties = inputs.clustering_data.tuft_properties.merge(
        inputs.projection_probabilities[
            ["morphology", "axon_id", "tuft_id", "target_population_id"]
        ],
        on=["morphology", "axon_id", "tuft_id"],
        how="left",
    )
    inputs.clustering_data.tuft_properties["population_id"] = tuft_properties[
        "target_population_id"
    ].to_numpy()
    inputs.clustering_data.save()

    # Set the source properties
    cells_df["source_brain_region_id"] = cells_df["population_id"].copy()
    cells_df["source_population_id"] = cells_df["population_id"].copy()
    cells = CellCollection.from_dataframe(cells_df)
    cells.save(morphology_data_file)

    # Synthesize the axons using the modified inputs
    synthesize_axons(
        input_dir,
        synthesis_output_dir,
        morphology_data_file,
        converted_morphologies_dir,
        ".h5",
        atlas_config=atlas_config,
        create_graph_config=create_graph_config,
        rebuild_existing_axons=True,
        rng=rng,
        debug=debug,
        parallel_config=parallel_config,
    )
