"""Validation workflow that mimics inputs morphologies."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from morph_tool.converter import convert
from morph_tool.utils import is_morphology
from neurom.core import Morphology
from neurom.geom.transform import Translation
from voxcell.cell_collection import CellCollection

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.constants import AXON_GRAFTING_POINT_HDF_GROUP
from axon_synthesis.constants import COMMON_ANCESTOR_COORDS_COLS
from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import DEFAULT_OUTPUT_PATH
from axon_synthesis.constants import DEFAULT_POPULATION
from axon_synthesis.constants import TARGET_COORDS_COLS
from axon_synthesis.inputs.create import create_inputs
from axon_synthesis.synthesis import ParallelConfig
from axon_synthesis.synthesis import SynthesisConfig
from axon_synthesis.synthesis import synthesize_axons
from axon_synthesis.synthesis.main_trunk.create_graph import CreateGraphConfig
from axon_synthesis.synthesis.main_trunk.post_process import PostProcessConfig
from axon_synthesis.synthesis.outputs import OutputConfig
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import disable_loggers

LOGGER = logging.getLogger(__name__)


def create_cell_collection(
    morphology_dir, output_path: FileType | None = None, convert_to: FileType | None = None
) -> CellCollection:
    """Create a CellCollection object from a directory containing morphologies."""
    morphology_dir = Path(morphology_dir)
    morph_files = [i for i in morphology_dir.iterdir() if is_morphology(i)]

    centers = np.zeros((len(morph_files), 3), dtype=float)
    if convert_to is not None:
        convert_to = Path(convert_to)
        convert_to.mkdir(parents=True, exist_ok=True)
        converted_files = []
        for num, file in enumerate(morph_files):
            converted_file = (convert_to / file.stem).with_suffix(".h5")
            morph = Morphology(file)
            centers[num] = morph.soma.center
            morph = morph.transform(Translation(-morph.soma.center))
            with disable_loggers("morph_tool.converter"):
                convert(morph, converted_file, nrn_order=True)
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
    cells_df[COORDS_COLS] = centers
    cells_df["orientation"] = [np.eye(3)] * len(cells_df)
    cells_df = cells_df.sort_values("morphology", ignore_index=True)
    cells_df.index += 1
    cells = CellCollection.from_dataframe(cells_df)
    if output_path is not None:
        cells.save(output_path)
    LOGGER.info("Found %s morphologies in %s", len(cells_df), morphology_dir)
    return cells


def create_probabilities(cells_df, tuft_properties):
    """Create the population and projection probabilities."""
    # Create population IDs with one unique population for each morphology-axon couple
    tuft_properties["population_id"] = tuft_properties.apply(
        lambda row: f"{row['morphology']}_{row['axon_id']}", axis=1
    )

    # Create population probabilities with all probs = 1
    population_probabilities = (
        tuft_properties[["population_id"]].drop_duplicates().reset_index(drop=True)
    )
    population_probabilities["brain_region_id"] = population_probabilities["population_id"]
    population_probabilities["probability"] = 1

    # Create projection probabilities with all probs = 1
    projection_probabilities = (
        (
            cells_df[["morphology", "morph_file"]].merge(
                tuft_properties[
                    [
                        "morphology",
                        "axon_id",
                        "tuft_id",
                        "population_id",
                        *COMMON_ANCESTOR_COORDS_COLS,
                    ]
                ],
                on="morphology",
                how="left",
            )
        )
        .dropna(subset="tuft_id")
        .rename(columns={"population_id": "source_population_id"})
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

    # Copy the dummy source brain region ID to cells_df
    cells_df["source_brain_region_id"] = cells_df.merge(
        projection_probabilities[["morphology", "source_brain_region_id"]].drop_duplicates(
            subset=["morphology"]
        ),
        on="morphology",
        how="left",
    )["source_brain_region_id"].to_numpy()

    return population_probabilities, projection_probabilities


def mimic_axons(
    morphology_dir: FileType,
    clustering_parameters: dict,
    *,
    atlas_config: AtlasConfig | None = None,
    create_graph_config: CreateGraphConfig | None = None,
    post_process_config: PostProcessConfig | None = None,
    output_config: OutputConfig | None = None,
    rng: SeedType = None,
    debug: bool = False,
    parallel_config: ParallelConfig | None = None,
):
    """Synthesize mimicking axons."""
    output_config = (
        output_config if output_config is not None else OutputConfig(DEFAULT_OUTPUT_PATH)
    )
    input_dir = output_config.path / "inputs"
    output_config.path = output_config.path / "synthesis"

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
        parallel_config=parallel_config,
    )

    # Modify the inputs for the mimic workflow

    # Convert the morphologies to h5 and create a CellCollection
    morphology_data_file = inputs.path / "circuit.h5"
    converted_morphologies_dir = inputs.path / "converted_morphologies"
    cells = create_cell_collection(morphology_dir, morphology_data_file, converted_morphologies_dir)
    cells_df = cells.as_dataframe()

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
    cells = CellCollection.from_dataframe(cells_df)
    cells.save(morphology_data_file)

    # Create the axon grafting point file
    axon_grafting_points_file = output_config.path.parent / "axon_grafting_points.h5"
    axon_grafting_points = (
        tuft_properties[["morphology", "population_id"]].drop_duplicates().reset_index(drop=True)
    )
    axon_grafting_points["grafting_section_id"] = -1
    axon_grafting_points["source_brain_region_id"] = axon_grafting_points["population_id"]
    axon_grafting_points["rebuilt_existing_axon_id"] = axon_grafting_points.groupby(
        "morphology"
    ).cumcount()
    axon_grafting_points.to_hdf(axon_grafting_points_file, key=AXON_GRAFTING_POINT_HDF_GROUP)

    # Synthesize the axons using the modified inputs
    synthesis_config = SynthesisConfig(
        converted_morphologies_dir,
        morphology_data_file,
        axon_grafting_points_file,
        input_dir,
        rebuild_existing_axons=True,
    )
    synthesize_axons(
        synthesis_config,
        atlas_config=atlas_config,
        create_graph_config=create_graph_config,
        post_process_config=post_process_config,
        output_config=output_config,
        rng=rng,
        parallel_config=parallel_config,
    )
