"""Validation workflow that mimics inputs morphologies."""
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import evolve
from morph_tool.converter import convert
from morph_tool.utils import is_morphology
from neurom.core import Morphology
from neurom.geom.transform import Translation
from voxcell.cell_collection import CellCollection

from axon_synthesis.constants import AXON_GRAFTING_POINT_HDF_GROUP
from axon_synthesis.constants import COMMON_ANCESTOR_COORDS_COLS
from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import DEFAULT_OUTPUT_PATH
from axon_synthesis.constants import DEFAULT_POPULATION
from axon_synthesis.constants import TARGET_COORDS_COLS
from axon_synthesis.inputs import Inputs
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
from axon_synthesis.utils import parallel_evaluator
from axon_synthesis.utils import setup_logger
from axon_synthesis.validation.atlas_utils import morph_atlas
from axon_synthesis.validation.utils import copy_morph_to_tmp_dir

LOGGER = logging.getLogger(__name__)


def create_cell_collection(
    morphology_dir, output_path: FileType | None = None, convert_to: FileType | None = None
) -> CellCollection:
    """Create a CellCollection object from a directory containing morphologies."""
    morphology_dir = Path(morphology_dir)
    raw_morph_files = [i for i in morphology_dir.iterdir() if is_morphology(i)]

    centers = np.zeros((len(raw_morph_files), 3), dtype=float)
    if convert_to is not None:
        convert_to = Path(convert_to)
        convert_to.mkdir(parents=True, exist_ok=True)
        morph_files = []
        for num, file in enumerate(raw_morph_files):
            converted_file = (convert_to / file.stem).with_suffix(".h5")
            morph = Morphology(file)
            centers[num] = morph.soma.center
            morph = morph.transform(Translation(-morph.soma.center))
            with disable_loggers("morph_tool.converter"):
                convert(morph, converted_file, nrn_order=True)
            morph_files.append(converted_file)
    else:
        morph_files = raw_morph_files

    morph_names = [i.stem for i in raw_morph_files]
    raw_morph_files = [str(i) for i in raw_morph_files]
    morph_files = [str(i) for i in morph_files]
    if not morph_files:
        msg = f"No morphology file found in '{morphology_dir}'"
        raise RuntimeError(msg)
    cells_df = pd.DataFrame(
        {"morphology": morph_names, "raw_file": raw_morph_files, "morph_file": morph_files}
    )
    cells_df["mtype"] = DEFAULT_POPULATION
    cells_df["region"] = DEFAULT_POPULATION
    cells_df[COORDS_COLS] = centers
    cells_df["orientation"] = [np.eye(3)] * len(cells_df)
    cells_df = cells_df.sort_values("morphology", ignore_index=True)
    cells_df.loc[:, "seed"] = cells_df.index.to_list()
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
        projection_probabilities["source_population_id"]
        + "-"
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


def update_cells(cells_df, projection_probabilities):
    """Update the cell collection according to the computed projection probabilities."""
    # Copy the dummy source brain region ID to cells_df
    cells_df["source_brain_region_id"] = cells_df.merge(
        projection_probabilities[["morphology", "source_brain_region_id"]].drop_duplicates(
            subset=["morphology"]
        ),
        on="morphology",
        how="left",
    )["source_brain_region_id"].to_numpy()

    # Drop morphologies with missing brain region IDs
    missing_brain_regions = cells_df.loc[cells_df["source_brain_region_id"].isna()]
    if not missing_brain_regions.empty:
        LOGGER.warning(
            "Could not find the source brain region of the following morphologies: %s",
            ", ".join(missing_brain_regions["morphology"].sort_values().tolist()),
        )
        cells_df.drop(missing_brain_regions.index, inplace=True)
        cells_df.reset_index(drop=True, inplace=True)
        cells_df.index += 1


def mimic_preferred_regions_workflow(
    morph,
    voxel_dimensions,
    synthesis_config,
    create_graph_config,
    post_process_config,
    output_config,
    rng,
    *,
    keep_tmp_atlas=False,
):
    """Synthesize axons using morphology preferred regions."""
    atlas_tmp_dir, atlas_tmp = morph_atlas(
        morph,
        voxel_dimensions=voxel_dimensions,
        preferred_regions_axon_id=0,  # TODO: Handle multiple axons case
        export=keep_tmp_atlas,
    )
    create_graph_config_tmp = evolve(
        create_graph_config,
        favored_regions=["dft"],
    )
    synthesize_axons(
        synthesis_config,
        atlas_config=atlas_tmp,
        create_graph_config=create_graph_config_tmp,
        post_process_config=post_process_config,
        output_config=output_config,
        rng=rng,
    )
    if not keep_tmp_atlas:
        atlas_tmp_dir.cleanup()


def run_workflows(
    data,
    workflows,
    synthesis_config,
    create_graph_config,
    post_process_config,
    output_config,
    rng,
    *,
    voxel_dimensions=None,
    keep_tmp_atlas=False,
):
    """Run all the workflows on each morphology."""
    morph_name = data["morphology"]

    # Prepare specific inputs for the current morphology
    tmp_dir, _filename = copy_morph_to_tmp_dir(data["raw_file"])
    synthesis_config_tmp = evolve(
        synthesis_config,
        morphology_dir=tmp_dir.name,
        morphology_data_file=Path(tmp_dir.name) / "circuit.h5",
    )

    # Create a circuit containing only the current morphology
    cells_df = CellCollection.load(synthesis_config.morphology_data_file).as_dataframe()
    cell_row = cells_df.loc[cells_df["morphology"] == morph_name].reset_index(drop=True)
    cell_row.index += 1
    CellCollection.from_dataframe(cell_row).save(synthesis_config_tmp.morphology_data_file)

    # Synthesize each morphology
    for workflow in workflows:
        output_config_tmp = evolve(
            output_config,
            path=output_config.path.with_name(
                output_config.path.name + f"_{workflow}_{morph_name}"
            ),
        )
        if workflow == "basic":
            LOGGER.info("Starting 'basic' workflow for %s", morph_name)
            synthesize_axons(
                synthesis_config_tmp,
                create_graph_config=create_graph_config,
                post_process_config=post_process_config,
                output_config=output_config_tmp,
                rng=rng,
            )
        elif workflow == "preferred_regions":
            LOGGER.info("Starting 'preferred regions' workflow for %s", morph_name)
            if voxel_dimensions is None:
                msg = "A 'voxel_dimensions' value is required for this workflow"
                raise ValueError(msg)

            mimic_preferred_regions_workflow(
                data["morph_file"],
                voxel_dimensions,
                synthesis_config_tmp,
                create_graph_config=create_graph_config,
                post_process_config=post_process_config,
                output_config=output_config_tmp,
                rng=rng,
                keep_tmp_atlas=keep_tmp_atlas,
            )

    # Cleanup tmp files
    tmp_dir.cleanup()

    return {}


def export_cells_df(cells_df, path):
    """Export cells DF for future debugging."""
    cells_df = cells_df.copy(deep=False)
    cells_df["orientation"] = cells_df["orientation"].apply(np.ndarray.tolist)
    cells_df.to_feather(path)


def mimic_axons(  # noqa: PLR0913
    morphology_dir: FileType,
    clustering_parameters: dict,
    *,
    create_graph_config: CreateGraphConfig | None = None,
    post_process_config: PostProcessConfig | None = None,
    output_config: OutputConfig | None = None,
    workflows: list[str] | None = None,
    voxel_dimensions: int | None = None,
    rng: SeedType = None,
    debug: bool = False,
    parallel_config: ParallelConfig | None = None,
    keep_tmp_atlas: bool = False,
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

    # Create raw inputs if they don't exist
    if not input_dir.exists():
        inputs = create_inputs(
            morphology_dir,
            input_dir,
            clustering_parameters,
            rng=rng,
            debug=debug,
            parallel_config=parallel_config,
        )
    else:
        inputs = Inputs(input_dir, morphology_dir)
        inputs.load_clustering_data()

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

    # Update the cell properties
    update_cells(cells_df, inputs.projection_probabilities)

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
    cells_df["st_level"] = 0
    export_cells_df(cells_df, output_config.path.parent / "cells_df.feather")
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

    # Choose the workflows to process
    if workflows is None:
        workflows = ["basic"]

    parallel_evaluator(
        cells_df,
        run_workflows,
        parallel_config,
        [],
        func_kwargs={
            "workflows": workflows,
            "synthesis_config": synthesis_config,
            "create_graph_config": create_graph_config,
            "post_process_config": post_process_config,
            "output_config": output_config,
            "rng": rng,
            "voxel_dimensions": voxel_dimensions,
            "keep_tmp_atlas": keep_tmp_atlas,
        },
        progress_bar=False,
        startup_func=partial(setup_logger, level=logging.getLevelName(LOGGER.getEffectiveLevel())),
    )
