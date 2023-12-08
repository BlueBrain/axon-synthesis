"""Base of the synthesis modules."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from neurom import NeuriteType
from neurom.core import Morphology
from voxcell.cell_collection import CellCollection

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.base_path_builder import FILE_SELECTION
from axon_synthesis.inputs import Inputs
from axon_synthesis.synthesis.main_trunk.create_graph import CreateGraphConfig
from axon_synthesis.synthesis.main_trunk.create_graph import one_graph
from axon_synthesis.synthesis.main_trunk.steiner_morphology import build_and_graft_trunk
from axon_synthesis.synthesis.main_trunk.steiner_tree import compute_solution
from axon_synthesis.synthesis.outputs import Outputs
from axon_synthesis.synthesis.source_points import SOURCE_COORDS_COLS
from axon_synthesis.synthesis.source_points import set_source_points
from axon_synthesis.synthesis.target_points import get_target_points
from axon_synthesis.synthesis.tuft_properties import create as create_tuft_properties
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import MorphNameAdapter

LOGGER = logging.getLogger(__name__)

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


def remove_existing_axons(morph):
    """Remove all existing axons from a given morphology."""
    for i in morph.root_sections:
        if i.type == NeuriteType.axon:
            morph.delete_section(i)


def synthesize_axons(  # noqa: PLR0913
    input_dir: FileType,
    output_dir: FileType,
    morphology_data_file: FileType,
    morphology_path: FileType,
    morphology_ext: str,
    axon_grafting_points_file: FileType = None,
    *,
    atlas_config: AtlasConfig | None = None,
    create_graph_config: CreateGraphConfig | None = None,
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
        create_graph_config: The config used to create the graph.
        rebuild_existing_axons: Rebuild the axons if they already exist.
        seed: The random seed.
        debug: Trigger the Debug mode.
    """
    rng = np.random.default_rng(seed)
    outputs = Outputs(output_dir, create=True)
    outputs.create_dirs(
        file_selection=FILE_SELECTION.ALL if debug else FILE_SELECTION.REQUIRED_ONLY
    )

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
        output_path=outputs.TARGET_POINTS if debug else None,
    )
    if rebuild_existing_axons:
        # If the existing axons are rebuilt all the new axons will be grafted to the soma
        target_points["grafting_section_id"] = -1

    # Ensure the graph creation config is complete
    if create_graph_config is None:
        create_graph_config = CreateGraphConfig()
    create_graph_config.compute_region_tree(inputs.atlas)

    for morph_name, morph_terminals in target_points.groupby("morphology"):
        morph = Morphology(morph_terminals["morph_file"].to_numpy()[0])

        if rebuild_existing_axons:
            # Remove existing axons and set grafting mode to soma
            remove_existing_axons(morph)

        for axon_id, axon_terminals in morph_terminals.groupby("axon_id"):
            # Create a custom logger to add the morph name and axon ID in the log entries
            custom_logger = MorphNameAdapter(
                LOGGER, extra={"morph_name": morph_name, "axon_id": axon_id}
            )

            file_name = f"{morph_name}_{axon_id}.h5"

            # Create the graph for each axon
            nodes, edges = one_graph(
                inputs.atlas,
                morph_terminals[SOURCE_COORDS_COLS].to_numpy()[0],
                axon_terminals,
                create_graph_config,
                favored_region_tree=create_graph_config.favored_region_tree,
                rng=rng,
                output_path=outputs.GRAPH_CREATION / file_name if debug else None,
                logger=custom_logger,
            )

            # Build the Steiner Tree for each axon
            solution_nodes, solution_edges = compute_solution(
                nodes,
                edges,
                output_path=outputs.STEINER_TREE_SOLUTIONS / file_name if debug else None,
                logger=custom_logger,
            )

            # Create the trunk morphology
            build_and_graft_trunk(
                morph,
                # source_coords,
                axon_terminals["grafting_section_id"].to_numpy()[0],
                solution_nodes,
                solution_edges,
                output_path=(outputs.MAIN_TRUNK_MORPHOLOGIES / file_name if debug else None),
                logger=custom_logger,
            )

            # Create the tufts for each axon
            create_tuft_properties(morph, inputs.atlas, axon_terminals, inputs.cluster_props_df)

            # Graft the axon to the morph

        morph.write(outputs.MORPHOLOGIES / f"{morph_name}.h5")
