"""Base of the synthesis modules."""
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from neurom import NeuriteType
from neurom.core import Morphology
from neurom.geom.transform import Translation
from voxcell.cell_collection import CellCollection

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.base_path_builder import FILE_SELECTION
from axon_synthesis.base_path_builder import BasePathBuilder
from axon_synthesis.inputs import Inputs
from axon_synthesis.synthesis.add_tufts import build_and_graft_tufts
from axon_synthesis.synthesis.main_trunk.create_graph import CreateGraphConfig
from axon_synthesis.synthesis.main_trunk.create_graph import one_graph
from axon_synthesis.synthesis.main_trunk.post_process import post_process_trunk
from axon_synthesis.synthesis.main_trunk.steiner_morphology import build_and_graft_trunk
from axon_synthesis.synthesis.main_trunk.steiner_tree import compute_solution
from axon_synthesis.synthesis.outputs import Outputs
from axon_synthesis.synthesis.plot import plot_final_morph
from axon_synthesis.synthesis.source_points import SOURCE_COORDS_COLS
from axon_synthesis.synthesis.source_points import set_source_points
from axon_synthesis.synthesis.target_points import get_target_points
from axon_synthesis.synthesis.tuft_properties import pick_barcodes
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import COORDS_COLS
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


def create_one_axon_paths(outputs, morph_file_name, figure_file_name, *, debug=False):
    """Create a BasePathBuilder object to store the paths needed for a specific axon."""

    class AxonPaths(BasePathBuilder):
        """Class to store the synthesis outputs of one specific axon."""

        _filenames: ClassVar[dict] = {
            "FIGURE_FILE_NAME": figure_file_name,
            "GRAPH_CREATION_FIGURE": (outputs.GRAPH_CREATION_FIGURES / figure_file_name)
            if debug
            else None,
            "GRAPH_CREATION_DATA": (outputs.GRAPH_CREATION_DATA / morph_file_name)
            if debug
            else None,
            "MAIN_TRUNK_FIGURE": (outputs.MAIN_TRUNK_FIGURES / figure_file_name) if debug else None,
            "MAIN_TRUNK_MORPHOLOGY": (outputs.MAIN_TRUNK_MORPHOLOGIES / morph_file_name)
            if debug
            else None,
            "MORPH_FILE_NAME": morph_file_name,
            "POSTPROCESS_TRUNK_FIGURE": (outputs.POSTPROCESS_TRUNK_FIGURES / figure_file_name)
            if debug
            else None,
            "POSTPROCESS_TRUNK_MORPHOLOGY": (
                outputs.POSTPROCESS_TRUNK_MORPHOLOGIES / morph_file_name
            )
            if debug
            else None,
            "STEINER_TREE_SOLUTION": (outputs.STEINER_TREE_SOLUTIONS / morph_file_name)
            if debug
            else None,
            "TUFT_FIGURES": outputs.TUFT_FIGURES if debug else None,
            "TUFT_MORPHOLOGIES": outputs.TUFT_MORPHOLOGIES if debug else None,
        }

        _optional_keys: ClassVar[set[str]] = {
            "GRAPH_CREATION_FIGURE",
            "GRAPH_CREATION_DATA",
            "MAIN_TRUNK_FIGURE",
            "MAIN_TRUNK_MORPHOLOGY",
            "POSTPROCESS_TRUNK_FIGURE",
            "POSTPROCESS_TRUNK_MORPHOLOGY",
            "STEINER_TREE_SOLUTION",
            "TUFT_FIGURES",
            "TUFT_MORPHOLOGIES",
        }

    return AxonPaths("")


def synthesize_axons(  # noqa: PLR0913
    input_dir: FileType,
    output_dir: FileType,
    morphology_data_file: FileType,
    morphology_dir: FileType,
    morphology_ext: str,
    axon_grafting_points_file: FileType = None,
    *,
    atlas_config: AtlasConfig | None = None,
    create_graph_config: CreateGraphConfig | None = None,
    rebuild_existing_axons: bool = False,
    rng: SeedType = None,
    debug: bool = False,
):  # pylint: disable=too-many-arguments
    """Synthesize the long-range axons.

    Args:
        input_dir: The directory containing the inputs.
        output_dir: The directory containing the outputs.
        morphology_data_file: The path to the MVD3/sonata file.
        morphology_dir: The directory containing the input morphologies.
        morphology_ext: The extension of the input morphologies.
        axon_grafting_points_file: The file containing the grafting points.
        atlas_config: The config used to load the Atlas.
        create_graph_config: The config used to create the graph.
        rebuild_existing_axons: Rebuild the axons if they already exist.
        rng: The random seed or the random generator.
        debug: Trigger the Debug mode.
    """
    rng = np.random.default_rng(rng)
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

    # Ensure the graph creation config is complete
    if create_graph_config is None:
        create_graph_config = CreateGraphConfig()
    if inputs.atlas is not None:
        create_graph_config.compute_region_tree(inputs.atlas)
    LOGGER.debug("The following config is used for graph creation: %s", create_graph_config)

    # Get source points for all axons
    source_points = set_source_points(
        cells_df,
        inputs.atlas,
        morphology_dir,
        morphology_ext,
        inputs.population_probabilities,
        axon_grafting_points,
        rng=rng,
        rebuild_existing_axons=rebuild_existing_axons,
    )

    # Find targets for all axons
    target_points = get_target_points(
        source_points,
        inputs.projection_probabilities,
        create_graph_config.duplicate_precision,
        atlas=inputs.atlas,
        brain_regions_masks=inputs.brain_regions_mask_file,
        rng=rng,
        output_path=outputs.TARGET_POINTS if debug else None,
    )
    if rebuild_existing_axons:
        # If the existing axons are rebuilt all the new axons will be grafted to the soma
        target_points["grafting_section_id"] = -1

    for morph_name, morph_terminals in target_points.groupby("morphology"):
        morph = Morphology(morph_terminals["morph_file"].to_numpy()[0])
        morph_custom_logger = MorphNameAdapter(LOGGER, extra={"morph_name": morph_name})

        # Translate the morphology to its position in the atlas
        morph = morph.transform(
            Translation(morph_terminals[COORDS_COLS].to_numpy()[0] - morph.soma.center)
        )

        morph.name = morph_name

        initial_morph = Morphology(morph) if debug else None

        if rebuild_existing_axons:
            # Remove existing axons
            morph_custom_logger.info("Removing existing axons")
            remove_existing_axons(morph)

        for axon_id, axon_terminals in morph_terminals.groupby("axon_id"):
            # Create a custom logger to add the morph name and axon ID in the log entries
            axon_custom_logger = MorphNameAdapter(
                LOGGER, extra={"morph_name": morph_name, "axon_id": axon_id}
            )

            one_axon_paths = create_one_axon_paths(
                outputs,
                f"{morph_name}_{axon_id}.h5",
                f"{morph_name}_{axon_id}.html",
                debug=debug,
            )

            # Create the graph for the current axon
            nodes, edges = one_graph(
                axon_terminals[SOURCE_COORDS_COLS].to_numpy()[0],
                axon_terminals,
                create_graph_config,
                bbox=inputs.atlas.brain_regions.bbox if inputs.atlas is not None else None,
                depths=inputs.atlas.depths if inputs.atlas is not None else None,
                favored_region_tree=create_graph_config.favored_region_tree,
                rng=rng,
                output_path=one_axon_paths.GRAPH_CREATION_DATA,
                figure_path=one_axon_paths.GRAPH_CREATION_FIGURE,
                logger=axon_custom_logger,
            )

            # Build the Steiner Tree for the current axon
            _, solution_edges = compute_solution(
                nodes,
                edges,
                output_path=one_axon_paths.STEINER_TREE_SOLUTION,
                logger=axon_custom_logger,
            )

            # Create the trunk morphology
            trunk_section_id = build_and_graft_trunk(
                morph,
                axon_terminals["grafting_section_id"].to_numpy()[0],
                solution_edges,
                output_path=one_axon_paths.MAIN_TRUNK_MORPHOLOGY,
                figure_path=one_axon_paths.MAIN_TRUNK_FIGURE,
                initial_morph=initial_morph,
                logger=axon_custom_logger,
            )

            # Choose a barcode for each tuft of the current axon
            barcodes = pick_barcodes(
                axon_terminals,
                solution_edges,
                inputs.clustering_data.tuft_properties,
                rng=rng,
                logger=axon_custom_logger,
            )

            # Post-process the trunk
            post_process_trunk(
                morph,
                trunk_section_id,
                inputs.clustering_data.trunk_properties,
                barcodes,
                rng=rng,
                output_path=one_axon_paths.POSTPROCESS_TRUNK_MORPHOLOGY,
                figure_path=one_axon_paths.POSTPROCESS_TRUNK_FIGURE,
                initial_morph=initial_morph,
                logger=axon_custom_logger,
            )

            # Create the tufts for each barcode
            build_and_graft_tufts(
                morph,
                barcodes,
                inputs.tuft_parameters,
                inputs.tuft_distributions,
                rng=rng,
                output_dir=one_axon_paths.TUFT_MORPHOLOGIES,
                figure_dir=one_axon_paths.TUFT_FIGURES,
                initial_morph=initial_morph,
                logger=axon_custom_logger,
            )

            # TODO: Diametrize the synthesized axon

        final_morph_path = outputs.MORPHOLOGIES / f"{morph_name}.h5"
        morph.write(final_morph_path)
        morph_custom_logger.info("Exported synthesized morphology to %s", final_morph_path)
        if debug:
            plot_final_morph(
                morph,
                outputs.FINAL_FIGURES / f"{morph_name}.html",
                initial_morph,
                logger=morph_custom_logger,
            )
