"""Base of the synthesis modules."""
import json
import logging
import os
from pathlib import Path
from typing import ClassVar

import dask.dataframe as dd
import dask.distributed
import numpy as np
import pandas as pd
from attrs import define
from attrs import field
from attrs import validators
from neurom import NeuriteType
from neurom.core import Morphology
from neurom.geom.transform import Translation
from voxcell.cell_collection import CellCollection

try:
    import dask_mpi
    from mpi4py import MPI

    mpi_enabled = True
except ImportError:
    mpi_enabled = False

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.base_path_builder import FILE_SELECTION
from axon_synthesis.base_path_builder import BasePathBuilder
from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import DEFAULT_OUTPUT_PATH
from axon_synthesis.constants import TARGET_COORDS_COLS
from axon_synthesis.inputs import Inputs
from axon_synthesis.synthesis.add_tufts import build_and_graft_tufts
from axon_synthesis.synthesis.main_trunk.create_graph import CreateGraphConfig
from axon_synthesis.synthesis.main_trunk.create_graph import one_graph
from axon_synthesis.synthesis.main_trunk.post_process import PostProcessConfig
from axon_synthesis.synthesis.main_trunk.post_process import post_process_trunk
from axon_synthesis.synthesis.main_trunk.steiner_morphology import build_and_graft_trunk
from axon_synthesis.synthesis.main_trunk.steiner_tree import compute_solution
from axon_synthesis.synthesis.outputs import OutputConfig
from axon_synthesis.synthesis.outputs import Outputs
from axon_synthesis.synthesis.plot import plot_final_morph
from axon_synthesis.synthesis.plot import plot_target_points
from axon_synthesis.synthesis.source_points import SOURCE_COORDS_COLS
from axon_synthesis.synthesis.source_points import set_source_points
from axon_synthesis.synthesis.target_points import get_target_points
from axon_synthesis.synthesis.tuft_properties import pick_barcodes
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import MorphNameAdapter
from axon_synthesis.utils import load_morphology
from axon_synthesis.utils import save_morphology

LOGGER = logging.getLogger(__name__)

_HDF_DEFAULT_GROUP = "axon_grafting_points"


@define
class ParallelConfig:
    """Class to store the parallel configuration.

    Attributes:
        nb_processes: The number of processes.
        dask_config: The dask configuration to use.
        progress_bar: If set to True, a progress bar is displayed during computation.
        use_mpi: Trigger the use of MPI.
    """

    nb_processes: int = field(default=0, validator=validators.ge(0))
    dask_config: dict | None = field(default=None)
    progress_bar: bool = field(default=True)
    use_mpi: bool = field(default=False)


def load_axon_grafting_points(path: FileType | None = None, key: str = _HDF_DEFAULT_GROUP):
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


def one_axon_paths(outputs, morph_file_name, figure_file_name):
    """Create a BasePathBuilder object to store the paths needed for a specific axon."""

    class AxonPaths(BasePathBuilder):
        """Class to store the synthesis outputs of one specific axon."""

        _filenames: ClassVar[dict] = {
            "FIGURE_FILE_NAME": figure_file_name,
            "GRAPH_CREATION_FIGURE": (outputs.GRAPH_CREATION_FIGURES / figure_file_name)
            if outputs.GRAPH_CREATION_FIGURES is not None
            else None,
            "GRAPH_CREATION_DATA": (outputs.GRAPH_CREATION_DATA / morph_file_name)
            if outputs.GRAPH_CREATION_DATA is not None
            else None,
            "MAIN_TRUNK_FIGURE": (outputs.MAIN_TRUNK_FIGURES / figure_file_name)
            if outputs.MAIN_TRUNK_FIGURES is not None
            else None,
            "MAIN_TRUNK_MORPHOLOGY": (outputs.MAIN_TRUNK_MORPHOLOGIES / morph_file_name)
            if outputs.MAIN_TRUNK_MORPHOLOGIES is not None
            else None,
            "MORPH_FILE_NAME": morph_file_name,
            "POSTPROCESS_TRUNK_FIGURE": (outputs.POSTPROCESS_TRUNK_FIGURES / figure_file_name)
            if outputs.POSTPROCESS_TRUNK_FIGURES is not None
            else None,
            "POSTPROCESS_TRUNK_MORPHOLOGY": (
                outputs.POSTPROCESS_TRUNK_MORPHOLOGIES / morph_file_name
            )
            if outputs.POSTPROCESS_TRUNK_MORPHOLOGIES is not None
            else None,
            "STEINER_TREE_SOLUTION": (outputs.STEINER_TREE_SOLUTIONS / morph_file_name)
            if outputs.STEINER_TREE_SOLUTIONS is not None
            else None,
            "TARGET_POINT_FIGURE": (outputs.TARGET_POINT_FIGURES / figure_file_name)
            if outputs.TARGET_POINT_FIGURES is not None
            else None,
            "TUFT_FIGURES": outputs.TUFT_FIGURES if outputs.TUFT_FIGURES is not None else None,
            "TUFT_MORPHOLOGIES": outputs.TUFT_MORPHOLOGIES
            if outputs.TUFT_MORPHOLOGIES is not None
            else None,
        }

        _optional_keys: ClassVar[set[str]] = {
            "GRAPH_CREATION_FIGURE",
            "GRAPH_CREATION_DATA",
            "MAIN_TRUNK_FIGURE",
            "MAIN_TRUNK_MORPHOLOGY",
            "POSTPROCESS_TRUNK_FIGURE",
            "POSTPROCESS_TRUNK_MORPHOLOGY",
            "STEINER_TREE_SOLUTION",
            "TARGET_POINT_FIGURE",
            "TUFT_FIGURES",
            "TUFT_MORPHOLOGIES",
        }

    return AxonPaths("")


def _init_parallel(
    parallel_config: ParallelConfig, *, mpi_only: bool = False, max_nb_processes: int | None = None
) -> dask.distributed.Client | None:
    """Initialize MPI workers if required or get the number of available processes."""
    if mpi_only and not parallel_config.use_mpi:
        return None

    if parallel_config.nb_processes == 0 and not parallel_config.use_mpi:
        return None

    # Define a default configuration to disable some dask.distributed things
    default_dask_config = {
        "distributed": {
            "worker": {
                "use_file_locking": False,
                "memory": {
                    "target": False,
                    "spill": False,
                    "pause": 0.8,
                    "terminate": 0.95,
                },
                "profile": {
                    "enabled": False,
                    "interval": "10s",
                    "cycle": "10m",
                },
            },
            "admin": {
                "tick": {
                    "limit": "1h",
                },
            },
            "comm": {
                "retry": {
                    "count": 10,
                },
                "timeouts": {
                    "connect": 30,
                },
            },
        },
        "dataframe": {
            "convert_string": False,
        },
    }

    # Merge the default config with the existing config (keep conflicting values from defaults)
    new_dask_config = dask.config.merge(dask.config.config, default_dask_config)

    # Get temporary-directory from environment variables
    _tmp = os.environ.get("SHMDIR", None) or os.environ.get("TMPDIR", None)
    if _tmp is not None:
        new_dask_config["temporary-directory"] = _tmp

    # Merge the config with the one given as argument
    if parallel_config.dask_config is not None:
        new_dask_config = dask.config.merge(new_dask_config, parallel_config.dask_config)

    # Set the dask config
    dask.config.set(new_dask_config)

    if parallel_config.use_mpi:  # pragma: no cover
        if not mpi_enabled:
            msg = (
                "The 'dask' and 'mpi4py' packages must be installed when using the MPI parallel "
                "backend"
            )
            raise ImportError(msg)
        dask_mpi.initialize(dashboard=False)
        comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
        parallel_config.nb_processes = comm.Get_size()
        client_kwargs = {}
        LOGGER.debug(
            "Initializing parallel workers using MPI (%s workers found)",
            parallel_config.nb_processes,
        )
    elif mpi_only:
        return None
    else:
        cpu_count: int = os.cpu_count() or 0
        parallel_config.nb_processes = min(
            parallel_config.nb_processes if parallel_config.nb_processes is not None else cpu_count,
            max_nb_processes if max_nb_processes is not None else cpu_count,
        )

        client_kwargs = {
            "n_workers": parallel_config.nb_processes,
            "dashboard_address": None,
        }
        LOGGER.debug("Initializing parallel workers using the following config: %s", client_kwargs)

    LOGGER.debug("Using the following dask configuration: %s", json.dumps(dask.config.config))

    # This is needed to make dask aware of the workers
    return dask.distributed.Client(**client_kwargs)


def synthesize_one_morph_axons(
    morph_terminals,
    inputs,
    outputs,
    create_graph_config,
    post_process_config,
    *,
    rebuild_existing_axons=False,
    logger=None,
):
    """Synthesize the axons of one morphology."""
    if logger is None:
        logger = LOGGER
    morph_name = morph_terminals.name
    morph_custom_logger = MorphNameAdapter(logger, extra={"morph_name": morph_name})
    morph_custom_logger.debug("Starting synthesis")
    try:
        morph = load_morphology(morph_terminals["morph_file"].to_numpy()[0])

        # Translate the morphology to its position in the atlas
        morph = morph.transform(
            Translation(morph_terminals[COORDS_COLS].to_numpy()[0] - morph.soma.center)
        )

        morph.name = morph_name

        initial_morph = (
            Morphology(morph, name=morph.name) if outputs.FINAL_FIGURES is not None else None
        )

        if rebuild_existing_axons:
            # Remove existing axons
            morph_custom_logger.info("Removing existing axons")
            remove_existing_axons(morph)

        for axon_id, axon_terminals in morph_terminals.groupby("axon_id"):
            # Create a custom logger to add the morph name and axon ID in the log entries
            axon_custom_logger = MorphNameAdapter(
                logger, extra={"morph_name": morph_name, "axon_id": axon_id}
            )

            rng = np.random.default_rng(axon_terminals["seed"].to_numpy()[0])

            axon_paths = one_axon_paths(
                outputs,
                f"{morph_name}_{axon_id}.h5",
                f"{morph_name}_{axon_id}.html",
            )

            # Create a plot for the initial morph with source and target points
            if axon_paths.TARGET_POINT_FIGURE is not None:
                plot_target_points(
                    initial_morph,
                    axon_terminals[SOURCE_COORDS_COLS].to_numpy()[0],
                    axon_terminals[TARGET_COORDS_COLS].to_numpy(),
                    axon_paths.TARGET_POINT_FIGURE,
                    logger=axon_custom_logger,
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
                output_path=axon_paths.GRAPH_CREATION_DATA,
                figure_path=axon_paths.GRAPH_CREATION_FIGURE,
                logger=axon_custom_logger,
            )

            # Build the Steiner Tree for the current axon
            _, solution_edges = compute_solution(
                nodes,
                edges,
                output_path=axon_paths.STEINER_TREE_SOLUTION,
                logger=axon_custom_logger,
            )

            # Create the trunk morphology
            trunk_section_id = build_and_graft_trunk(
                morph,
                axon_terminals["grafting_section_id"].to_numpy()[0],
                solution_edges,
                output_path=axon_paths.MAIN_TRUNK_MORPHOLOGY,
                figure_path=axon_paths.MAIN_TRUNK_FIGURE,
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
                post_process_config,
                rng=rng,
                output_path=axon_paths.POSTPROCESS_TRUNK_MORPHOLOGY,
                figure_path=axon_paths.POSTPROCESS_TRUNK_FIGURE,
                logger=axon_custom_logger,
            )

            # Create the tufts for each barcode
            build_and_graft_tufts(
                morph,
                barcodes,
                inputs.tuft_parameters,
                inputs.tuft_distributions,
                rng=rng,
                output_dir=axon_paths.TUFT_MORPHOLOGIES,
                figure_dir=axon_paths.TUFT_FIGURES,
                initial_morph=initial_morph,
                logger=axon_custom_logger,
            )

            # TODO: Diametrize the synthesized axon

        # Export the final morph
        final_morph_path = outputs.MORPHOLOGIES / f"{morph_name}.h5"
        save_morphology(
            morph,
            final_morph_path,
            msg=f"Export synthesized morphology to {final_morph_path}",
            logger=morph_custom_logger,
        )

        # Create a plot for the final morph
        if outputs.FINAL_FIGURES is not None:
            plot_final_morph(
                morph,
                morph_terminals,
                outputs.FINAL_FIGURES / f"{morph_name}.html",
                initial_morph,
                logger=morph_custom_logger,
            )
        res = {
            "file_path": final_morph_path,
            "debug_infos": None,
        }
    except Exception as exc:
        morph_custom_logger.exception(
            "Skip the morphology because of the following error",
        )
        res = {
            "file_path": None,
            "debug_infos": str(exc),
        }
    return pd.Series(res, dtype=object)


def synthesize_group_morph_axons(df: pd.DataFrame, inputs: Inputs, **func_kwargs) -> pd.DataFrame:
    """Synthesize all axons of each morphology."""
    if "target_orientation" not in df.columns:
        if inputs.atlas is not None:
            target_orientations = inputs.atlas.orientations.lookup(
                df[TARGET_COORDS_COLS].to_numpy()
            )
            missing_orientations = np.isnan(target_orientations).any(axis=(1, 2))
            if missing_orientations.any():
                target_orientations[missing_orientations] = np.repeat(
                    [np.eye(3)], missing_orientations.sum(), axis=0
                )
            df["target_orientation"] = target_orientations.tolist()
        else:
            df["target_orientation"] = np.repeat([np.eye(3)], len(df), axis=0).tolist()

    return df.groupby("morphology", group_keys=True).apply(
        lambda group: synthesize_one_morph_axons(group, inputs=inputs, **func_kwargs)
    )


def _partition_wrapper(
    df: pd.DataFrame,
    input_path: FileType,
    atlas_config: AtlasConfig | None,
    **func_kwargs,
) -> pd.DataFrame:
    """Wrapper to process dask partitions."""
    inputs = Inputs(input_path)
    if atlas_config is not None:
        atlas_config.load_region_map = False
        inputs.load_atlas(atlas_config)
    inputs.load_clustering_data()
    inputs.load_probabilities()
    inputs.load_tuft_params_and_distrs()

    return synthesize_group_morph_axons(df.copy(deep=False), inputs=inputs, **func_kwargs)


def create_dask_dataframe(data: pd.DataFrame, npartitions: int, group_col="morphology"):
    """Ensure all rows of the same group belong to the same partition."""
    ddf = dd.from_pandas(data, npartitions)
    if len(ddf.divisions) > 2:
        groups = np.array_split(data[group_col].unique(), npartitions)
        new_divisions = [
            data.loc[data[group_col].isin(i)].index.min() for i in groups if len(i) > 0
        ] + [data.index.max()]
        ddf = ddf.repartition(divisions=new_divisions)
    return ddf


def synthesize_axons(  # noqa: PLR0912, PLR0913
    input_dir: FileType,
    morphology_data_file: FileType,
    morphology_dir: FileType,
    axon_grafting_points_file: FileType | None = None,
    output_config: OutputConfig | None = None,
    *,
    atlas_config: AtlasConfig | None = None,
    create_graph_config: CreateGraphConfig | None = None,
    post_process_config: PostProcessConfig | None = None,
    rebuild_existing_axons: bool = False,
    rng: SeedType = None,
    parallel_config: ParallelConfig | None = None,
):  # pylint: disable=too-many-arguments
    """Synthesize the long-range axons.

    Args:
        input_dir: The directory containing the inputs.
        morphology_data_file: The path to the MVD3/sonata file.
        morphology_dir: The directory containing the input morphologies.
        axon_grafting_points_file: The file containing the grafting points.
        atlas_config: The config used to load the Atlas.
        create_graph_config: The config used to create the graph.
        post_process_config: The config used to post-process the long-range trunk.
        output_config: The config used to adjust the outputs.
        rebuild_existing_axons: Rebuild the axons if they already exist.
        rng: The random seed or the random generator.
        parallel_config: The configuration for parallel computation.
    """
    if parallel_config is None:
        parallel_config = ParallelConfig()
    _parallel_client = _init_parallel(parallel_config)

    rng = np.random.default_rng(rng)
    outputs = Outputs(
        output_config if output_config is not None else OutputConfig(DEFAULT_OUTPUT_PATH),
        create=True,
    )
    outputs.create_dirs(file_selection=FILE_SELECTION.REQUIRED_ONLY)

    # Load all inputs
    if atlas_config is not None:
        atlas_config.load_region_map = True
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

    if post_process_config is None:
        post_process_config = PostProcessConfig()

    # Get source points for all axons
    source_points = set_source_points(
        cells_df,
        inputs.atlas,
        morphology_dir,
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
        output_path=outputs.TARGET_POINTS,
    )
    if rebuild_existing_axons:
        # If the existing axons are rebuilt all the new axons will be grafted to the soma
        target_points["grafting_section_id"] = -1

    if "seed" not in target_points.columns:
        # If no random seed is provided in the data, new ones are created for each morphologys
        target_points = target_points.merge(
            pd.Series(1, index=target_points["morphology"].unique(), name="seed")
            .sample(frac=1, random_state=rng)
            .cumsum()
            - 1,
            left_on="morphology",
            right_index=True,
            how="left",
        )

    func_kwargs = {
        "outputs": outputs,
        "create_graph_config": create_graph_config,
        "post_process_config": post_process_config,
        "rebuild_existing_axons": rebuild_existing_axons,
        "logger": LOGGER,
    }

    if parallel_config.nb_processes == 0:
        LOGGER.info("Start computation")
        computed = synthesize_group_morph_axons(target_points, inputs=inputs, **func_kwargs)
    else:
        LOGGER.info("Start parallel computation using %s workers", parallel_config.nb_processes)
        ddf = create_dask_dataframe(
            target_points, parallel_config.nb_processes, group_col="morphology"
        )
        meta = pd.DataFrame(
            {
                name: pd.Series(dtype="object")
                for name in [
                    "file_path",
                    "debug_infos",
                ]
            }
        )
        future = ddf.map_partitions(
            _partition_wrapper,
            meta=meta,
            input_path=inputs.path,
            atlas_config=inputs.atlas.config if inputs.atlas is not None else None,
            **func_kwargs,
        )
        computed = future.compute()

    LOGGER.info("Format results")
    res = target_points.merge(computed, left_on="morphology", right_index=True, how="left")
    if res["file_path"].isna().any():
        LOGGER.error(
            "The following morphologies could not be synthesized (see the logs for details): %s",
            res.loc[res["file_path"].isna(), "morphology"].unique().tolist(),
        )
    LOGGER.info(
        "Synthesized %s morphologies", len(res.loc[~res["file_path"].isna(), "morphology"].unique())
    )
    try:
        if _parallel_client is not None:
            _parallel_client.close()
    except Exception:
        LOGGER.exception(
            "The internal parallel client may not have been closed properly because of this error"
        )
    del _parallel_client
    return res
