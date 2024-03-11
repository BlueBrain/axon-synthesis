"""Module with tools to compute the projection intensity."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import converters
from attrs import define
from attrs import field
from attrs import validators
from bluepyparallel import evaluate
from bluepyparallel import init_parallel_factory
from dask.distributed import LocalCluster
from morph_tool.utils import is_morphology
from neurom import COLS
from voxcell import VoxelData
from voxcell.math_utils import voxel_intersection

from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.synthesis import ParallelConfig
from axon_synthesis.typing import FileType
from axon_synthesis.utils import MorphNameAdapter
from axon_synthesis.utils import compute_bbox
from axon_synthesis.utils import disable_loggers
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import load_morphology
from axon_synthesis.utils import neurite_to_pts

LOGGER = logging.getLogger(__name__)


def to_tuple(x: object) -> tuple:
    """Convert input data into a tuple."""
    try:
        new_x = json.loads(x)
    except Exception:
        new_x = x
    return tuple(new_x)


@define
class ProjectionIntensityConfig:
    """Class to store the projection intensity configuration.

    Attributes:
        morphology_dir: Path to the directory containing the input morphologies.
        grid_corner: The first corner of the grid to use.
        grid_voxel_sizes: The voxel sizes of the grid.
        output_dir: The directory to which the results will be exported.
        figure_dir: The directory to which the figures will be exported.
    """

    morphology_dir: FileType = field(converter=Path)
    # grid_corner: tuple[float, float, float] | None = field(
    #     default=None,
    #     converter=converters.optional(to_tuple),
    #     validator=validators.optional(
    #         validators.deep_iterable(
    #             member_validator=validators.instance_of(float),
    #             iterable_validator=validators.instance_of(tuple),
    #         )
    #     ),
    # )
    grid_voxel_sizes: list[tuple[float, float, float]] = field(
        converter=converters.optional(to_tuple),
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.deep_iterable(
                    validators.instance_of((int, float)),
                    iterable_validator=validators.instance_of(tuple),
                ),
                # iterable_validator=validators.instance_of(list)
            )
        ),
    )

    output_dir: FileType = field(converter=converters.optional(Path))
    figure_dir: FileType | None = field(default=None, converter=converters.optional(Path))


def get_morphologies(morphology_dir) -> pd.DataFrame:
    """Create a DataFrame from a directory containing morphologies."""
    morphology_dir = Path(morphology_dir)
    morph_files = [i for i in morphology_dir.iterdir() if is_morphology(i)]

    morph_names = [i.stem for i in morph_files]
    morph_files = [str(i) for i in morph_files]
    if not morph_files:
        msg = f"No morphology file found in '{morphology_dir}'"
        raise RuntimeError(msg)

    cells_df = pd.DataFrame({"morphology": morph_names, "morph_file": morph_files})
    return cells_df.sort_values("morphology", ignore_index=True)


def segment_voxel_intersections(row, grid, *, return_sub_segments=False):
    """Get indices and intersection lengths of the voxels intersected by the given segment."""
    start_pt = [row[FROM_COORDS_COLS.X], row[FROM_COORDS_COLS.Y], row[FROM_COORDS_COLS.Z]]
    end_pt = [row[TO_COORDS_COLS.X], row[TO_COORDS_COLS.Y], row[TO_COORDS_COLS.Z]]
    indices, sub_segments = voxel_intersection(
        [start_pt, end_pt],
        grid,
        return_sub_segments=True,
    )
    res = {
        "indices": indices,
        "lengths": np.linalg.norm(sub_segments[:, [3, 4, 5]] - sub_segments[:, [0, 1, 2]], axis=1),
    }
    if return_sub_segments:
        res["sub_segments"] = sub_segments
    return pd.Series(res)


def segments_proj_intensities(segments, bbox, voxel_sizes, center, logger=None):
    """Compute the projection intensities of the given segments."""
    shape = ((bbox[1] - bbox[0]) // voxel_sizes) + 3
    if logger is not None:
        logger.debug(
            "Create grid with size=%s, voxel_dimensions=%s and offset=%s",
            shape.astype(int),
            voxel_sizes,
            bbox[0],
        )
    grid = VoxelData(np.zeros(shape.astype(int)), voxel_sizes, offset=bbox[0])

    # Ensure the center is located at the center of a voxel
    grid.offset -= (
        1.5 - np.modf(grid.positions_to_indices(center, keep_fraction=True))[0]
    ) * voxel_sizes

    # Compute intersections
    intersections = segments.apply(segment_voxel_intersections, args=(grid,), axis=1)

    elements = pd.DataFrame(
        {
            "indices": intersections["indices"].explode(),
            "lengths": intersections["lengths"].explode(),
        }
    )
    elements["indices"] = elements["indices"].apply(lambda x: tuple(x))

    lengths = elements.groupby("indices")["lengths"].sum().reset_index()
    indices = tuple(np.vstack(lengths["indices"].to_numpy()).T.tolist())

    heat_map = VoxelData(grid.raw.copy(), grid.voxel_dimensions.copy())
    heat_map.raw[indices] += lengths["lengths"].astype(float).to_numpy()
    return heat_map


def proj_intensities_one_morph(morph_data, config):
    """Compute the projection intensities from of a given morphology file."""
    morph_name = morph_data.get("morphology", "NO MORPH NAME FOUND")
    morph_custom_logger = MorphNameAdapter(LOGGER, extra={"morph_name": morph_name})
    try:
        # Load the morphology
        morph = load_morphology(morph_data["morph_file"])
        file_paths = []

        for num, axon in enumerate(get_axons(morph)):
            morph_custom_logger.debug("Processing axon %s", num)
            bbox = compute_bbox(axon.points[:, COLS.XYZ])
            center = axon.root_node.points[0, COLS.XYZ]

            # Create the DF of segments
            nodes, edges = neurite_to_pts(axon, keep_section_segments=True)

            edges = edges.merge(nodes, left_on="source", right_index=True).rename(
                columns={
                    COORDS_COLS.X: FROM_COORDS_COLS.X,
                    COORDS_COLS.Y: FROM_COORDS_COLS.Y,
                    COORDS_COLS.Z: FROM_COORDS_COLS.Z,
                }
            )
            edges = edges.merge(
                nodes, left_on="target", right_index=True, suffixes=("_from", "_to")
            ).rename(
                columns={
                    COORDS_COLS.X: TO_COORDS_COLS.X,
                    COORDS_COLS.Y: TO_COORDS_COLS.Y,
                    COORDS_COLS.Z: TO_COORDS_COLS.Z,
                }
            )
            edges["length"] = np.linalg.norm(
                edges[FROM_COORDS_COLS].to_numpy() - edges[TO_COORDS_COLS].to_numpy(),
                axis=1,
            )

            for voxel_sizes in config.grid_voxel_sizes:
                # Create the grid
                heat_map = segments_proj_intensities(edges, bbox, voxel_sizes, center)

                # Export the result
                filename = (
                    config.output_dir
                    / f"{morph_name}_{'_'.join([str(i) for i in voxel_sizes])}.nrrd"
                )
                heat_map.save_nrrd(filename)
                file_paths.append(filename)

                # Export the figure
                # if config.figure_dir is not None:
                #     plot_heat_map(
                #         heat_map,
                #         (config.figure_dir / filename.name).with_suffix(".html"),
                #     )

        res = {
            "file_paths": file_paths,
            "debug_infos": None,
        }
    except Exception as exc:
        morph_custom_logger.exception(
            "Skip the morphology because of the following error:",
        )
        res = {
            "file_paths": None,
            "debug_infos": str(exc),
        }
    return pd.Series(res, dtype=object)


def compute_projection_intensities(
    config: ProjectionIntensityConfig,
    *,
    parallel_config: ParallelConfig | None = None,
):
    """Compute projection intensities of the given morphologies."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.figure_dir is not None:
        config.figure_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over morphologies
    morphologies = get_morphologies(config.morphology_dir)

    # Initialize Dask cluster
    with disable_loggers("asyncio", "distributed", "distributed.worker"):
        if parallel_config.nb_processes > 1:
            LOGGER.info("Start parallel computation using %s workers", parallel_config.nb_processes)
            cluster = LocalCluster(n_workers=parallel_config.nb_processes, timeout="60s")
            parallel_factory = init_parallel_factory("dask_dataframe", address=cluster)
        else:
            LOGGER.info("Start computation")
            parallel_factory = init_parallel_factory(None)

        # Compute region indices of each segment
        results = evaluate(
            morphologies,
            proj_intensities_one_morph,
            [
                ["file_paths", None],
                ["debug_infos", None],
            ],
            parallel_factory=parallel_factory,
            progress_bar=parallel_config.progress_bar,
            func_args=[config],
        )

        # Close the Dask cluster if opened
        if parallel_config.nb_processes > 1:
            parallel_factory.shutdown()
            cluster.close()

    return results
