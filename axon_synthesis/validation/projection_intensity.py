"""Module with tools to compute the projection intensity."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
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
    return [tuple(i) for i in new_x]


@define
class ProjectionIntensityConfig:
    """Class to store the projection intensity configuration.

    Attributes:
        morphology_dir: Path to the directory containing the input morphologies.
        grid_corner: The first corner of the grid to use.
        grid_voxel_dimensions: The voxel sizes of the grid.
        output_dir: The directory to which the results will be exported.
        figure_dir: The directory to which the figures will be exported.
    """

    morphology_dir: FileType = field(converter=Path)
    grid_voxel_dimensions: list[tuple[float, float, float]] = field(
        converter=to_tuple,
        validator=validators.deep_iterable(
            member_validator=validators.deep_iterable(
                validators.instance_of((int, float)),
                iterable_validator=validators.instance_of(tuple),
            ),
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


def segments_proj_intensities(segments, bbox, voxel_dimensions, center, logger=None):
    """Compute the projection intensities of the given segments."""
    shape = np.clip((bbox[1] - bbox[0]) // voxel_dimensions, 1, np.inf) + 3
    if logger is not None:
        logger.debug(
            "Create grid with size=%s, voxel_dimensions=%s and offset=%s",
            shape.astype(int),
            voxel_dimensions,
            bbox[0],
        )
    grid = VoxelData(np.zeros(shape.astype(int)), voxel_dimensions, offset=bbox[0])

    # Ensure the center is located at the center of a voxel
    grid.offset -= (
        1.5 - np.modf(grid.positions_to_indices(center, keep_fraction=True))[0]
    ) * voxel_dimensions

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

    grid.raw[indices] += lengths["lengths"].astype(float).to_numpy()
    return grid


def proj_intensities_one_morph(morph_data, config):
    """Compute the projection intensities from of a given morphology file."""
    morph_name = morph_data.get("morphology", "NO MORPH NAME FOUND")
    morph_custom_logger = MorphNameAdapter(LOGGER, extra={"morph_name": morph_name})
    file_paths = []
    grid_voxel_dimensions = []
    axon_ids = []
    try:
        # Load the morphology
        morph = load_morphology(morph_data["morph_file"])

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

            for voxel_dimensions in config.grid_voxel_dimensions:
                # Create the grid
                heat_map = segments_proj_intensities(edges, bbox, voxel_dimensions, center)

                # Export the result
                filename = str(
                    config.output_dir
                    / f"{morph_name}_{num}-{'_'.join([str(i) for i in voxel_dimensions])}.nrrd"
                )
                heat_map.save_nrrd(filename)
                file_paths.append(filename)
                grid_voxel_dimensions.append(voxel_dimensions)
                axon_ids.append(num)

                # Export the figure
                # if config.figure_dir is not None:
                #     plot_heat_map(
                #         heat_map,
                #         (config.figure_dir / filename.name).with_suffix(".html"),
                #     )

    except Exception:
        morph_custom_logger.exception(
            "Skip the morphology because of the following error:",
        )
        raise
    else:
        return {
            "file_paths": file_paths,
            "voxel_dimensions": grid_voxel_dimensions,
            "axon_ids": axon_ids,
        }


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

    # Initialize parallel computation
    if parallel_config is None:
        parallel_config = ParallelConfig(0)
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
                ["voxel_dimensions", None],
                ["axon_ids", None],
            ],
            parallel_factory=parallel_factory,
            progress_bar=parallel_config.progress_bar,
            func_args=[config],
        )

        # Close the Dask cluster if opened
        if parallel_config.nb_processes > 1:
            parallel_factory.shutdown()
            cluster.close()

        results = (
            results.explode(["file_paths", "voxel_dimensions", "axon_ids"])
            .rename(columns={"file_paths": "file_path", "axon_ids": "axon_id"})
            .reset_index(drop=True)
        )
        results.to_hdf(config.output_dir / "files.h5", key="projection_intensity_files")

    return results


def projection_intensities_diff(data):
    """Compare two projection intensities."""
    ref_file = data.loc["file_path_ref"]
    comp_file = data.loc["file_path_comp"]
    voxel_dimensions = data.loc["voxel_dimensions"]

    ref_data = VoxelData.load_nrrd(ref_file)
    comp_data = VoxelData.load_nrrd(comp_file)

    if (
        not np.isclose(ref_data.voxel_dimensions, voxel_dimensions).all()
        or not np.isclose(comp_data.voxel_dimensions, voxel_dimensions).all()
    ):
        msg = (
            f"Inconsistent voxel dimensions: {list(voxel_dimensions)} in data, "
            f"{list(ref_data.voxel_dimensions)} in reference file and "
            f"{list(comp_data.voxel_dimensions)} in compared file"
        )
        raise ValueError(msg)
    if not np.isclose(ref_data.offset, comp_data.offset).all() or ref_data.shape != comp_data.shape:
        LOGGER.debug("Resize data to overlap properly for %s", data.loc["morphology"])
        target_bbox = compute_bbox(np.vstack([ref_data.bbox, comp_data.bbox]))

        for i in [ref_data, comp_data]:
            # Zero padding to align the grids
            min_pads = np.round(np.abs(i.bbox[0] - target_bbox[0]) / voxel_dimensions)
            max_pads = np.round(np.abs(i.bbox[1] - target_bbox[1]) / voxel_dimensions)
            i.raw = np.pad(i.raw, np.vstack([min_pads, max_pads]).T.astype(int), "constant")
            i.offset -= min_pads * voxel_dimensions

    diff = ref_data.with_data(ref_data.raw - comp_data.raw)

    if "file_path_diff" in data:
        diff.save_nrrd(data.loc["file_path_diff"])


def compute_projection_intensities_differences(
    ref_config: ProjectionIntensityConfig,
    compared_config: ProjectionIntensityConfig,
    output_dir: FileType,
    *,
    parallel_config: ParallelConfig | None = None,
    overwrite: bool = False,
):
    """Compute and compare the projection intensities from 2 sets of morphologies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute of load the projection intensities
    if overwrite or not ref_config.output_dir.exists():
        ref = compute_projection_intensities(ref_config, parallel_config=parallel_config)
    else:
        ref = pd.read_hdf(ref_config.output_dir / "files.h5", key="projection_intensity_files")
    if overwrite or not compared_config.output_dir.exists():
        comp = compute_projection_intensities(compared_config, parallel_config=parallel_config)
    else:
        comp = pd.read_hdf(
            compared_config.output_dir / "files.h5", key="projection_intensity_files"
        )

    # Merge the data
    data = ref.merge(
        comp,
        on=["morphology", "axon_id", "voxel_dimensions"],
        suffixes=("_ref", "_comp"),
        how="outer",
        indicator=True,
    )

    # Check for inconsistent or missing data
    matching_data = data.loc[data["_merge"] == "both"].drop(columns=["_merge"])
    missing = data.loc[data["_merge"] != "both"].copy(deep=False)
    if len(missing) > 0:
        missing["found"] = (
            missing["_merge"]
            .str.replace("left_only", "in reference only")
            .str.replace("right_only", "in compared only")
        )
        LOGGER.warning(
            "Could not find corresponding entries for the following: %s",
            missing.loc[:, ["morphology", "axon_id", "voxel_dimensions", "found"]].to_dict(
                "records"
            ),
        )

    # Build the output paths
    matching_data["file_path_diff"] = matching_data.apply(
        lambda x: str(
            output_dir
            / (
                f"{x['morphology']}_{x['axon_id']}-"
                f"{'_'.join([str(i) for i in x['voxel_dimensions']])}_diff.nrrd"
            )
        ),
        axis=1,
    )

    # Export the metadata
    matching_data.to_hdf(output_dir / "metadata.h5", key="projection_intensity_differences")

    # Compute and export the diff
    matching_data.apply(projection_intensities_diff, axis=1)

    return matching_data


def abs_diff_stats(data):
    """Compute basic stats of the absolute difference."""
    diff = VoxelData.load_nrrd(data.loc["file_path_diff"])
    ref = VoxelData.load_nrrd(data.loc["file_path_ref"])

    mask = np.where(ref.raw != np.nan)

    diff_masked = diff.raw[mask]
    ref_masked = ref.raw[mask]

    return {
        "L1": np.abs(diff_masked).sum() / np.abs(ref_masked).sum(),
        "L2": np.sqrt(np.square(diff_masked).sum() / np.square(ref_masked).sum()),
        "sum": np.abs(diff_masked).sum(),
        "mean": np.abs(diff_masked).mean(),
        "min": np.abs(diff_masked).min(),
        "max": np.abs(diff_masked).max(),
        "std": np.abs(diff_masked).std(),
    }


def diff_stats(diff_dir: FileType):
    """Compute simple statistics of projection intensity differences."""
    data = pd.read_hdf(Path(diff_dir) / "metadata.h5", key="projection_intensity_differences")
    stats = data.apply(abs_diff_stats, axis=1).apply(pd.Series)
    return data.join(stats)


def plot_diff_stats(
    data, output_path: FileType, *, stat_type="L1", log_x=True, log_y=False, show=False
):
    """Plot the projection intensity differences for the given data."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    data = data.copy(deep=False)
    data["voxel_size"] = data["voxel_dimensions"].apply(lambda row: row[0])
    data["axon"] = data.apply(lambda row: row["morphology"] + "_" + str(row["axon_id"]), axis=1)
    fig = px.line(data, x="voxel_size", y=stat_type, color="axon", log_x=log_x, log_y=log_y)
    fig.write_html(output_path, auto_open=show)
