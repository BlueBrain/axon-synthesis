"""Module with tools to compute the projection intensity."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

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

from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.synthesis import ParallelConfig
from axon_synthesis.typing import FileType
from axon_synthesis.utils import MorphNameAdapter
from axon_synthesis.utils import compute_bbox
from axon_synthesis.utils import disable_distributed_loggers
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import load_morphology
from axon_synthesis.utils import neurite_to_pts
from axon_synthesis.validation.utils import segment_intersection_lengths

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
            _nodes, edges = neurite_to_pts(axon, keep_section_segments=True, edges_with_coords=True)
            edges["length"] = np.linalg.norm(
                edges[FROM_COORDS_COLS].to_numpy() - edges[TO_COORDS_COLS].to_numpy(),
                axis=1,
            )

            for voxel_dimensions in config.grid_voxel_dimensions:
                # Create the grid
                heat_map = segment_intersection_lengths(edges, bbox, voxel_dimensions, center)

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
    with disable_distributed_loggers():
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


def resize_volume(volume, target_bbox):
    """Zero padding to align the volume to the target grid."""
    min_pads = np.round(np.abs(volume.bbox[0] - target_bbox[0]) / volume.voxel_dimensions)
    max_pads = np.round(np.abs(volume.bbox[1] - target_bbox[1]) / volume.voxel_dimensions)
    volume.raw = np.pad(volume.raw, np.vstack([min_pads, max_pads]).T.astype(int), "constant")
    volume.offset -= min_pads * volume.voxel_dimensions


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
            resize_volume(i, target_bbox)

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


def all_diff_stats(data):
    """Compute basic stats of the absolute difference."""
    diff = VoxelData.load_nrrd(data.loc["file_path_diff"])
    ref = VoxelData.load_nrrd(data.loc["file_path_ref"])
    comp = VoxelData.load_nrrd(data.loc["file_path_comp"])

    # mask = np.where(ref.raw != np.nan)

    diff_masked = diff.raw  # [mask]
    ref_masked = ref.raw  # [mask]
    comp_masked = comp.raw  # [mask]

    total = np.abs(ref_masked).sum() + np.abs(comp_masked).sum()
    total_square = np.square(ref_masked).sum() + np.square(comp_masked).sum()

    return {
        "L1": np.abs(diff_masked).sum() / total,
        "L2": np.sqrt(np.square(diff_masked).sum() / total_square),
        "sum": np.abs(diff_masked).sum(),
        "mean": np.abs(diff_masked).mean(),
        "min": np.abs(diff_masked).min(),
        "max": np.abs(diff_masked).max(),
        "std": np.abs(diff_masked).std(),
    }


def diff_stats(diff_dir: FileType):
    """Compute simple statistics of projection intensity differences."""
    data = pd.read_hdf(Path(diff_dir) / "metadata.h5", key="projection_intensity_differences")
    stats = data.apply(all_diff_stats, axis=1).apply(pd.Series)
    return data.join(stats)


def plot_diff_stats(
    data,
    *,
    output_path: FileType | None = None,
    stat_type="L1",
    log_x=True,
    log_y=False,
    show=False,
):
    """Plot the projection intensity differences for the given data."""
    data = data.copy(deep=False)
    data["voxel_size"] = data["voxel_dimensions"].apply(lambda row: row[0])
    data["axon"] = data.apply(lambda row: row["morphology"] + "_" + str(row["axon_id"]), axis=1)
    fig = px.line(data, x="voxel_size", y=stat_type, color="axon", log_x=log_x, log_y=log_y)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path, auto_open=show)
    return fig
