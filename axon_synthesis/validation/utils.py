"""Module with some validation utils."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import numpy as np
import pandas as pd
from morph_tool.converter import convert
from voxcell import VoxelData
from voxcell.math_utils import voxel_intersection

from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.utils import disable_loggers
from axon_synthesis.utils import temp_dir


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


def segment_intersection_lengths(segments, bbox, voxel_dimensions, center=None, logger=None):
    """Compute the intersection lengths of the given segments with the given grid."""
    shape = np.clip((bbox[1] - bbox[0]) // voxel_dimensions, 1, np.inf).astype(int)
    if center is not None:
        shape += 3
    if logger is not None:
        logger.debug(
            "Create grid with size=%s, voxel_dimensions=%s and offset=%s",
            shape,
            voxel_dimensions,
            bbox[0],
        )
    grid = VoxelData(np.zeros(shape), voxel_dimensions, offset=bbox[0])

    # Ensure the center is located at the center of a voxel
    if center is not None:
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
    outside_segments = elements.loc[elements["indices"].isna()]
    if not outside_segments.empty:
        if logger is not None:
            logger.warning("Found %s segments outside the given grid", len(outside_segments))
        elements = elements.dropna(subset=["indices"])
    elements["indices"] = elements["indices"].apply(tuple)

    lengths = elements.groupby("indices")["lengths"].sum().reset_index()
    indices = tuple(np.vstack(lengths["indices"].to_numpy()).T.tolist())  # type: ignore[call-overload]

    grid.raw[indices] += lengths["lengths"].astype(float).to_numpy()
    return grid


def copy_morph_to_tmp_dir(morph):
    """Copy the given morphology in a temporary directory."""
    tmp_dir = temp_dir()
    if isinstance(morph, FileType):  # type: ignore[arg-type,misc]
        filename = Path(morph)
        name = filename.stem
        ext = filename.suffix
    else:
        name = morph.name
        ext = ".asc"
    filename = (Path(tmp_dir.name) / name).with_suffix(ext)
    with disable_loggers("morph_tool.converter"):
        convert(morph, filename, nrn_order=True)
    return tmp_dir, filename
