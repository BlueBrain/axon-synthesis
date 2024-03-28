"""Module with some validation utils."""
import numpy as np
import pandas as pd
from voxcell import VoxelData
from voxcell.math_utils import voxel_intersection

from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS


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


def segment_intersection_lengths(segments, bbox, voxel_dimensions, center, logger=None):
    """Compute the intersection lengths of the given segments with the given grid."""
    shape = (np.clip((bbox[1] - bbox[0]) // voxel_dimensions, 1, np.inf) + 3).astype(int)
    if logger is not None:
        logger.debug(
            "Create grid with size=%s, voxel_dimensions=%s and offset=%s",
            shape,
            voxel_dimensions,
            bbox[0],
        )
    grid = VoxelData(np.zeros(shape), voxel_dimensions, offset=bbox[0])

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
    elements["indices"] = elements["indices"].apply(tuple)

    lengths = elements.groupby("indices")["lengths"].sum().reset_index()
    indices = tuple(np.vstack(lengths["indices"].to_numpy()).T.tolist())

    grid.raw[indices] += lengths["lengths"].astype(float).to_numpy()
    return grid
