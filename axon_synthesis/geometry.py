"""Module for geometry functions."""
from typing import List
from typing import Tuple

try:
    from typing import Annotated

    Point3D = Annotated[List[float], 3]
    Seg3D = Annotated[List[Point3D], 2]
except ImportError:
    Point3D = List[float]
    Seg3D = Tuple[Point3D, Point3D]

import numpy as np
import pandas as pd


def voxel_intersection(seg, data, return_sub_segments=False):  # pylint: disable=too-many-locals
    """Find voxels intersected by a given segment and cut the segment according to these voxels.

    .. note::

        A point is considered as intersecting a voxel using the following rules:
            x_min <= x < x_max
            y_min <= y < y_max
            z_min <= z < z_max

        where x_min and x_max are the min and max coordinates along the X axis of the voxel, y_min
        and y_max are the same along the Y axis, and z_min and z_max are the same along the Z axis.

    .. warning::

        This function is being moved to voxcell: https://github.com/BlueBrain/voxcell/pull/5

    Args:
        seg: The segment with the following form: [[x_min, y_min, z_min], [x_max, y_max, z_max]].
        data: The VoxelData object.
        return_sub_segments: If est to `True`, the sub segments are also returned with the voxel
            indices.

    Returns:
        List of 3D indices.
        If `return_sub_segments` is set to `True`, the list of coordinates of the sub-segment
        points is also returned.
    """
    # If the segment is outside the bounding box, then it does not intersect any voxel.
    if (seg < data.bbox).all() or (seg >= data.bbox).all():
        seg_point_indices = np.zeros((0, 3), dtype=np.result_type(0))
        if return_sub_segments:
            return seg_point_indices, np.reshape(seg, (1, -1))
        return seg_point_indices

    # The segment is clipped inside the global bbox.
    cut_seg = np.clip(
        seg,
        a_min=data.bbox[0],
        a_max=np.nextafter(data.bbox[1], np.full_like(data.bbox[1], -1)),
    )

    # Compute the actual bbox of the segment
    bbox = np.sort(data.positions_to_indices(cut_seg), axis=0)

    # Unpack input data.
    start_pt, end_pt = cut_seg
    [start_x, start_y, start_z], [end_x, end_y, end_z] = cut_seg

    # Build the grid of all voxels included in the bbox.
    i_planes, j_planes, k_planes = [np.arange(bbox[0, i], bbox[1, i] + 1) for i in range(3)]
    sub_grid = np.array(np.meshgrid(i_planes, j_planes, k_planes)).T

    # Compute the boundary planes of each voxel.
    lower_left_corners = data.indices_to_positions(sub_grid)

    # Compute the vector of the segment.
    seg_vector = end_pt - start_pt

    def get_intersections(dst1, dst2, start_pt, seg_vector):
        """Compute intersection point."""
        same_sign = np.sign(dst1) == np.sign(dst2)
        coplanar = (dst1 == 0) & (dst2 == 0)
        denomimator = dst2 - dst1
        denomimator = np.where(denomimator == 0, np.nan, denomimator)
        f = np.where(same_sign | coplanar, np.nan, -dst1 / denomimator)

        # Multiply vector by factor.
        result = seg_vector * f[:, np.newaxis]

        # Return the hit position.
        return start_pt + result

    # Get the coordinates of the planes between voxels
    x_planes = lower_left_corners[0, :, 0, 0]
    y_planes = lower_left_corners[0, 0, :, 1]
    z_planes = lower_left_corners[:, 0, 0, 2]

    # Get the coordinates of the intersection points
    x_hits = get_intersections(start_x - x_planes, end_x - x_planes, start_pt, seg_vector)
    y_hits = get_intersections(start_y - y_planes, end_y - y_planes, start_pt, seg_vector)
    z_hits = get_intersections(start_z - z_planes, end_z - z_planes, start_pt, seg_vector)

    # Check how the points are ordered along each axis
    xyz_ascending = np.sign(end_pt - start_pt)
    xyz_ascending_sum = xyz_ascending.sum()

    # Build the sub-segment coordinate DF
    seg_points = np.vstack([x_hits, y_hits, z_hits])
    seg_points = seg_points[~np.isnan(seg_points).all(axis=1)]
    seg_points = np.unique(seg_points, axis=0)

    # Remove duplicated points when the extremities of the segment are on a voxel boundary, except
    # if they are in the ascending quadrant.
    if xyz_ascending_sum >= 0:
        seg_pt_start = (seg_points == start_pt).all(axis=1)
        if seg_pt_start.any():
            seg_points = seg_points[~seg_pt_start]
    else:
        seg_pt_end = (seg_points == end_pt).all(axis=1)
        if seg_pt_end.any():
            seg_points = seg_points[~seg_pt_end]

    # Build and sort the sub-segment points
    seg_points = np.vstack([start_pt, seg_points, end_pt])
    df_seg_points = pd.DataFrame(seg_points, columns=["x", "y", "z"])
    ascending = bool(
        xyz_ascending[0] == 1
        or (xyz_ascending[0] == 0 and xyz_ascending[1] == 1)
        or (xyz_ascending[0] == 0 and xyz_ascending[1] == 0 and xyz_ascending[2] == 1)
    )
    df_seg_points.sort_values(["x", "y", "z"], ascending=ascending, inplace=True)

    # Find the intersection indices
    seg_point_indices = data.positions_to_indices(
        df_seg_points.rolling(2, center=True, min_periods=2).mean().dropna().values
    )

    if return_sub_segments:
        sub_segments = np.hstack([df_seg_points.values[:-1], df_seg_points.values[1:]])
        return seg_point_indices, sub_segments

    return seg_point_indices
