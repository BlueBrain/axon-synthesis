"""Test the axon_synthesis.geometry module."""
import numpy as np
from voxcell.voxel_data import VoxelData

from axon_synthesis.geometry import voxel_intersection


def test_voxel_intersection():
    """Tests for the voxel_intersection function."""
    # pylint: disable=too-many-statements

    test_brain_regions = VoxelData(np.ones((3, 4, 5)), [1, 1, 1], [0, 0, 0])

    # The segment does not intersect any voxel
    indices, sub_segments = voxel_intersection(
        [[-1, -1, -1], [-2, -2, -2]], test_brain_regions, return_sub_segments=True
    )
    assert indices.size == 0
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [-1, -1, -1, -2, -2, -2],
        ],
    )

    # The segment intersects several voxels
    indices, sub_segments = voxel_intersection(
        [[1.5, 0, 3], [0, 2.5, 0]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(
        indices,
        [
            [1, 0, 3],
            [1, 0, 2],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [0, 2, 0],
        ],
    )
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [1.5, 0.0, 3.0, 1.5, 0.0, 3.0],
            [1.5, 0.0, 3.0, 1.0, 0.83333333, 2.0],
            [1.0, 0.83333333, 2.0, 0.9, 1.0, 1.8],
            [0.9, 1.0, 1.8, 0.5, 1.66666667, 1.0],
            [0.5, 1.66666667, 1.0, 0.3, 2.0, 0.6],
            [0.3, 2.0, 0.6, 0.0, 2.5, 0.0],
        ],
    )

    # The segment is entirely inside the voxel
    indices, sub_segments = voxel_intersection(
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        test_brain_regions,
        return_sub_segments=True,
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.25, 0.25, 0.25, 0.75, 0.75, 0.75],
        ],
    )

    # The segment touches the xmin plane and is inside the first voxel
    indices, sub_segments = voxel_intersection(
        [[0, 0.5, 0.5], [0.5, 0.5, 0.5]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
        ],
    )

    # The segment is contained in the xmin plane
    indices, sub_segments = voxel_intersection(
        [[1, 0.25, 0.25], [1, 0.75, 0.75]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[1, 0, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [1, 0.25, 0.25, 1, 0.75, 0.75],
        ],
    )

    # The segment touches the xmin plane and is inside the second voxel
    indices, sub_segments = voxel_intersection(
        [[1, 0.5, 0.5], [1.5, 0.5, 0.5]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[1, 0, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [1.0, 0.5, 0.5, 1.5, 0.5, 0.5],
        ],
    )

    # The segment touches the xmax plane and is inside the voxel
    indices, sub_segments = voxel_intersection(
        [[0.5, 0.5, 0.5], [1, 0.5, 0.5]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.5, 0.5, 0.5, 1, 0.5, 0.5],
            [1, 0.5, 0.5, 1, 0.5, 0.5],
        ],
    )

    # The segment touches the ymin plane and is inside the first voxel
    indices, sub_segments = voxel_intersection(
        [[0.5, 0, 0.5], [0.5, 0.5, 0.5]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.5, 0.0, 0.5, 0.5, 0.5, 0.5],
        ],
    )

    # The segment touches the ymin plane and is inside the second voxel
    indices, sub_segments = voxel_intersection(
        [[0.5, 1, 0.5], [0.5, 1.5, 0.5]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 1, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.5, 1.0, 0.5, 0.5, 1.5, 0.5],
        ],
    )

    # The segment touches the ymax plane and is inside the voxel
    indices, sub_segments = voxel_intersection(
        [[0.5, 0.5, 0.5], [0.5, 1, 0.5]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0], [0, 1, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.5, 0.5, 0.5, 0.5, 1.0, 0.5],
            [0.5, 1.0, 0.5, 0.5, 1.0, 0.5],
        ],
    )

    # The segment is contained in the ymin plane
    indices, sub_segments = voxel_intersection(
        [[0.25, 1, 0.25], [0.75, 1, 0.75]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 1, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.25, 1.0, 0.25, 0.75, 1.0, 0.75],
        ],
    )

    # The segment touches the zmin plane and is inside the first voxel
    indices, sub_segments = voxel_intersection(
        [[0.5, 0.5, 0], [0.5, 0.5, 0.5]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.5, 0.5, 0.0, 0.5, 0.5, 0.5],
        ],
    )

    # The segment touches the zmin plane and is inside the second voxel
    indices, sub_segments = voxel_intersection(
        [[0.5, 0.5, 1], [0.5, 0.5, 1.5]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 0, 1]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.5, 0.5, 1.0, 0.5, 0.5, 1.5],
        ],
    )

    # The segment touches the zmax plane and is inside the voxel
    indices, sub_segments = voxel_intersection(
        [[0.5, 0.5, 0.5], [0.5, 0.5, 1]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0], [0, 0, 1]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 0.5, 0.5, 1.0],
        ],
    )

    # The segment is contained in the zmin plane
    indices, sub_segments = voxel_intersection(
        [[0.25, 0.25, 1], [0.75, 0.75, 1]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 0, 1]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.25, 0.25, 1.0, 0.75, 0.75, 1.0],
        ],
    )

    # The segment touches the corner of several voxels and is inside one voxel
    indices, sub_segments = voxel_intersection(
        [[0.25, 0.25, 0.25], [1, 1, 1]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0], [1, 1, 1]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.25, 0.25, 0.25, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    )

    # The segment crosses several voxels along X
    indices, sub_segments = voxel_intersection(
        [[0.25, 0.25, 0.25], [2.25, 0.25, 0.25]],
        test_brain_regions,
        return_sub_segments=True,
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.25, 0.25, 0.25, 1.0, 0.25, 0.25],
            [1.0, 0.25, 0.25, 2.0, 0.25, 0.25],
            [2.0, 0.25, 0.25, 2.25, 0.25, 0.25],
        ],
    )

    # The segment crosses several voxels along Y
    indices, sub_segments = voxel_intersection(
        [[0.25, 0.25, 0.25], [0.25, 2.25, 0.25]],
        test_brain_regions,
        return_sub_segments=True,
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0], [0, 1, 0], [0, 2, 0]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.25, 0.25, 0.25, 0.25, 1.0, 0.25],
            [0.25, 1.0, 0.25, 0.25, 2.0, 0.25],
            [0.25, 2.0, 0.25, 0.25, 2.25, 0.25],
        ],
    )

    # The segment crosses several voxels along Z
    indices, sub_segments = voxel_intersection(
        [[0.25, 0.25, 0.25], [0.25, 0.25, 2.25]],
        test_brain_regions,
        return_sub_segments=True,
    )
    np.testing.assert_array_equal(indices, [[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.25, 0.25, 0.25, 0.25, 0.25, 1.0],
            [0.25, 0.25, 1.0, 0.25, 0.25, 2.0],
            [0.25, 0.25, 2.0, 0.25, 0.25, 2.25],
        ],
    )

    # The segment touches the boundaries of several voxels along X
    indices, sub_segments = voxel_intersection(
        [[0, 1, 1], [3, 1, 1]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(
        indices,
        [
            [0, 1, 1],
            [1, 1, 1],
            [2, 1, 1],
        ],
    )
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 3.0, 1.0, 1.0],
        ],
    )

    # The segment touches the boundaries of several voxels along X and is reversed
    indices, sub_segments = voxel_intersection(
        [[3, 1, 1], [0, 1, 1]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(
        indices,
        [
            [2, 1, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
    )
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [3.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        ],
    )

    # The segment touches the boundaries of several voxels along Y
    indices, sub_segments = voxel_intersection(
        [[1, 0, 1], [1, 3, 1]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(
        indices,
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 2, 1],
            [1, 3, 1],  # The data is longer along Y than along X
        ],
    )
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0, 1.0, 3.0, 1.0],
            [1.0, 3.0, 1.0, 1.0, 3.0, 1.0],
        ],
    )

    # The segment touches the boundaries of several voxels along Y and is reversed
    indices, sub_segments = voxel_intersection(
        [[1, 3, 1], [1, 0, 1]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(
        indices,
        [
            [1, 3, 1],  # The data is longer along Y than along X
            [1, 2, 1],
            [1, 1, 1],
            [1, 0, 1],
        ],
    )
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [1.0, 3.0, 1.0, 1.0, 3.0, 1.0],
            [1.0, 3.0, 1.0, 1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        ],
    )

    # The segment touches the boundaries of several voxels along Z
    indices, sub_segments = voxel_intersection(
        [[1, 1, 0], [1, 1, 3]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(
        indices,
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 3],  # The data is longer along Z than along X
        ],
    )
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 1.0, 1.0, 3.0],
            [1.0, 1.0, 3.0, 1.0, 1.0, 3.0],
        ],
    )

    # The segment touches the boundaries of several voxels along Z and is reversed
    indices, sub_segments = voxel_intersection(
        [[1, 1, 3], [1, 1, 0]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(
        indices,
        [
            [1, 1, 3],  # The data is longer along Z than along X
            [1, 1, 2],
            [1, 1, 1],
            [1, 1, 0],
        ],
    )
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [1.0, 1.0, 3.0, 1.0, 1.0, 3.0],
            [1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ],
    )

    # The segment is oblique and passes only in the upper right corner of the voxel
    indices, sub_segments = voxel_intersection(
        [[1.5, 1.5, 1.5], [0.5, 0.5, 0.5]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(
        indices,
        [
            [1, 1, 1],
            [0, 0, 0],
        ],
    )
    np.testing.assert_array_almost_equal(
        sub_segments,
        [
            [1.5, 1.5, 1.5, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
        ],
    )

    # Test that the returned indices can be used to get the data
    indices, sub_segments = voxel_intersection(
        [[1, 1, 1], [3, 1, 1]], test_brain_regions, return_sub_segments=True
    )
    np.testing.assert_array_equal(test_brain_regions.raw[tuple(indices.T.tolist())], [1, 1])


if __name__ == "__main__":
    test_voxel_intersection()
