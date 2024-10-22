"""Tests for the clustering processing functions."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from axon_synthesis.inputs import clustering


class TestFromBrainRegion:
    """Test functions from `axon_synthesis.main_trunk.clustering.from_brain_regions`."""

    @pytest.mark.parametrize(
        ("regions", "sub_segments", "expected_regions", "expected_sub_segments"),
        [
            pytest.param(
                [1, 2],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 1, 0, 0]],
                [1, 2],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 1, 0, 0]],
                id="two_single_regions",
            ),
            pytest.param(
                [1, 1, 2],
                [[0, 0, 0, 0.25, 0, 0], [0.25, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 1, 0, 0]],
                [1, 2],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 1, 0, 0]],
                id="two_regions_with_first_duplicated",
            ),
            pytest.param(
                [1, 2, 2],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 0.75, 0, 0], [0.75, 0, 0, 1, 0, 0]],
                [1, 2],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 1, 0, 0]],
                id="two_regions_with_second_duplicated",
            ),
            pytest.param(
                [1, 1, 2, 2],
                [
                    [0, 0, 0, 0.25, 0, 0],
                    [0.25, 0, 0, 0.5, 0, 0],
                    [0.5, 0, 0, 0.75, 0, 0],
                    [0.75, 0, 0, 1, 0, 0],
                ],
                [1, 2],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 1, 0, 0]],
                id="two_regions_with_both_duplicated",
            ),
            pytest.param(
                [1, 2, 3],
                [
                    [0, 0, 0, 0.5, 0, 0],
                    [0.5, 0, 0, 0.75, 0, 0],
                    [0.75, 0, 0, 1, 0, 0],
                ],
                [1, 2, 3],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 0.75, 0, 0], [0.75, 0, 0, 1, 0, 0]],
                id="three_single_regions",
            ),
            pytest.param(
                [1, 1, 2, 3],
                [
                    [0, 0, 0, 0.25, 0, 0],
                    [0.25, 0, 0, 0.5, 0, 0],
                    [0.5, 0, 0, 0.75, 0, 0],
                    [0.75, 0, 0, 1, 0, 0],
                ],
                [1, 2, 3],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 0.75, 0, 0], [0.75, 0, 0, 1, 0, 0]],
                id="three_regions_first_duplicated",
            ),
            pytest.param(
                [1, 1, 2, 3],
                [
                    [0, 0, 0, 0.25, 0, 0],
                    [0.25, 0, 0, 0.5, 0, 0],
                    [0.5, 0, 0, 0.5, 0, 0],
                    [0.5, 0, 0, 1, 0, 0],
                ],
                [1, 3],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 1, 0, 0]],
                id="three_regions_first_duplicated_second_0_length",
            ),
            pytest.param(
                [1, 1],
                [[0, 0, 0, 0.5, 0, 0], [0.5, 0, 0, 1, 0, 0]],
                [1],
                [[0, 0, 0, 1, 0, 0]],
                id="one_duplicated_region",
            ),
            pytest.param(
                [1, 1],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [],
                [],
                id="one_duplicated_region_0_length",
            ),
            pytest.param(
                [],
                [],
                [],
                [],
                id="empty_region_list",
            ),
            pytest.param(
                [1],
                [[0, 0, 0, 1, 0, 0]],
                [1],
                [[0, 0, 0, 1, 0, 0]],
                id="one_single_region",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "reverse",
        [pytest.param(True, id="reversed"), pytest.param(False, id="regular")],
    )
    def test_merge_similar_regions(
        self,
        regions,
        sub_segments,
        expected_regions,
        expected_sub_segments,
        reverse,
    ):
        """Test that consecutive regions are properly merged."""
        regions = np.array(regions)
        sub_segments = np.array(sub_segments)

        # Prepare data for reversed case
        if reverse and len(regions) > 0:
            regions = regions[::-1]
            sub_segments = sub_segments[::-1, [3, 4, 5, 0, 1, 2]]
            expected_regions = expected_regions[::-1]
            if len(expected_sub_segments) > 0:
                expected_sub_segments = (
                    np.array(expected_sub_segments)[::-1, [3, 4, 5, 0, 1, 2]]
                ).tolist()

        # Call merge_similar_regions()
        res = clustering.from_brain_regions.merge_similar_regions(
            regions.tolist(),
            sub_segments.tolist(),
        )

        # Handle empty case
        if len(expected_sub_segments) == 0:
            expected_sub_segments = np.zeros_like([], shape=(0, 6), dtype=float)

        # Check results
        np.testing.assert_array_equal(res["brain_regions"], expected_regions)
        np.testing.assert_array_equal(res["sub_segments"], expected_sub_segments)

    class TestSegmentRegionIds:
        """Test that the segments are properly cut by brain regions."""

        def test_zero(self, brain_regions):
            """Test with 0-length segment."""
            seg = {
                "source": 0,
                "source_x": 0,
                "source_y": 0,
                "source_z": 0,
                "target": 1,
                "target_x": 0,
                "target_y": 0,
                "target_z": 0,
            }
            res = clustering.from_brain_regions.segment_region_ids(seg, brain_regions)
            assert res["brain_regions"].tolist() == []
            assert res["sub_segments"].tolist() == []

        def test_seg_in_one_region(self, brain_regions):
            """Test with a segment that is completely contained in a region."""
            seg = {
                "source": 0,
                "source_x": 0,
                "source_y": 0,
                "source_z": 0,
                "target": 1,
                "target_x": 10,
                "target_y": 20,
                "target_z": 0,
            }
            res = clustering.from_brain_regions.segment_region_ids(seg, brain_regions)
            np.testing.assert_array_equal(res["brain_regions"], [20])
            np.testing.assert_array_equal(res["sub_segments"], [[0, 0, 0, 10, 20, 0]])

        def test_seg_in_two_regions_one_voxel_both(self, brain_regions):
            """Test with a segment crossing 2 regions, but only one voxel in each."""
            seg = {
                "source": 0,
                "source_x": -10,
                "source_y": 0,
                "source_z": 0,
                "target": 1,
                "target_x": 10,
                "target_y": 0,
                "target_z": 0,
            }
            res = clustering.from_brain_regions.segment_region_ids(seg, brain_regions)
            np.testing.assert_array_equal(res["brain_regions"], [2, 20])
            np.testing.assert_array_equal(
                res["sub_segments"],
                [[-10, 0, 0, 0, 0, 0], [0, 0, 0, 10, 0, 0]],
            )

        def test_seg_in_two_regions_one_voxel_start(self, brain_regions):
            """Test with a segment crossing 2 regions, but only one voxel in start region."""
            seg = {
                "source": 0,
                "source_x": -10,
                "source_y": 0,
                "source_z": 0,
                "target": 1,
                "target_x": 500,
                "target_y": 0,
                "target_z": 0,
            }
            res = clustering.from_brain_regions.segment_region_ids(seg, brain_regions)
            np.testing.assert_array_equal(res["brain_regions"], [2, 20])
            np.testing.assert_array_equal(
                res["sub_segments"],
                [[-10, 0, 0, 0, 0, 0], [0, 0, 0, 500, 0, 0]],
            )

        def test_seg_in_two_regions_one_voxel_end(self, brain_regions):
            """Test with a segment crossing 2 regions, but only one voxel in end region."""
            seg = {
                "source": 0,
                "source_x": -500,
                "source_y": 0,
                "source_z": 0,
                "target": 1,
                "target_x": 10,
                "target_y": 0,
                "target_z": 0,
            }
            res = clustering.from_brain_regions.segment_region_ids(seg, brain_regions)
            np.testing.assert_array_equal(res["brain_regions"], [2, 20])
            np.testing.assert_array_equal(
                res["sub_segments"],
                [[-500, 0, 0, 0, 0, 0], [0, 0, 0, 10, 0, 0]],
            )

        def test_seg_in_two_regions_several_voxels(self, brain_regions):
            """Test with a segment crossing 2 regions on several voxel in each."""
            seg = {
                "source": 0,
                "source_x": -500,
                "source_y": 0,
                "source_z": 0,
                "target": 1,
                "target_x": 500,
                "target_y": 0,
                "target_z": 0,
            }
            res = clustering.from_brain_regions.segment_region_ids(seg, brain_regions)
            np.testing.assert_array_equal(res["brain_regions"], [2, 20])
            np.testing.assert_array_equal(
                res["sub_segments"],
                [[-500, 0, 0, 0, 0, 0], [0, 0, 0, 500, 0, 0]],
            )
