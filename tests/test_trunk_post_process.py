"""Tests for the post-processing functions."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from axon_synthesis.synthesis.main_trunk import post_process

from . import use_matplotlib_backend


class TestRandomWalk:
    """Test random walk functions."""

    def test_simple(self, interactive_plots):
        """Test that the random walk works properly."""
        start_pt = np.array([0, 0, 0])
        intermediate_pts = np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 2, 0],
            ],
        )
        length_stats = {
            "norm": 0.05,
            "std": 0.01,
        }
        # angle_stats = {
        #     "norm": 30,
        #     "std": 10,
        # }
        rng = np.random.default_rng(0)

        config = post_process.PostProcessConfig(
            history_path_length=None,
            global_target_coeff=0,
            target_coeff=2,
            random_coeff=2,
            history_coeff=2,
        )

        points, (latest_lengths, latest_directions) = post_process.random_walk(
            start_pt,
            intermediate_pts,
            length_stats,
            # angle_stats,
            config=config,
            previous_history=([length_stats["norm"]] * 10, [[0, -1, 0]] * 10),
            rng=rng,
        )

        # ################################################################## #
        # Plot before testing the results
        if interactive_plots:
            with use_matplotlib_backend("QtAgg"):
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    label="Random walk",
                )
                ax.scatter(*start_pt, label="Start point")
                ax.scatter(
                    intermediate_pts[:, 0],
                    intermediate_pts[:, 1],
                    intermediate_pts[:, 2],
                    label="Intermediate targets",
                )
                for num, i in enumerate(intermediate_pts):
                    ax.text(i[0], i[1], i[2], str(num), size=20, zorder=1, color="k")  # type: ignore[arg-type]
                ax.legend()
                ax.set(xlim3d=(-0.5, 1.5), xlabel="X")
                ax.set(ylim3d=(-0.5, 1.5), ylabel="Y")
                ax.set(zlim3d=(-0.5, 1.5), zlabel="Z")
                plt.show()
        # ################################################################## #

        assert len(points) == 93
        npt.assert_array_equal(points[0], [0, 0, 0])
        npt.assert_array_equal(points[-1], [0, 2, 0])
        npt.assert_array_almost_equal(points[-2], [-1.3573e-2, 2.005857, 0.018934])
        assert len(latest_lengths) == 5
        assert len(latest_directions) == 5
