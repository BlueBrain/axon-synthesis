"""Tests for the post-processing functions."""
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from axon_synthesis.PCSF import post_process
from axon_synthesis.utils import use_matplotlib_backend


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
            ]
        )
        length_stats = {
            "norm": 0.25,
            "std": 0.01,
        }
        angle_stats = {
            "norm": 30,
            "std": 10,
        }
        rng = np.random.default_rng(0)

        points, (latest_lengths, latest_directions) = post_process.random_walk(
            start_pt,
            intermediate_pts,
            length_stats,
            angle_stats,
            history_path_length=None,
            previous_history=None,
            global_target_coeff=0,
            target_coeff=2,
            random_coeff=0,
            history_coeff=1.5,
            rng=rng,
            debug=True,
        )

        assert len(points) == 16
        npt.assert_array_equal(points[0], [0, 0, 0])
        npt.assert_array_equal(points[-1], [0, 2, 0])
        npt.assert_array_almost_equal(points[-2], [-6.51903e-2, 1.96985, 0])
        assert (points[:, 2] == 0).all()
        assert len(latest_lengths) == 4
        assert len(latest_directions) == 4

        if interactive_plots:
            with use_matplotlib_backend("QtAgg"):
                fig = plt.figure()
                ax = fig.gca(projection="3d")
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
                    ax.text(i[0], i[1], i[2], "%s" % (str(num)), size=20, zorder=1, color="k")
                ax.legend()
                plt.show()
