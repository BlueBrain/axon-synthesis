"""Tests for the post-processing functions."""
import numpy as np

from axon_synthesis.PCSF import post_process


class TestRandomWalk:
    """Test random walk functions."""

    def test_simple(self):
        """Test that the random walk works properly."""
        starting_pt = np.array([0, 0, 0])
        intermediate_pts = np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 2, 0],
            ]
        )
        length_stats = {
            "norm": 1,
            "std": 0.25,
        }
        angle_stats = {
            "norm": 30,
            "std": 10,
        }
        rng = np.random.default_rng(0)
        post_process.random_walk(
            starting_pt,
            intermediate_pts,
            length_stats,
            angle_stats,
            history_path_length=5,
            previous_history=None,
            global_target_coeff=0.5,
            target_coeff=2,
            random_coeff=3.5,
            history_coeff=2,
            rng=rng,
        )
