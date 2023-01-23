"""AxonSynthesis package."""
import importlib.metadata

import luigi
import numpy as np

__version__ = importlib.metadata.version("axon-synthesis")


def seed_param(desc=None):
    """Create a numerical parameter used for seeding random number generators."""
    if desc is None:
        desc = "The seed used by the random number generator."
    return luigi.NumericalParameter(
        description=desc,
        var_type=int,
        default=0,
        min_value=0,
        max_value=int(np.iinfo(int).max),
    )
