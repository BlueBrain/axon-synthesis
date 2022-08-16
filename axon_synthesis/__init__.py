"""AxonSynthesis package."""
import luigi
import numpy as np
import pkg_resources

__version__ = pkg_resources.get_distribution("AxonSynthesis").version


def seed_param(desc=None):
    """Create a numerical parameter used for seeding random number generators."""
    if desc is None:
        desc = "The seed used by the random number generator."
    return luigi.NumericalParameter(
        description=desc,
        var_type=int,
        default=0,
        min_value=0,
        max_value=int(np.iinfo(np.int).max),
    )
