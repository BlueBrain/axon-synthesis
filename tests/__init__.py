"""Tests suite for the axon-synthesis package."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl

TEST_ROOT = Path(__file__).parent
EXAMPLES = TEST_ROOT.parent / "examples"
DATA = TEST_ROOT / "data"


class SharedData:
    """A class used to pass data from a test to another."""

    create_inputs_path = None


@contextmanager
def use_matplotlib_backend(new_backend):
    """A context manager to set a new temporary backend to matplotlib then restore the old one.

    Args:
        new_backend (str): The name of the backend to use in this context.
    """
    old_backend = mpl.get_backend()
    mpl.use(new_backend)
    try:
        yield
    finally:
        mpl.use(old_backend)
