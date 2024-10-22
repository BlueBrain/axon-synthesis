"""Tests for the atlas processing functions."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of AxonSynthesis.
# See https://github.com/BlueBrain/AxonSynthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import pytest


@pytest.mark.xfail
def test_atlas(atlas):
    """Test the atlas."""
    raise RuntimeError(atlas)
