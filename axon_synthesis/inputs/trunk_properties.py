"""Compute the long-range trunk properties after clustering."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import logging
from functools import partial

import numpy as np
from neurom import COLS
from neurom import features
from neurom.apps import morph_stats
from neurom.core import Morphology

from axon_synthesis.utils import get_axons

logger = logging.getLogger(__name__)


def angle_between_vectors(p1, p2):
    """Computes the angle in radians between vectors 'p1' and 'p2'.

    Normalizes the input vectors and computes the relative angle
    between them.

        >>> angle_between((1, 0), (0, 1))
        1.5707963267948966
        >>> angle_between((1, 0), (1, 0))
        0.0
        >>> angle_between((1, 0), (-1, 0))
        3.141592653589793
    """
    if p1.shape == p2.shape and np.equal(p1, p2).all():
        return 0.0
    p1 = np.array(p1, ndmin=2)[:, COLS.XYZ]
    p2 = np.array(p2, ndmin=2)[:, COLS.XYZ]
    if p1.shape[0] == 1:
        p1 = np.repeat(p1, p2.shape[0], axis=0)
    if p2.shape[0] == 1:
        p2 = np.repeat(p2, p1.shape[0], axis=0)
    v1 = p1 / np.linalg.norm(p1, axis=1)[:, np.newaxis]
    v2 = p2 / np.linalg.norm(p2, axis=1)[:, np.newaxis]
    dot = np.einsum("ij,ij->i", v1, v2)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def vector(p1: np.ndarray, p2: np.ndarray):
    """Compute vector between two 3D points.

    Args:
        p1: the first point of the vector
        p2: the second point of the vector

    Returns:
        3-vector from p1 - p2
    """
    return np.subtract(p1[..., COLS.XYZ], p2[..., COLS.XYZ])


def section_segment_angles(section, reference=None):
    """Angles between the segments of a section and a reference vector."""
    ref = np.array(reference) if reference is not None else np.array([0, 1, 0])

    seg_vectors = vector(section.points[1:], section.points[:-1])
    return angle_between_vectors(seg_vectors, ref)


@features.feature(shape=(...,), namespace=features.NameSpace.NEURITE)
def segment_angles(neurite, reference=None):
    """Compute the angles between segments of the sections of a neurite."""
    # pylint: disable=protected-access
    func = partial(section_segment_angles, reference=reference)
    return features.neurite._map_segments(func, neurite)  # noqa: SLF001


def compute_trunk_properties(
    trunk_morph: Morphology,
    morph_name: str,
    morph_path: str,
    axon_id: int,
    config_name: str,
    atlas_region_id: int | None,
) -> tuple:
    """Compute the properties of the trunk morphologies listed in the given DataFrame."""
    # Load morph paths
    logger.info("Extracting trunk properties from %s", trunk_morph.name)

    # Compute long-range trunk features that will be used for smoothing and jittering
    long_range_trunk = get_axons(trunk_morph, 0)
    trunk_stats = morph_stats.extract_stats(
        long_range_trunk,
        {
            "neurite": {
                "segment_lengths": {"modes": ["mean", "std"]},
                "segment_meander_angles": {"modes": ["mean", "std"]},
            },
        },
    )["axon"]
    return (
        morph_name,
        morph_path,
        config_name,
        axon_id,
        atlas_region_id,
        trunk_stats["mean_segment_lengths"],
        trunk_stats["std_segment_lengths"],
        trunk_stats["mean_segment_meander_angles"],
        trunk_stats["std_segment_meander_angles"],
    )
