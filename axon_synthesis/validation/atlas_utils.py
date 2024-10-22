"""Module to create custom atlases."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import json
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
from attrs import evolve
from brainbuilder.app.atlases import _dump_atlases  # type: ignore[import-untyped]
from brainbuilder.app.atlases import _normalize_hierarchy
from neurom import COLS
from neurom.core import Morphology
from voxcell import RegionMap
from voxcell import VoxelData

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.utils import CleanableDirectory
from axon_synthesis.utils import compute_bbox
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import load_morphology
from axon_synthesis.utils import neurite_to_pts
from axon_synthesis.utils import temp_dir
from axon_synthesis.validation.utils import segment_intersection_lengths

LOGGER = logging.getLogger(__name__)


def _empty_hierarchy() -> OrderedDict:
    """Build 'hierarchy' dict for empty atlas."""
    return OrderedDict(
        [
            ("id", 0),
            ("acronym", "root"),
            ("name", "root"),
            (
                "children",
                [],
            ),
        ]
    )


def empty_atlas(shape, voxel_dimensions, offset, layer_thicknesses=None, tmp_dir=None):
    """Create an empty Atlas."""
    tmp_dir = (
        temp_dir(delete=False) if tmp_dir is None else CleanableDirectory(tmp_dir, exist_ok=True)
    )
    brain_regions = VoxelData(np.zeros(shape, dtype=np.int32), voxel_dimensions, offset)

    if layer_thicknesses is None:
        y_extent = brain_regions.bbox[1, 1] - brain_regions.bbox[0, 1]
        thickness = y_extent / 6
        layer_thicknesses = {str(i): i * thickness for i in range(1, 7)}
    _dump_atlases(brain_regions, layer_thicknesses, tmp_dir.name)

    hierarchy = _empty_hierarchy()
    with (Path(tmp_dir.name) / "hierarchy.json").open(mode="w", encoding="utf-8") as f:
        json.dump(_normalize_hierarchy(hierarchy), f, indent=4)

    return tmp_dir


def update_source_target_regions(brain_regions, hierarchy, source_pt, target_pts, suffix=""):
    """Create brain regions for the given source and targets."""
    hierarchy_df = hierarchy.as_dataframe()

    root = hierarchy_df.loc[hierarchy_df["parent_id"] == -1]
    root_idx = root.index[0]

    new_ind = max(1, hierarchy_df.index.max() + 1)
    hierarchy_df.loc[new_ind, ["acronym", "name", "parent_id", "children_count"]] = [
        "s",
        f"source_region{suffix}",
        root_idx,
        0,
    ]
    brain_regions.raw[brain_regions.positions_to_indices(source_pt)] = new_ind
    new_ind += 1

    for num, i in enumerate(target_pts):
        hierarchy_df.loc[new_ind, ["acronym", "name", "parent_id", "children_count"]] = [
            f"t_{num}",
            f"target_region{suffix}_{num}",
            root_idx,
            0,
        ]
        brain_regions.raw[brain_regions.positions_to_indices(i)] = new_ind
        new_ind += 1

    return brain_regions, RegionMap.from_dataframe(hierarchy_df)


def dummy_preferred_regions(brain_regions, hierarchy, morph, axon_id=0):
    """Create preferred regions as brain regions for the given morphology."""
    hierarchy_df = hierarchy.as_dataframe()

    root = hierarchy_df.loc[hierarchy_df["parent_id"] == -1]

    new_ind = max(1, hierarchy_df.index.max() + 1)
    hierarchy_df.loc[new_ind, ["acronym", "name", "parent_id", "children_count"]] = [
        "dft",
        "dummy_preferred_regions",
        root.index[0],
        0,
    ]

    morph = Morphology(morph)
    axon = get_axons(morph)[axon_id]
    _nodes, edges = neurite_to_pts(axon, keep_section_segments=True, edges_with_coords=True)
    heat_map = segment_intersection_lengths(
        edges, brain_regions.bbox, brain_regions.voxel_dimensions
    )
    brain_regions.raw[np.nonzero(heat_map.raw != 0)] = new_ind

    return brain_regions, RegionMap.from_dataframe(hierarchy_df)


def morph_atlas(
    morph,
    voxel_dimensions=None,
    atlas=None,
    preferred_regions_axon_id=None,
    target_neurite_types=None,
    *,
    export=False,
    tmp_dir=None,
):
    """Create an empty Atlas based on the dimensions of the given morphology."""
    morph = load_morphology(morph, recenter=True)
    if atlas is None:
        bbox = compute_bbox(morph.points[:, COLS.XYZ], absolute_buffer=1)
        extent = bbox[1] - bbox[0]
        shape = (np.clip(extent // voxel_dimensions, 1, np.inf) + 3).astype(int)
        offset = bbox[0]
        atlas_dir = empty_atlas(shape, voxel_dimensions, offset, tmp_dir=tmp_dir)
        atlas_path = Path(atlas_dir.name)
        atlas_config = AtlasConfig(
            atlas_path,
            "brain_regions",
            layer_names=list(range(1, 7)),
            load_region_map=True,
            use_boundary=False,
        )
        atlas = AtlasHelper(atlas_config)
    else:
        atlas_dir = (
            temp_dir(delete=False)
            if tmp_dir is None
            else CleanableDirectory(tmp_dir, exist_ok=True)
        )
        atlas_path = Path(atlas_dir.name)
        atlas_config = evolve(
            atlas.config, path=atlas_path, load_region_map=True, use_boundary=False
        )
        atlas = atlas.copy()
        atlas.config = atlas_config
    brain_regions = atlas.brain_regions
    hierarchy = atlas.region_map

    if target_neurite_types is not None:
        if np.isscalar(target_neurite_types):
            target_neurite_types = [target_neurite_types]

        for num, neurite in enumerate(morph.neurites):
            if neurite.type not in target_neurite_types:
                continue
            source = neurite.root_node.points[0, COLS.XYZ]
            targets = np.array([i.points[0, COLS.XYZ] for i in neurite.iter_sections()])
            brain_regions, hierarchy = update_source_target_regions(
                brain_regions, hierarchy, source, targets, suffix=f"_{neurite.type.name}_{num}"
            )

    if preferred_regions_axon_id is not None:
        brain_regions, hierarchy = dummy_preferred_regions(
            brain_regions, hierarchy, morph, preferred_regions_axon_id
        )

    atlas.brain_regions = brain_regions
    atlas.region_map = hierarchy

    if export:
        atlas.save(atlas_path)
        LOGGER.debug("Exported atlas to '%s'", atlas_path)

    return atlas_dir, atlas
