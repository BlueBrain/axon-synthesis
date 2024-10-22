"""Generate atlas for tests."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=missing-function-docstring
import json
import os
import shutil
from pathlib import Path
from subprocess import call

from voxcell.nexus.voxelbrain import Atlas

from . import DATA


def generate_small_O1(atlas_dir):  # noqa: N802
    """Dump a small O1 atlas in folder path."""
    atlas_dir = Path(atlas_dir)

    # fmt: off
    with Path(os.devnull).open("w", encoding="utf-8") as f:
        call(["brainbuilder", "atlases",  # noqa: S603, S607
              "-n", "6,5,4,3,2,1",
              "-t", "200,100,100,100,100,200",
              "-d", "100",
              "-o", str(atlas_dir),
              "column",
              "-a", "1000",
              ], stdout=f, stderr=f)
    # fmt: on

    # Add metadata and region structure
    shutil.copyfile(DATA / "atlas_metadata.json", atlas_dir / "metadata.json")
    shutil.copyfile(DATA / "region_structure.yaml", atlas_dir / "region_structure.yaml")

    # Split each brain region into 2 sub-regions
    atlas = Atlas.open(str(atlas_dir))
    brain_regions = atlas.load_data("brain_regions")
    brain_regions.raw[11:, :, :] *= 10
    brain_regions.save_nrrd(str(atlas_dir / "brain_regions.nrrd"))

    # Update the brain region hierarchy
    with (atlas_dir / "hierarchy.json").open(encoding="utf-8") as f:
        region_map = json.load(f)
    region_map["children"][0]["acronym"] = "mc0"
    region_map["children"].append(
        {
            "id": 10,
            "acronym": "mc1",
            "name": "hypercolumn 1",
            "children": [
                {"id": 20, "acronym": "mc1;6", "name": "hypercolumn 1, 6"},
                {"id": 30, "acronym": "mc1;5", "name": "hypercolumn 1, 5"},
                {"id": 40, "acronym": "mc1;4", "name": "hypercolumn 1, 4"},
                {"id": 50, "acronym": "mc1;3", "name": "hypercolumn 1, 3"},
                {"id": 60, "acronym": "mc1;2", "name": "hypercolumn 1, 2"},
                {"id": 70, "acronym": "mc1;1", "name": "hypercolumn 1, 1"},
            ],
        },
    )

    def recursive_atlas_id(data) -> None:
        if "id" in data:
            data["atlas_id"] = data["id"]
        if "children" in data:
            for i in data["children"]:
                recursive_atlas_id(i)

    recursive_atlas_id(region_map)

    with (atlas_dir / "hierarchy.json").open("w", encoding="utf-8") as f:
        json.dump(region_map, f, indent=4, sort_keys=True)

    return atlas_dir
