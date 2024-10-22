"""Store some constant values."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

from enum import IntEnum


class CoordsCols(list):
    """Class to associate column names to coordinates."""

    def __init__(self, *args):
        """Constructor of the CoordsCols class."""
        if len(args) != 3:
            msg = "Exactly 3 column names should be given"
            raise ValueError(msg)
        super().__init__(args)
        self.X = self[0]
        self.Y = self[1]
        self.Z = self[2]
        self.XYZ = [self.X, self.Y, self.Z]


# Point coordinates
COORDS_COLS = CoordsCols("x", "y", "z")
ATLAS_COORDS_COLS = CoordsCols("atlas_x", "atlas_y", "atlas_z")
COMMON_ANCESTOR_COORDS_COLS = CoordsCols(
    "common_ancestor_x", "common_ancestor_y", "common_ancestor_z"
)
SOURCE_COORDS_COLS = CoordsCols("source_x", "source_y", "source_z")
TARGET_COORDS_COLS = CoordsCols("target_x", "target_y", "target_z")
TUFT_COORDS_COLS = CoordsCols("tuft_x", "tuft_y", "tuft_z")

# Graph coordinates
FROM_COORDS_COLS = CoordsCols("x_from", "y_from", "z_from")
TO_COORDS_COLS = CoordsCols("x_to", "y_to", "z_to")

# Constants
AXON_GRAFTING_POINT_HDF_GROUP = "axon_grafting_points"
DEFAULT_OUTPUT_PATH = "out"
DEFAULT_POPULATION = "default"
WMR_ATLAS_ID = "id"  # Can also be "atlas_id" in other atlas versions

NodeProvider = IntEnum(
    "NodeProvider", ["source", "target", "intermediate", "random", "bbox", "Voronoi"]
)
