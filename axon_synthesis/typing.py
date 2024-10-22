"""Module to define custom types used in axon-synthesis."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=unused-import
import os
import sys
from collections.abc import Sequence

import morphio
import neurom
from numpy import number as any_num
from numpy.random import BitGenerator
from numpy.random import Generator
from numpy.random import SeedSequence
from numpy.typing import ArrayLike  # noqa: F401
from numpy.typing import NDArray

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self  # noqa: F401

FileType = str | os.PathLike
LayerNamesType = list[int | str]
RegionIdsType = int | str | list[int | str]
LoadableMorphology = FileType | neurom.core.Morphology | morphio.Morphology | morphio.mut.Morphology
SeedType = None | int | Sequence[int] | SeedSequence | BitGenerator | Generator
CoordsType = list[any_num] | NDArray[any_num]
