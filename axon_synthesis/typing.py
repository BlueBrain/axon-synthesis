"""Module to define custom types used in axon-synthesis."""
import os

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self  # noqa: F401

import morphio
import neurom

FileType = str | os.PathLike
LayerNamesType = list[int | str]
LoadableMorphology = FileType | neurom.core.Morphology | morphio.Morphology | morphio.mut.Morphology
