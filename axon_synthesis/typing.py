"""Module to define custom types used in axon-synthesis."""
import os

import morphio
import neurom

FileType = str | os.PathLike
LayerNamesType = list[int | str]
LoadableMorphology = FileType | neurom.core.Morphology | morphio.Morphology | morphio.mut.Morphology
