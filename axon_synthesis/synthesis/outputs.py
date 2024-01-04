"""Define the class to store the synthesis outputs."""
from typing import ClassVar

from axon_synthesis.base_path_builder import FILE_SELECTION
from axon_synthesis.base_path_builder import BasePathBuilder


class Outputs(BasePathBuilder):
    """Class to store the synthesis outputs."""

    _filenames: ClassVar[dict] = {
        "GRAPH_CREATION": "CraphCreation",
        "MAIN_TRUNK_MORPHOLOGIES": "MainTrunkMorphologies",
        "MORPHOLOGIES": "Morphologies",
        "POSTPROCESS_TRUNK_FIGURES": "PostProcessTrunkFigures",
        "POSTPROCESS_TRUNK_MORPHOLOGIES": "PostProcessTrunkMorphologies",
        "STEINER_TREE_SOLUTIONS": "SteinerTreeSolutions",
        "TARGET_POINTS": "target_points.h5",
        "TUFT_FIGURES": "TuftFigures",
        "TUFT_MORPHOLOGIES": "TuftMorphologies",
    }

    _optional_keys: ClassVar[set[str]] = {
        "GRAPH_CREATION",
        "MAIN_TRUNK_MORPHOLOGIES",
        "STEINER_TREE_SOLUTIONS",
        "TARGET_POINTS",
        "TUFT_MORPHOLOGIES",
        "TUFT_FIGURES",
    }

    def create_dirs(self, *, file_selection: FILE_SELECTION = FILE_SELECTION.REQUIRED_ONLY):
        """Create internal directories."""
        self.MORPHOLOGIES.mkdir(parents=True, exist_ok=True)
        if file_selection == FILE_SELECTION.ALL:
            self.GRAPH_CREATION.mkdir(parents=True, exist_ok=True)
            self.MAIN_TRUNK_MORPHOLOGIES.mkdir(parents=True, exist_ok=True)
            self.POSTPROCESS_TRUNK_FIGURES.mkdir(parents=True, exist_ok=True)
            self.POSTPROCESS_TRUNK_MORPHOLOGIES.mkdir(parents=True, exist_ok=True)
            self.STEINER_TREE_SOLUTIONS.mkdir(parents=True, exist_ok=True)
            self.TUFT_FIGURES.mkdir(parents=True, exist_ok=True)
            self.TUFT_MORPHOLOGIES.mkdir(parents=True, exist_ok=True)
