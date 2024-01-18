"""Define the class to store the synthesis outputs."""
from typing import ClassVar

from axon_synthesis.base_path_builder import BasePathBuilder


class Outputs(BasePathBuilder):
    """Class to store the synthesis outputs."""

    _filenames: ClassVar[dict] = {
        "FINAL_FIGURES": "FinalFigures",
        "GRAPH_CREATION_FIGURES": "GraphCreationFigures",
        "GRAPH_CREATION_MORPHOLOGIES": "GraphCreationMorphologies",
        "MAIN_TRUNK_FIGURES": "MainTrunkFigures",
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
        "FINAL_FIGURES",
        "GRAPH_CREATION_FIGURES",
        "GRAPH_CREATION_MORPHOLOGIES",
        "MAIN_TRUNK_FIGURES",
        "MAIN_TRUNK_MORPHOLOGIES",
        "POSTPROCESS_TRUNK_FIGURES",
        "POSTPROCESS_TRUNK_MORPHOLOGIES",
        "STEINER_TREE_SOLUTIONS",
        "TARGET_POINTS",
        "TUFT_FIGURES",
        "TUFT_MORPHOLOGIES",
    }

    _dir_keys: ClassVar[set[str]] = {
        "FINAL_FIGURES",
        "GRAPH_CREATION_FIGURES",
        "GRAPH_CREATION_MORPHOLOGIES",
        "MAIN_TRUNK_FIGURES",
        "MAIN_TRUNK_MORPHOLOGIES",
        "MORPHOLOGIES",
        "POSTPROCESS_TRUNK_FIGURES",
        "POSTPROCESS_TRUNK_MORPHOLOGIES",
        "STEINER_TREE_SOLUTIONS",
        "TUFT_FIGURES",
        "TUFT_MORPHOLOGIES",
    }
