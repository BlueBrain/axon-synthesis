"""Define the class to store the synthesis outputs."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import ClassVar

from attrs import asdict
from attrs import define
from attrs import field

from axon_synthesis.base_path_builder import BasePathBuilder


@define
class OutputConfig:
    """Class to store the parameters for the outputs.

    Attributes:
        path: The path to the output directory.
        final_figures: Enable export of the final figures.
        graph_creation_figures: Enable export of the graph creation figures.
        graph_creation_data: Enable export of the graph creation data.
        main_trunk_figures: Enable export of the main trunk figures.
        main_trunk_morphologies: Enable export of the main trunk morphologies.
        morphologies: Enable export of the final morphologies.
        morphologies_edges: Enable export of the final morphologies as edges.
        postprocess_trunk_figures: Enable export of the postprocess trunk figures.
        postprocess_trunk_morphologies: Enable export of the postprocess trunk morphologies.
        steiner_tree_solution_figures: Enable export of the steiner tree solution figures.
        steiner_tree_solutions: Enable export of the steiner tree solutions.
        target_point_figures: Enable export of the target point figures.
        target_points: Enable export of the target points.
        tuft_figures: Enable export of the tuft figures.
        tuft_morphologies: Enable export of the tuft morphologies.
    """

    path: Path = field(converter=Path)
    final_figures: bool = False
    graph_creation_figures: bool = False
    graph_creation_data: bool = False
    main_trunk_figures: bool = False
    main_trunk_morphologies: bool = False
    morphologies: bool = True
    morphologies_edges: bool = False
    postprocess_trunk_figures: bool = False
    postprocess_trunk_morphologies: bool = False
    steiner_tree_solution_figures: bool = False
    steiner_tree_solutions: bool = False
    target_point_figures: bool = False
    target_points: bool = False
    tuft_figures: bool = False
    tuft_morphologies: bool = False


class Outputs(BasePathBuilder):
    """Class to store the synthesis outputs."""

    def __init__(self, config: OutputConfig, *, exists: bool = False, create: bool = False):
        """Constructor for the Outputs class."""
        super().__init__(config.path, exists=exists, create=create)
        self.config = config

        for name in asdict(self.config):  # pylint: disable=not-an-iterable
            if name == "path":
                continue
            if getattr(self.config, name, False):
                self._optional_keys.discard(name.upper())
            else:
                self._filenames[name.upper()] = None
        self._reset_attributes()

    _filenames: ClassVar[dict] = {
        "FINAL_FIGURES": "FinalFigures",
        "GRAPH_CREATION_FIGURES": "GraphCreationFigures",
        "GRAPH_CREATION_DATA": "GraphCreationData",
        "MAIN_TRUNK_FIGURES": "MainTrunkFigures",
        "MAIN_TRUNK_MORPHOLOGIES": "MainTrunkMorphologies",
        "MORPHOLOGIES": "Morphologies",
        "MORPHOLOGIES_EDGES": "MorphologiesEdges",
        "POSTPROCESS_TRUNK_FIGURES": "PostProcessTrunkFigures",
        "POSTPROCESS_TRUNK_MORPHOLOGIES": "PostProcessTrunkMorphologies",
        "STEINER_TREE_SOLUTION_FIGURES": "SteinerTreeSolutionFigures",
        "STEINER_TREE_SOLUTIONS": "SteinerTreeSolutions",
        "TARGET_POINT_FIGURES": "TargetPointsFigures",
        "TARGET_POINTS": "target_points.h5",
        "TUFT_FIGURES": "TuftFigures",
        "TUFT_MORPHOLOGIES": "TuftMorphologies",
    }

    _optional_keys: ClassVar[set[str]] = {
        "FINAL_FIGURES",
        "GRAPH_CREATION_FIGURES",
        "GRAPH_CREATION_DATA",
        "MAIN_TRUNK_FIGURES",
        "MAIN_TRUNK_MORPHOLOGIES",
        "MORPHOLOGIES_EDGES",
        "POSTPROCESS_TRUNK_FIGURES",
        "POSTPROCESS_TRUNK_MORPHOLOGIES",
        "STEINER_TREE_SOLUTION_FIGURES",
        "STEINER_TREE_SOLUTIONS",
        "TARGET_POINT_FIGURES",
        "TARGET_POINTS",
        "TUFT_FIGURES",
        "TUFT_MORPHOLOGIES",
    }

    _dir_keys: ClassVar[set[str]] = {
        "FINAL_FIGURES",
        "GRAPH_CREATION_FIGURES",
        "GRAPH_CREATION_DATA",
        "MAIN_TRUNK_FIGURES",
        "MAIN_TRUNK_MORPHOLOGIES",
        "MORPHOLOGIES",
        "MORPHOLOGIES_EDGES",
        "POSTPROCESS_TRUNK_FIGURES",
        "POSTPROCESS_TRUNK_MORPHOLOGIES",
        "STEINER_TREE_SOLUTION_FIGURES",
        "STEINER_TREE_SOLUTIONS",
        "TARGET_POINT_FIGURES",
        "TUFT_FIGURES",
        "TUFT_MORPHOLOGIES",
    }
