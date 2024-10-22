"""Add tufts to Steiner solutions."""

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
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from morph_tool.converter import single_point_sphere_to_circular_contour
from morphio import SomaType
from morphio.mut import Morphology as MorphIoMorphology
from neurom.core import Morphology
from neurots.generate.section import SectionGrowerPath
from neurots.generate.tree import TreeGrower
from neurots.generate.tree import section_growers
from neurots.morphmath.utils import get_random_point
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.synthesis.tuft_properties import TUFT_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import build_layout_properties
from axon_synthesis.utils import disable_loggers
from axon_synthesis.utils import sublogger


class SectionGrowerBoundary(SectionGrowerPath):
    """Section grower that will not cross the given boundary."""

    def next_point(self, current_point) -> tuple:
        """Compute the next point and update it if close to the boundary."""
        boundary = self.context.get("boundary", None) if self.context is not None else None
        if boundary is None:
            return super().next_point(current_point)

        direction = (
            self.params.targeting * self.direction
            + self.params.randomness * get_random_point(random_generator=self._rng)
            + self.params.history * self.history()
        )

        direction = direction / np.linalg.norm(direction)  # From NeuroTS
        seg_length = self.step_size_distribution.draw_positive()  # From NeuroTS

        # Check where is the next point compared to the boundaries
        voxel_pos, voxel_idx = np.modf(
            (current_point - boundary.offset) / boundary.voxel_dimensions
        )
        boundary_vec = boundary.raw[tuple(voxel_idx.astype(int))]
        distance = np.linalg.norm(boundary_vec)

        # Refine boundary vector with intra-pixel position (simple linear interpolation)
        actual_boundary_vec = boundary_vec * (
            1 + np.dot(voxel_pos, boundary_vec) / max(1e-3, distance**2)
        )
        actual_distance = np.linalg.norm(actual_boundary_vec)

        # Compute the boundary attenuation component
        if actual_distance <= self.context["boundary_max_distance"]:
            boundary_direction = actual_boundary_vec / actual_distance
            attenuation = boundary_direction * np.exp(
                -self.context["boundary_scale_coeff"] * actual_distance
            )
            direction += attenuation
            direction = direction / np.linalg.norm(direction)

        # Update the next point
        next_point = current_point + seg_length * direction  # From NeuroTS
        self.update_pathlength(seg_length)  # From NeuroTS

        return next_point, direction


section_growers["path_distances+boundary"] = SectionGrowerBoundary


def plot_tuft(morph, title, output_path, initial_morph=None, morph_title=None, logger=None):
    """Plot the given morphology.

    If `initial_morph` is not None then the given morphology is also plotted for comparison.
    """
    morph = Morphology(morph)
    fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=title)
    fig_data = [fig_builder.get_figure()["data"]]
    left_title = "Morphology with tufts"

    if initial_morph is not None:
        if morph_title is None:
            morph_title = "Raw morphology"

        fig = make_subplots(
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=[left_title, morph_title],
        )

        if initial_morph.root_sections:
            fig.add_traces(
                NeuronBuilder(initial_morph, "3d", line_width=4, title=title).get_figure()["data"]
            )
        else:
            fig_builder = fig.add_traces(
                go.Scatter3d(
                    x=[initial_morph.soma.center[0]],
                    y=[initial_morph.soma.center[1]],
                    z=[initial_morph.soma.center[2]],
                    marker={"color": "black", "size": 4},
                    mode="markers",
                    name="Soma",
                )
            )
    else:
        fig = make_subplots(cols=1, specs=[[{"type": "scene"}]], subplot_titles=[left_title])

    for col_num, data in enumerate(fig_data):
        fig.add_traces(data, rows=[1] * len(data), cols=[col_num + 1] * len(data))

    layout_props = build_layout_properties(morph.points, 0.5)

    fig.update_scenes(layout_props)
    fig.update_layout(title=morph.name)

    # Export figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)

    if initial_morph is not None:
        add_camera_sync(output_path)

    if logger is not None:
        logger.info("Exported figure to %s", output_path)


def build_and_graft_tufts(
    morph: Morphology,
    tuft_properties: pd.DataFrame,
    parameters: dict,
    distributions: dict,
    *,
    context=None,
    output_dir: FileType | None = None,
    figure_dir: FileType | None = None,
    initial_morph: Morphology | None = None,
    rng: SeedType = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Build the tufts and graft them to the given morphology.

    .. warning::
        The directories passed to ``output_dir`` and ``figure_dir`` should already exist.
    """
    logger = sublogger(logger, __name__)

    if output_dir is not None:
        output_dir = Path(output_dir)
    if figure_dir is not None:
        figure_dir = Path(figure_dir)

    rng = np.random.default_rng(rng)

    for _, row in tuft_properties.iterrows():
        # Create specific parameters
        params = deepcopy(parameters)
        tuft_orientation = np.dot(row["target_orientation"], row["tuft_orientation"])
        params["axon"]["orientation"]["values"]["orientations"] = [tuft_orientation]
        logger.debug("Tuft orientation: %s", tuft_orientation)

        # Create specific distributions
        distrib = deepcopy(distributions)
        distrib["axon"]["persistence_diagram"] = [
            row["barcode"],
        ]
        logger.debug("Tuft barcode: %s", row["barcode"])

        initial_point = [row[col] for col in TUFT_COORDS_COLS]
        logger.debug("Tuft start point: %s", initial_point)

        # Grow a tuft
        new_morph = MorphIoMorphology()

        grower = TreeGrower(
            new_morph,
            initial_direction=tuft_orientation,
            initial_point=initial_point,
            parameters=params["axon"],
            distributions=distrib["axon"],
            context=context,
            random_generator=rng,
        )
        while not grower.end():
            grower.next_point()

        filename = f"{row['morphology']}_{row['axon_id']}_{row['terminal_id']}"
        if output_dir is not None:
            new_morph.soma.points = [initial_point]
            new_morph.soma.diameters = [0.5]
            new_morph.soma.type = SomaType.SOMA_SINGLE_POINT
            with disable_loggers("morph_tool.converter"):
                single_point_sphere_to_circular_contour(new_morph)
            output_path = (output_dir / filename).with_suffix(".h5")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            new_morph.write(output_path)

        if figure_dir is not None:
            plot_tuft(
                new_morph,
                filename,
                (figure_dir / filename).with_suffix(".html"),
                initial_morph,
                logger=logger,
            )

        # Graft the tuft to the current terminal
        sec = morph.section(row["section_id"])
        if row["use_parent"]:
            sec = sec.parent
        sec.append_section(new_morph.root_sections[0], recursive=True)
