"""Create morphologies from the Steiner Tree solutions."""

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
from pathlib import Path

import numpy as np
import pandas as pd
from morphio import PointLevel
from morphio import SectionType
from neurom.core import Morphology
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import build_layout_properties
from axon_synthesis.utils import get_morph_pts
from axon_synthesis.utils import save_morphology
from axon_synthesis.utils import sublogger


def plot(morph, initial_morph, figure_path):
    """Plot the Steiner morphology."""
    morph_name = morph.name

    fig = make_subplots(
        cols=2,
        specs=[[{"is_3d": True}, {"is_3d": True}]],
        subplot_titles=["Main trunk", "Initial morphology"],
    )

    # Build the generated figure
    gen_builder = NeuronBuilder(morph, "3d", line_width=4, title=f"{morph_name}")
    gen_fig = gen_builder.get_figure()["data"]
    fig.add_traces(
        gen_fig,
        rows=[1] * len(gen_fig),
        cols=[1] * len(gen_fig),
    )

    # Build the initial figure
    if initial_morph.sections:
        initial_builder = NeuronBuilder(initial_morph, "3d", line_width=4, title=f"{morph_name}")
        initial_fig = initial_builder.get_figure()["data"]
        fig.add_traces(
            initial_fig,
            rows=[1] * len(initial_fig),
            cols=[2] * len(initial_fig),
        )

    layout_props = build_layout_properties(
        np.vstack([morph.points, get_morph_pts(initial_morph)]), 0.1
    )

    fig.update_scenes(layout_props)
    fig.update_layout(title=morph_name)

    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(figure_path)

    # Update the HTML file to synchronize the cameras between the two plots
    add_camera_sync(figure_path)


def build_and_graft_trunk(
    morph: Morphology,
    source_section_id: int,
    # nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    output_path: FileType | None = None,
    figure_path: FileType | None = None,
    initial_morph: Morphology | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> int:
    """Build and graft a trunk to a morphology from a set of nodes and edges."""
    logger = sublogger(logger, __name__)

    # Build the synthesized axon
    active_sections = []

    edges["section_id"] = -999
    edges["reversed_edge"] = False

    edges_tmp = edges[
        [
            "from",
            "to",
            *FROM_COORDS_COLS,
            *TO_COORDS_COLS,
            "source_is_terminal",
            "target_is_terminal",
        ]
    ].copy()

    if source_section_id == -1:
        # Create a root section to start a new axon
        roots = edges_tmp.loc[edges_tmp["from"] == 0]
        if len(roots) > 1:
            # Handle bifurcation at root
            from_pt = roots[FROM_COORDS_COLS].to_numpy()[0]
            to_pt = np.concatenate([[from_pt], roots[TO_COORDS_COLS].to_numpy()]).mean(axis=0)
            # edges.loc[roots.index, FROM_COORDS_COLS] = [to_pt] * len(roots)
            edges_tmp.loc[roots.index, FROM_COORDS_COLS] = [to_pt] * len(roots)
            target_idx = 0
            roots_is_terminal = False
        else:
            from_pt = roots[FROM_COORDS_COLS].to_numpy()[0]
            to_pt = roots[TO_COORDS_COLS].to_numpy()[0]
            target_idx = roots["to"].to_numpy()[0]
            roots_is_terminal = bool(roots["target_is_terminal"].to_numpy()[0])

            # Remove the root edge
            edges_tmp = edges_tmp.drop(roots.index)

        # Build the root section
        root_section = morph.append_root_section(
            PointLevel(
                [
                    from_pt,
                    to_pt,
                ],
                [0, 0],
            ),
            SectionType.axon,
        )
        if roots_is_terminal:
            edges.loc[roots.index, "section_id"] = root_section.id
    else:
        # Attach the axon to the grafting section
        root_section = morph.section(source_section_id)
        edges.loc[edges["from"] == 0, "section_id"] = root_section.id
        target_idx = 0

        # Check that the source point is consistent with the last section point
        shifts = edges_tmp.loc[edges_tmp["from"] == 0, FROM_COORDS_COLS] - root_section.points[-1]
        if (~np.isclose(shifts, 0)).any():
            logger.warning(
                "The source points (%s) are not all equal to the parent section point (%s) and "
                "are thus shifted",
                edges_tmp.loc[edges_tmp["from"] == 0, FROM_COORDS_COLS].to_numpy().tolist(),
                root_section.points[-1].tolist(),
            )
            edges_tmp.loc[edges_tmp["from"] == 0, FROM_COORDS_COLS] -= shifts

    active_sections.append((root_section, target_idx))

    while active_sections:
        current_section, target = active_sections.pop()
        already_added = []
        for label, row in edges_tmp.loc[edges_tmp["from"] == target].iterrows():
            already_added.append(label)
            new_section = current_section.append_section(
                PointLevel(
                    [
                        row[FROM_COORDS_COLS].to_numpy(),
                        row[TO_COORDS_COLS].to_numpy(),
                    ],
                    [0, 0],
                ),
                SectionType.axon,
            )
            active_sections.append(
                (
                    new_section,
                    row["to"],
                ),
            )
            edges.loc[label, "section_id"] = new_section.id  # type: ignore[index]
        for label, row in edges_tmp.loc[edges_tmp["to"] == target].iterrows():
            already_added.append(label)
            new_section = current_section.append_section(
                PointLevel(
                    [
                        row[TO_COORDS_COLS].to_numpy(),
                        row[FROM_COORDS_COLS].to_numpy(),
                    ],
                    [0, 0],
                ),
                SectionType.axon,
            )
            active_sections.append(
                (
                    new_section,
                    row["from"],
                ),
            )
            edges.loc[label, ["section_id", "reversed_edge"]] = [new_section.id, True]  # type: ignore[index]
        edges_tmp = edges_tmp.drop(already_added)

    # At this point we do not merge consecutive sections that are not separated by a
    # bifurcation, we do it only at the very end of the process. This is to keep section
    # IDs synced with point IDs.

    if output_path is not None:
        # Export the morphology
        save_morphology(morph, output_path, msg=f"Export trunk morphology to {output_path}")

    if figure_path is not None:
        logger.info("Export trunk figure to %s", figure_path)
        plot(morph, initial_morph, figure_path)

    return root_section.id
