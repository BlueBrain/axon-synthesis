"""Add tufts to Steiner solutions."""
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
from neurots.generate.tree import TreeGrower
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.synthesis.tuft_properties import TUFT_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import build_layout_properties
from axon_synthesis.utils import disable_loggers
from axon_synthesis.utils import sublogger


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

        # Grow a tuft
        new_morph = MorphIoMorphology()

        grower = TreeGrower(
            new_morph,
            initial_direction=tuft_orientation,
            initial_point=initial_point,
            parameters=params["axon"],
            distributions=distrib["axon"],
            context=None,
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
            new_morph.write((output_dir / filename).with_suffix(".h5"))

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
