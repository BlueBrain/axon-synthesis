"""Add tufts to Steiner solutions."""
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from morph_tool import resampling
from morph_tool.converter import single_point_sphere_to_circular_contour
from morphio import SomaType
from morphio.mut import Morphology as MorphIoMorphology
from neurom import load_morphology
from neurom.core import Morphology
from neurots.generate.tree import TreeGrower
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.synthesis.tuft_properties import TUFT_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import sublogger

logger = logging.getLogger(__name__)


def plot_tuft(morph, title, output_path, morph_file=None, morph_title=None):
    """Plot the given morphology.

    If `morph_file` is not None then the given morphology is also plotted for comparison.
    """
    morph = Morphology(morph)
    fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=title)
    fig_data = [fig_builder.get_figure()["data"]]
    left_title = "Morphology with tufts"

    if morph_file is not None:
        if morph_title is None:
            morph_title = "Simplified raw morphology"
        raw_morph = load_morphology(morph_file)
        raw_morph = Morphology(resampling.resample_linear_density(raw_morph, 0.005))

        raw_builder = NeuronBuilder(raw_morph, "3d", line_width=4, title=title)

        fig = make_subplots(
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=[left_title, morph_title],
        )
        fig_data.append(raw_builder.get_figure()["data"])
    else:
        fig = make_subplots(cols=1, specs=[[{"type": "scene"}]], subplot_titles=[left_title])

    for col_num, data in enumerate(fig_data):
        fig.add_traces(data, rows=[1] * len(data), cols=[col_num + 1] * len(data))

    fig.update_scenes({"aspectmode": "data"})

    fig.layout.update(title=morph.name)

    # Export figure
    fig.write_html(output_path)

    if morph_file is not None:
        add_camera_sync(output_path)
    logger.info("Exported figure to %s", output_path)


def build_and_graft_tufts(
    morph: Morphology,
    tuft_properties: pd.DataFrame,
    parameters: dict,
    distributions: dict,
    *,
    output_dir: FileType | None = None,
    figure_dir: FileType | None = None,
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
        params["axon"]["orientation"]["values"]["orientations"] = [
            row["tuft_orientation"],
        ]
        logger.debug("Tuft orientation: %s", row["tuft_orientation"])

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
            initial_direction=row["tuft_orientation"],
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
            single_point_sphere_to_circular_contour(new_morph)
            new_morph.write((output_dir / filename).with_suffix(".h5"))

        if figure_dir is not None:
            plot_tuft(
                new_morph,
                filename,
                (figure_dir / filename).with_suffix(".html"),
            )

        # Graft the tuft to the current terminal
        sec = morph.section(row["section_id"])
        if row["use_parent"]:
            sec = sec.parent
        sec.append_section(new_morph.root_sections[0], recursive=True)
