"""Extract the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import logging
from pathlib import Path

import pandas as pd
from morph_tool.utils import is_morphology

from axon_synthesis.typing import FileType
from axon_synthesis.utils import COORDS_COLS
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import load_morphology

logger = logging.getLogger(__name__)


def process_morph(morph_path: FileType) -> list[tuple[str, int, int, int, float, float, float]]:
    """Extract the terminal points from a morphology."""
    morph_name = Path(morph_path).name
    morph_path_str = str(morph_path)
    morph = load_morphology(morph_path)
    pts = []
    axons = get_axons(morph)

    nb_axons = len(axons)
    logger.info("%s: %s axon%s found", morph_name, nb_axons, "s" if nb_axons > 1 else "")

    for axon_id, axon in enumerate(axons):
        # Add root point
        pts.append(
            (morph_path_str, axon_id, 0, axon.root_node.id, *axon.root_node.points[0][:3].tolist()),
        )

        # Add terminal points
        terminal_id = 1
        for section in axon.iter_sections():
            if not section.children:
                pts.append(
                    (
                        morph_path_str,
                        axon_id,
                        terminal_id,
                        section.id,
                        *section.points[-1][:3].tolist(),
                    ),
                )
                terminal_id += 1

    return pts


def process_morphologies(morph_dir: FileType) -> pd.DataFrame:
    """Extract terminals from all the morphologies in the given directory."""
    morph_dir = Path(morph_dir)
    all_pts = []
    for morph_path in morph_dir.iterdir():
        # TODO: Parallelize this loop
        if not is_morphology(morph_path):
            continue
        all_pts.extend(process_morph(morph_path))

    return pd.DataFrame(
        all_pts,
        columns=[
            "morph_file",
            "axon_id",
            "terminal_id",
            "section_id",
            *COORDS_COLS,
        ],
    )
