"""Create morphologies from the Steiner Tree solutions."""
import logging

import numpy as np
import pandas as pd
from morphio import PointLevel
from morphio import SectionType
from neurom.core import Morphology

# from axon_synthesis.synthesis.target_points import TARGET_COORDS_COLS
# from axon_synthesis.synthesis.source_points import SOURCE_COORDS_COLS
from axon_synthesis.synthesis.main_trunk.create_graph import FROM_COORDS_COLS
from axon_synthesis.synthesis.main_trunk.create_graph import TO_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.utils import sublogger


def build_and_graft_trunk(
    morph: Morphology,
    source_section_id: int,
    # nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    output_path: FileType | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Build and graft a trunk to a morphology from a set of nodes and edges."""
    logger = sublogger(logger, __name__)

    # Build the synthesized axon
    active_sections = []
    already_added = []

    edges["section_id"] = -999

    edges_tmp = edges.copy()

    if source_section_id == -1:
        # Create a root section to start a new axon
        roots = edges_tmp.loc[edges_tmp["from"] == 0]
        if len(roots) > 1:
            # Handle bifurcation at root
            from_pt = roots[FROM_COORDS_COLS].to_numpy()[0]
            to_pt = np.concatenate(
                [[roots[FROM_COORDS_COLS].to_numpy()[0]], roots[TO_COORDS_COLS].to_numpy()]
            ).mean(axis=0)
            edges.loc[roots.index][FROM_COORDS_COLS] = [to_pt] * len(roots)
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

    active_sections.append((root_section, target_idx))

    # import pdb
    # pdb.set_trace()

    while active_sections:
        current_section, target = active_sections.pop()
        already_added = []
        for row in edges_tmp.loc[edges_tmp["from"] == target].iterrows():
            already_added.append(row[0])
            active_sections.append(
                (
                    current_section.append_section(
                        PointLevel(
                            [
                                row[1][FROM_COORDS_COLS].to_numpy(),
                                row[1][TO_COORDS_COLS].to_numpy(),
                            ],
                            [0, 0],
                        ),
                        SectionType.axon,
                    ),
                    row[1]["to"],
                ),
            )
            if row[1]["target_is_terminal"]:
                edges.loc[row[1].name, "section_id"] = active_sections[-1][0].id
        for row in edges_tmp.loc[edges_tmp["to"] == target].iterrows():
            already_added.append(row[0])
            active_sections.append(
                (
                    current_section.append_section(
                        PointLevel(
                            [
                                row[1][TO_COORDS_COLS].to_numpy(),
                                row[1][FROM_COORDS_COLS].to_numpy(),
                            ],
                            [0, 0],
                        ),
                        SectionType.axon,
                    ),
                    row[1]["from"],
                ),
            )
            if row[1]["source_is_terminal"]:
                edges.loc[row[1].name, "section_id"] = active_sections[-1][0].id
        edges_tmp = edges_tmp.drop(already_added)

    # At this point we do not merge consecutive sections that are not separated by a
    # bifurcation, we do it only at the very end of the process. This is to keep section
    # IDs synced with point IDs.

    if output_path is not None:
        # Export the morphology
        morph.write(output_path)
        logger.info("Exported to %s", output_path)
