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

    # morph_paths = []

    # Build the synthesized axon
    active_sections = []
    already_added = []

    if source_section_id == -1:
        # Create a root section to start a new axon
        roots = edges.loc[edges["from"] == 0]
        if len(roots) > 1:
            # Handle bifurcation at root
            from_pt = roots[FROM_COORDS_COLS].to_numpy()[0]
            to_pt = np.concatenate(
                [[roots[FROM_COORDS_COLS].to_numpy()[0]], roots[TO_COORDS_COLS].to_numpy()]
            ).mean(axis=0)
            edges.loc[roots.index][FROM_COORDS_COLS] = [to_pt] * len(roots)
            target_idx = 0
        else:
            from_pt = roots[FROM_COORDS_COLS].to_numpy()[0]
            to_pt = roots[TO_COORDS_COLS].to_numpy()[0]
            target_idx = roots["to"].to_numpy()[0]

            # Remove the root edge
            edges = edges.drop(roots.index)

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
    else:
        # Attach the axon to the grafting section
        root_section = morph.section(source_section_id)
        target_idx = 0

    active_sections.append((root_section, target_idx))

    while active_sections:
        current_section, target = active_sections.pop()
        already_added = []
        for row in edges.loc[edges["from"] == target].iterrows():
            already_added.append(row[0])
            active_sections.append(
                (
                    current_section.append_section(
                        PointLevel(
                            [
                                row[1][["x_from", "y_from", "z_from"]].to_numpy(),
                                row[1][["x_to", "y_to", "z_to"]].to_numpy(),
                            ],
                            [0, 0],
                        ),
                        SectionType.axon,
                    ),
                    row[1]["to"],
                ),
            )
        for row in edges.loc[edges["to"] == target].iterrows():
            already_added.append(row[0])
            active_sections.append(
                (
                    current_section.append_section(
                        PointLevel(
                            [
                                row[1][["x_to", "y_to", "z_to"]].to_numpy(),
                                row[1][["x_from", "y_from", "z_from"]].to_numpy(),
                            ],
                            [0, 0],
                        ),
                        SectionType.axon,
                    ),
                    row[1]["from"],
                ),
            )
        edges = edges.drop(already_added)

    # At this point we do not merge consecutive sections that are not separated by a
    # bifurcation, we do it only at the very end of the process. This is to keep section
    # IDs synced with point IDs.

    if output_path is not None:
        # Export the morphology
        morph.write(output_path)
        logger.info("Exported to %s", output_path)

    # morph_paths.append((str(group_name), morph_path))

    # Set the path of the new morph in the node DF
    # nodes.loc[nodes["morphology"] == group_name, "steiner_morph_file"] = morph_path

    # Export the node DF
    # nodes.to_csv(self.output()["nodes"].path, index=False)

    # Export the morph path DF
    # pd.DataFrame(morph_paths, columns=["morphology", "steiner_morph_file"]).to_csv(
    #     self.output()["morphology_paths"].path,
    #     index=False,
    # )

    # return morph


# def output(self):
#     return {
#         "nodes": TaggedOutputLocalTarget(
#             self.output_dir / "steiner_morph_nodes.csv",
#             create_parent=True,
#         ),
#         "morphologies": TaggedOutputLocalTarget(self.output_dir, create_parent=True),
#         "morphology_paths": TaggedOutputLocalTarget(
#             self.output_dir / "steiner_morph_paths.csv",
#             create_parent=True,
#         ),
#     }
