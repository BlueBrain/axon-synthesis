"""Update the properties of the tufts that will be generated later."""
import logging

import pandas as pd

from axon_synthesis.synthesis.target_points import TARGET_COORDS_COLS
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import COORDS_COLS
from axon_synthesis.utils import CoordsCols
from axon_synthesis.utils import sublogger

TUFT_COORDS_COLS = CoordsCols("tuft_x", "tuft_y", "tuft_z")


def pick_barcodes(
    terminals,
    edges,
    tuft_properties,
    *,
    rng: SeedType = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Choose a barcode among the ones available."""
    logger = sublogger(logger, __name__)

    tuft_properties = tuft_properties[["population_id", "weight", "orientation", "barcode"]].rename(
        columns={"orientation": "tuft_orientation"}
    )

    source_terminals = terminals.merge(
        edges.loc[
            edges["source_is_terminal"],
            [
                "source_terminal_id",
                "source_is_terminal",
                "target_is_terminal",
                "section_id",
                "reversed_edge",
            ],
        ].rename(columns={"source_terminal_id": "terminal_id"}),
        on="terminal_id",
    )
    source_terminals["target_is_terminal"] = False
    target_terminals = terminals.merge(
        edges.loc[
            edges["target_is_terminal"],
            [
                "target_terminal_id",
                "source_is_terminal",
                "target_is_terminal",
                "section_id",
                "reversed_edge",
            ],
        ].rename(columns={"target_terminal_id": "terminal_id"}),
        on="terminal_id",
    )
    target_terminals["source_is_terminal"] = False

    edge_terminals = pd.concat(
        [
            source_terminals,
            target_terminals,
        ]
    )

    edges_with_props = edge_terminals.merge(
        tuft_properties, left_on="target_population_id", right_on="population_id", how="left"
    )
    missing_barcode = edges_with_props.loc[edges_with_props["barcode"].isna()]
    if not missing_barcode.empty:
        msg = "No barcode found for the populations %s, the default barcodes are used for them"
        missing_populations = missing_barcode["target_population_id"].unique().tolist()
        logger.info(msg, missing_populations)
        missing_df = edge_terminals.loc[
            edge_terminals["target_population_id"].isin(missing_populations)
        ].merge(tuft_properties.loc[tuft_properties["population_id"] == "default"], how="cross")
        edges_with_props = pd.concat([edges_with_props.dropna(subset="barcode"), missing_df])

    if "weight" not in edges_with_props.columns:
        if "cluster_weight" in edges_with_props.columns:
            edges_with_props["weight"] = edges_with_props["cluster_weight"]
        else:
            edges_with_props["weight"] = 1
    edges_with_props["weight"] = edges_with_props["weight"].fillna(1)
    potential_barcodes = (
        edges_with_props.reset_index()
        .merge(
            edges_with_props.groupby("terminal_id")["weight"].sum().rename("weight_sum"),
            left_on="terminal_id",
            right_index=True,
            how="left",
        )
        .set_index("index")
    )
    potential_barcodes["prob"] = potential_barcodes["weight"] / potential_barcodes["weight_sum"]
    potential_barcodes = potential_barcodes.rename(
        columns={
            TARGET_COORDS_COLS.X: TUFT_COORDS_COLS.X,
            TARGET_COORDS_COLS.Y: TUFT_COORDS_COLS.Y,
            TARGET_COORDS_COLS.Z: TUFT_COORDS_COLS.Z,
        }
    )
    return potential_barcodes.groupby("terminal_id").sample(weights="prob", random_state=rng)[
        [
            "morphology",
            "axon_id",
            "terminal_id",
            "grafting_section_id",
            *COORDS_COLS,
            "barcode",
            "section_id",
            *TUFT_COORDS_COLS,
            "tuft_orientation",
            "source_is_terminal",
            "target_is_terminal",
            "reversed_edge",
        ]
    ]
