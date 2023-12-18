"""Update the properties of the tufts that will be generated later."""
import logging

import pandas as pd

from axon_synthesis.utils import sublogger


def pick_barcodes(
    terminals,
    edges,
    tuft_properties,
    *,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Choose a barcode among the ones available."""
    logger = sublogger(logger, __name__)

    tuft_properties = tuft_properties[["population_id", "weight", "orientation", "barcode"]].rename(
        columns={"orientation": "tuft_orientation"}
    )
    edge_terminals = pd.concat(
        [
            terminals.merge(
                edges.loc[edges["source_is_terminal"], ["source_terminal_id", "section_id"]].rename(
                    columns={"source_terminal_id": "terminal_id"}
                ),
                on="terminal_id",
            ),
            terminals.merge(
                edges.loc[edges["target_is_terminal"], ["target_terminal_id", "section_id"]].rename(
                    columns={"target_terminal_id": "terminal_id"}
                ),
                on="terminal_id",
            ),
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
        columns={"target_x": "tuft_x", "target_y": "tuft_y", "target_z": "tuft_z"}
    )
    return potential_barcodes.sample(weights="prob")[
        [
            "morphology",
            "axon_id",
            "terminal_id",
            "grafting_section_id",
            "x",
            "y",
            "z",
            "barcode",
            "section_id",
            "tuft_x",
            "tuft_y",
            "tuft_z",
            "tuft_orientation",
        ]
    ]