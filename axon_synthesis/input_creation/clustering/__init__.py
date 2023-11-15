"""Cluster the terminal points of a morphology to define a main truk and a set of tufts."""
import json
import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from neurom import load_morphology

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.input_creation.clustering import extract_terminals
from axon_synthesis.input_creation.clustering.from_barcodes import (
    compute_clusters as clusters_from_barcodes,
)
from axon_synthesis.input_creation.clustering.from_brain_regions import (
    compute_clusters as clusters_from_brain_regions,
)
from axon_synthesis.input_creation.clustering.from_sphere_parents import (
    compute_clusters as clusters_from_sphere_parents,
)
from axon_synthesis.input_creation.clustering.from_spheres import (
    compute_clusters as clusters_from_spheres,
)
from axon_synthesis.input_creation.clustering.plot import plot_cluster_properties
from axon_synthesis.input_creation.clustering.plot import plot_clusters
from axon_synthesis.input_creation.clustering.utils import DefaultValidatingValidator
from axon_synthesis.input_creation.clustering.utils import create_clustered_morphology
from axon_synthesis.input_creation.clustering.utils import export_morph
from axon_synthesis.input_creation.clustering.utils import reduce_clusters
from axon_synthesis.input_creation.trunk_properties import compute_trunk_properties
from axon_synthesis.typing import FileType
from axon_synthesis.utils import cols_to_json
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import neurite_to_graph
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe

LOGGER = logging.getLogger(__name__)

CLUSTERING_CONFIGURATIONS_FILENAME = "clustering_configurations.csv"

FIGURE_DIRNAME = "figures"

CLUSTERED_MORPHOLOGIES = "clustered_morphologies"
CLUSTERED_MORPHOLOGIES_FILENAME = "clustered_morphologies.csv"
CLUSTERED_MORPHOLOGIES_PATHS_FILENAME = "clustered_morphologies_paths.csv"
CLUSTERED_TERMINALS_FILENAME = "clustered_terminals.csv"

TRUNK_MORPHOLOGIES_DIRNAME = "trunk_morphologies"
TRUNK_MORPHOLOGIES_PATHS_FILENAME = "trunk_morphologies_paths.csv"
TRUNK_PROPS_FILENAME = "trunk_properties.json"

TUFT_MORPHOLOGIES_DIRNAME = "tuft_morphologies"
TUFT_MORPHOLOGIES_PATHS_FILENAME = "tuft_morphologies_paths.csv"
TUFT_PROPS_FILENAME = "tuft_properties.json"
TUFT_PROPS_PLOT_FILENAME = "tuft_properties.pdf"


CLUSTERING_PARAM_SCHEMA = {
    "type": "array",
    "items": {
        "oneOf": [
            {
                # For 'sphere' clustering mode
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["sphere"],
                    },
                    "sphere_radius": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "default": 100,
                    },
                    "min_size": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 10,
                    },
                },
                "required": [
                    "method",
                ],
                "type": "object",
            },
            {
                # For 'sphere_parents' clustering mode
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["sphere_parents"],
                    },
                    "sphere_radius": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "default": 100,
                    },
                    "max_path_distance": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                },
                "required": [
                    "method",
                ],
                "type": "object",
            },
            {
                # For 'brain_regions' clustering mode
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["brain_regions"],
                    },
                    "wm_unnesting": {
                        "type": "boolean",
                        "default": True,
                    },
                },
                "required": [
                    "method",
                ],
                "type": "object",
            },
            {
                # For 'barcode' clustering mode
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["barcode"],
                    },
                },
                "required": [
                    "method",
                ],
                "type": "object",
            },
        ],
    },
}
CLUSTERING_PARAM_SCHEMA_VALIDATOR = DefaultValidatingValidator(CLUSTERING_PARAM_SCHEMA)


def cluster_morphologies(
    atlas: AtlasHelper,
    wmr: WhiteMatterRecipe,
    morph_dir: FileType,
    clustering_parameters: list,
    output_path: FileType,
    debug: bool = False,
    nb_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the cluster of all morphologies of the given directory."""
    CLUSTERING_PARAM_SCHEMA_VALIDATOR.validate(clustering_parameters)
    for num_config, config in enumerate(clustering_parameters):
        if "name" not in config:
            config["name"] = str(num_config)
    if len(clustering_parameters) != len(set([i["name"] for i in clustering_parameters])):
        raise ValueError(
            "The 'clustering_parameters' must contain elements with unique 'name' entries."
        )
    LOGGER.info(
        "Clustering morphologies using the following configuration: %s", clustering_parameters
    )

    output_path = Path(output_path)

    if output_path.exists():
        LOGGER.warning(
            "The '%s' folder already exists, the new morpholgies will be added to it",
            output_path,
        )

    # Export the clustering configurations so they can can be retrieved afterwards
    config_df = pd.DataFrame(
        [
            {
                "name": str(i["name"]),
                "method": i["method"],
                "params": dict(filter(lambda pair: pair[0] not in ["name", "method"], i.items())),
            }
            for i in clustering_parameters
        ]
    ).set_index("name")
    cols_to_json(config_df, ["params"]).to_csv(output_path / CLUSTERING_CONFIGURATIONS_FILENAME)

    clustered_terminals_path = output_path / CLUSTERED_TERMINALS_FILENAME
    figure_path = output_path / FIGURE_DIRNAME
    clustered_morphologies_path = output_path / CLUSTERED_MORPHOLOGIES
    trunk_morphologies_path = output_path / TRUNK_MORPHOLOGIES_DIRNAME
    tuft_morphologies_path = output_path / TUFT_MORPHOLOGIES_DIRNAME

    output_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)
    clustered_morphologies_path.mkdir(parents=True, exist_ok=True)
    trunk_morphologies_path.mkdir(parents=True, exist_ok=True)
    if debug:
        tuft_morphologies_path.mkdir(parents=True, exist_ok=True)

    terminals = extract_terminals.process_morphologies(morph_dir)
    terminals["config"] = None
    terminals["cluster_id"] = -1

    all_terminal_points = []
    cluster_props = []
    trunk_props = []
    morph_paths = defaultdict(list)
    output_cols = ["morph_file", "axon_id", "terminal_id", "cluster_size", "x", "y", "z", "config"]

    clustering_funcs = {
        "sphere": clusters_from_spheres,
        "sphere_parents": clusters_from_sphere_parents,
        "barcode": clusters_from_barcodes,
        "brain_regions": clusters_from_brain_regions,
    }

    for group_name, group in terminals.groupby("morph_file"):
        # TODO: Parallelize this loop?
        LOGGER.debug("%s: %s points", group_name, len(group))

        # Load the morphology
        morph = load_morphology(group_name)

        # Get the list of axons of the morphology
        axons = get_axons(morph)

        # Run the clustering function on each axon
        for axon_id, axon in enumerate(axons):
            # Create a graph for each axon and compute the shortest paths from the soma to all
            # terminals
            if len(axon.points) < 5:
                LOGGER.warning(
                    "The axon %s of %s is skipped because it has only %s points while we need at "
                    "least 5 points are needed",
                    axon_id,
                    group_name,
                    len(axon.points),
                )
                continue
            nodes, edges, directed_graph = neurite_to_graph(axon)
            shortest_paths = nx.single_source_shortest_path(directed_graph, -1)

            for config in clustering_parameters:
                axon_group = group.loc[group["axon_id"] == axon_id].copy(deep=True)
                config_name = str(config["name"])
                config_str = json.dumps(config)
                axon_group["config"] = config_str
                suffix = f"_{config_name}_{axon_id}"
                clustering_kwargs = {
                    "atlas": atlas,
                    "wmr": wmr,
                    "config": config,
                    "config_str": config_str,
                    "morph": morph,
                    "axon": axon,
                    "axon_id": axon_id,
                    "group_name": group_name,
                    "group": axon_group,
                    "nodes": nodes,
                    "edges": edges,
                    "directed_graph": directed_graph,
                    "output_cols": output_cols,
                    "clustered_morphologies_path": clustered_morphologies_path,
                    "trunk_morphologies_path": trunk_morphologies_path,
                    "tuft_morphologies_path": tuft_morphologies_path,
                    "figure_path": figure_path,
                    "nb_workers": nb_workers,
                    "debug": debug,
                }
                (
                    new_terminal_points,
                    cluster_ids,
                    _,
                ) = clustering_funcs[
                    config["method"]
                ](**clustering_kwargs)

                # Add the cluster to the final points
                all_terminal_points.extend(new_terminal_points)

                # Propagate cluster IDs
                axon_group["cluster_id"] = cluster_ids
                # terminals.loc[axon_group.index, "cluster_id"] = axon_group["cluster_id"]

                LOGGER.info(
                    "%s (axon %s): %s points after merge",
                    group_name,
                    axon_id,
                    len(new_terminal_points),
                )

                cluster_df = pd.DataFrame(new_terminal_points, columns=output_cols)

                # Reduce clusters to one section
                sections_to_add = defaultdict(list)
                kept_path = reduce_clusters(
                    axon_group,
                    group_name,
                    morph,
                    axon,
                    axon_id,
                    cluster_df,
                    directed_graph,
                    sections_to_add,
                    morph_paths,
                    cluster_props,
                    shortest_paths,
                    export_tuft_morph_dir=tuft_morphologies_path if debug else None,
                    config_name=config_name,
                )

                # Create the clustered morphology
                clustered_morph, trunk_morph = create_clustered_morphology(
                    morph, group_name, kept_path, sections_to_add, suffix=suffix
                )

                # Compute trunk properties
                trunk_props.extend(
                    compute_trunk_properties(trunk_morph, group_name, axon_id, config_name)
                )

                # Export the trunk and clustered morphologies
                morph_paths["clustered"].append(
                    (
                        group_name,
                        config_name,
                        export_morph(
                            clustered_morphologies_path,
                            group_name,
                            clustered_morph,
                            "clustered",
                            suffix=suffix,
                        ),
                    )
                )
                morph_paths["trunks"].append(
                    (
                        group_name,
                        config_name,
                        export_morph(
                            trunk_morphologies_path,
                            group_name,
                            trunk_morph,
                            "trunk",
                            suffix=suffix,
                        ),
                    )
                )

                # Plot the clusters
                if debug:
                    plot_clusters(
                        morph,
                        clustered_morph,
                        axon_group,
                        group_name,
                        cluster_df,
                        figure_path / f"{Path(group_name).with_suffix('').name}{suffix}.html",
                    )

    # Export long-range trunk properties
    trunk_props_df = pd.DataFrame(
        trunk_props,
        columns=[
            "morph_file",
            "config_name",
            "axon_id",
            "raw_segment_lengths",
            "mean_segment_lengths",
            "std_segment_lengths",
            "raw_segment_meander_angles",
            "mean_segment_meander_angles",
            "std_segment_meander_angles",
            "raw_segment_angles",
            "raw_segment_path_lengths",
        ],
    )
    trunk_props_df.sort_values(["morph_file", "axon_id"], inplace=True)
    trunk_props_df.to_csv(output_path / TRUNK_PROPS_FILENAME, index=False)
    LOGGER.info("Exported trunk properties to %s", output_path / TRUNK_PROPS_FILENAME)

    # Export tuft properties
    cluster_props_df = pd.DataFrame(
        cluster_props,
        columns=[
            "morph_file",
            "config_name",
            "axon_id",
            "cluster_id",
            "cluster_center_coords",
            "common_ancestor_id",
            "common_ancestor_coords",
            "path_distance",
            "radial_distance",
            "path_length",
            "cluster_size",
            "cluster_orientation",
            "cluster_barcode",
        ],
    )
    with (output_path / TUFT_PROPS_FILENAME).open(mode="w") as f:
        json.dump(cluster_props_df.to_dict("records"), f, indent=4)
    LOGGER.info("Exported tuft properties to %s", output_path / TUFT_PROPS_FILENAME)

    # Plot cluster properties
    if debug:
        plot_cluster_properties(cluster_props_df, figure_path / TUFT_PROPS_PLOT_FILENAME)
        LOGGER.info(
            "Exported figure of cluster properties to %s", figure_path / TUFT_PROPS_PLOT_FILENAME
        )

    # Export the terminals
    new_terminals = pd.DataFrame(all_terminal_points, columns=output_cols)
    # TODO: Check if some terminals are missing here and if can remove the merge below
    # new_terminals = pd.merge(
    #     new_terminals,
    #     terminals.groupby(
    #         ["morph_file", "axon_id", "cluster_id", "config"]
    #     ).size().rename("cluster_size"),
    #     left_on=["morph_file", "axon_id", "terminal_id"],
    #     right_on=["morph_file", "axon_id", "cluster_id"],
    #     how="left",
    # )
    new_terminals["cluster_size"] = new_terminals["cluster_size"].fillna(1).astype(int)
    new_terminals.sort_values(["morph_file", "axon_id", "terminal_id"], inplace=True)
    new_terminals.to_csv(clustered_terminals_path, index=False)
    LOGGER.info("Exported cluster terminals to %s", clustered_terminals_path)

    # Export morphology paths
    clustered_morph_paths = pd.DataFrame(
        morph_paths["clustered"], columns=["morph_file", "config_name", "morph_path"]
    )
    clustered_morph_paths.to_csv(output_path / CLUSTERED_MORPHOLOGIES_PATHS_FILENAME, index=False)
    LOGGER.info(
        "Exported clustered morphologies paths to %s",
        output_path / CLUSTERED_MORPHOLOGIES_PATHS_FILENAME,
    )

    trunk_morph_paths = pd.DataFrame(
        morph_paths["trunks"], columns=["morph_file", "config_name", "morph_path"]
    )
    trunk_morph_paths.to_csv(output_path / TRUNK_MORPHOLOGIES_PATHS_FILENAME, index=False)
    if debug:
        tuft_morph_paths = pd.DataFrame(
            morph_paths["tufts"],
            columns=["morph_file", "config_name", "axon_id", "cluster_id", "morph_path"],
        )
        tuft_morph_paths.to_csv(output_path / TUFT_MORPHOLOGIES_PATHS_FILENAME, index=False)
        LOGGER.info(
            "Exported tuft morphologies paths to %s", output_path / TUFT_MORPHOLOGIES_PATHS_FILENAME
        )
    else:
        tuft_morph_paths = None

    return clustered_morph_paths, trunk_morph_paths, tuft_morph_paths
