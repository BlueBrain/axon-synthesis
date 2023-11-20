"""Cluster the terminal points of a morphology to define a main truk and a set of tufts."""
import json
import logging
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import ClassVar

import networkx as nx
import pandas as pd
from neurom import load_morphology

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.base_path_builder import BasePathBuilder
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
from axon_synthesis.typing import Self
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import neurite_to_graph
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe

LOGGER = logging.getLogger(__name__)

LOADING_TYPE = Enum("LoadingType", ["ALL", "REQUIRED_ONLY", "PATHS_ONLY"])

MIN_AXON_POINTS = 5


class Clustering(BasePathBuilder):
    """The class to store Clustering data."""

    _filenames: ClassVar[dict] = {
        "CLUSTERING_CONFIGURATIONS_FILENAME": "clustering_configurations.json",
        "FIGURE_DIRNAME": "figures",
        "CLUSTERED_MORPHOLOGIES_DIRNAME": "clustered_morphologies",
        "CLUSTERED_MORPHOLOGIES_PATHS_FILENAME": "clustered_morphologies_paths.csv",
        "CLUSTERED_TERMINALS_FILENAME": "clustered_terminals.csv",
        "TRUNK_MORPHOLOGIES_DIRNAME": "trunk_morphologies",
        "TRUNK_MORPHOLOGIES_PATHS_FILENAME": "trunk_morphologies_paths.csv",
        "TRUNK_PROPS_FILENAME": "trunk_properties.json",
        "TUFT_MORPHOLOGIES_DIRNAME": "tuft_morphologies",
        "TUFT_MORPHOLOGIES_PATHS_FILENAME": "tuft_morphologies_paths.csv",
        "TUFT_PROPS_FILENAME": "tuft_properties.json",
        "TUFT_PROPS_PLOT_FILENAME": "tuft_properties.pdf",
    }
    _optional_keys: ClassVar[set[str]] = {
        "TUFT_MORPHOLOGIES_DIRNAME",
        "TUFT_MORPHOLOGIES_PATHS_FILENAME",
        "TUFT_PROPS_PLOT_FILENAME",
    }

    PARAM_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "patternProperties": {
            ".*": {
                "oneOf": [
                    {
                        # For 'sphere' clustering mode
                        "additionalProperties": False,
                        "properties": {
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
        },
    }
    PARAM_SCHEMA_VALIDATOR = DefaultValidatingValidator(PARAM_SCHEMA)

    def __init__(self, path: FileType, parameters: dict):
        """The Clustering constructor.

        Args:
            path: The base path used to build the relative paths.
            parameters: The parameters used for clustering.
        """
        super().__init__(path)

        # Process parameters
        parameters = deepcopy(parameters)
        self.PARAM_SCHEMA_VALIDATOR.validate(parameters)
        self._parameters = parameters

        # Clustering results
        self.clustered_terminals = None
        self.cluster_props_df = None
        self.clustered_morph_paths = None
        self.trunk_props_df = None
        self.trunk_morph_paths = None
        self.tuft_morph_paths = None

    @property
    def parameters(self):
        """Return the parameters used for clustering."""
        return self._parameters

    def init(self):
        """Initialize the associated directories."""
        self.path.mkdir(parents=True, exist_ok=True)
        for k, v in self:
            if k.endswith("_DIRNAME"):
                v.mkdir(parents=True, exist_ok=True)

    def plot_cluster_properties(self):
        """Plot cluster properties."""
        if self.cluster_props_df is not None:
            plot_cluster_properties(self.cluster_props_df, self.TUFT_PROPS_PLOT_FILENAME)
            LOGGER.info(
                "Exported figure of cluster properties to %s",
                self.TUFT_PROPS_PLOT_FILENAME,
            )
        else:
            LOGGER.warning(
                "Can't export figure of cluster properties because they were not computed yet"
            )

    def save(self):
        """Save the clustering data to the associated path."""
        # Export long-range trunk properties
        self.trunk_props_df.to_csv(self.TRUNK_PROPS_FILENAME, index=False)
        LOGGER.info("Exported trunk properties to %s", self.TRUNK_PROPS_FILENAME)

        # Export tuft properties
        with self.TUFT_PROPS_FILENAME.open(mode="w") as f:
            json.dump(self.cluster_props_df.to_dict("records"), f, indent=4)
        LOGGER.info("Exported tuft properties to %s", self.TUFT_PROPS_FILENAME)

        # Export the terminals
        self.clustered_terminals.to_csv(self.CLUSTERED_TERMINALS_FILENAME, index=False)
        LOGGER.info("Exported cluster terminals to %s", self.CLUSTERED_TERMINALS_FILENAME)

        # Export morphology paths
        self.clustered_morph_paths.to_csv(self.CLUSTERED_MORPHOLOGIES_PATHS_FILENAME, index=False)
        LOGGER.info(
            "Exported clustered morphologies paths to %s",
            self.CLUSTERED_MORPHOLOGIES_PATHS_FILENAME,
        )

        # Export trunk morphology paths
        self.trunk_morph_paths.to_csv(self.TRUNK_MORPHOLOGIES_PATHS_FILENAME, index=False)
        LOGGER.info(
            "Exported trunk morphologies paths to %s",
            self.TRUNK_MORPHOLOGIES_PATHS_FILENAME,
        )

        if self.tuft_morph_paths is not None:
            # Export trunk morphology paths
            self.tuft_morph_paths.to_csv(self.TUFT_MORPHOLOGIES_PATHS_FILENAME, index=False)
            LOGGER.info(
                "Exported tuft morphologies paths to %s",
                self.TUFT_MORPHOLOGIES_PATHS_FILENAME,
            )

        # Export the clustering configurations so they can can be retrieved afterwards
        with self.CLUSTERING_CONFIGURATIONS_FILENAME.open(mode="w") as f:
            json.dump(self.parameters, f, indent=4)
        LOGGER.info("Exported clustering parameters to %s", self.CLUSTERING_CONFIGURATIONS_FILENAME)

    @classmethod
    def load(
        cls,
        path: FileType,
        loading_type: LOADING_TYPE = LOADING_TYPE.PATHS_ONLY,
        *,
        allow_missing: bool = False,
    ) -> Self:
        """Load the clustering data from a given directory."""
        path = Path(path)
        paths = cls.build_paths(path)

        # Import the clustering configurations
        with paths["CLUSTERING_CONFIGURATIONS_FILENAME"].open() as f:
            parameters = json.load(f)

        # Create the object
        obj = cls(path, parameters)

        # Load data if they exist
        msg = "Some of the following files are missing: %s"
        if loading_type <= LOADING_TYPE.REQUIRED_ONLY:
            if obj.exists(require_optional=False):
                obj.trunk_props_df = pd.read_csv(obj.TRUNK_PROPS_FILENAME)
                with obj.TUFT_PROPS_FILENAME.open() as f:
                    obj.trunk_props_df = pd.read_json(f)
                obj.clustered_terminals = pd.read_csv(obj.CLUSTERED_TERMINALS_FILENAME)
                obj.clustered_morph_paths = pd.read_csv(obj.CLUSTERED_MORPHOLOGIES_PATHS_FILENAME)
                obj.trunk_morph_paths = pd.read_csv(obj.TRUNK_MORPHOLOGIES_PATHS_FILENAME)
            elif not allow_missing:
                raise FileNotFoundError(msg, list(obj.required_filenames.keys()))
        if loading_type <= LOADING_TYPE.ALL:
            if obj.exists(require_optional=True):
                obj.tuft_morph_paths = pd.read_csv(obj.TUFT_MORPHOLOGIES_PATHS_FILENAME)
            elif not allow_missing:
                raise FileNotFoundError(msg, list(obj.optional_filenames.keys()))

        return obj


def cluster_morphologies(
    atlas: AtlasHelper,
    wmr: WhiteMatterRecipe,
    morph_dir: FileType,
    clustering_parameters: dict,
    output_path: FileType,
    *,
    debug: bool = False,
    nb_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the cluster of all morphologies of the given directory."""
    clustering = Clustering(output_path, clustering_parameters)
    LOGGER.info(
        "Clustering morphologies using the following configuration: %s",
        clustering.parameters,
    )

    if clustering.path.exists():
        LOGGER.warning(
            "The '%s' folder already exists, the new morpholgies will be added to it",
            clustering.path,
        )

    clustering.init()

    terminals = extract_terminals.process_morphologies(morph_dir)
    terminals["config"] = None
    terminals["cluster_id"] = -1

    all_terminal_points = []
    cluster_props = []
    trunk_props = []
    morph_paths = defaultdict(list)
    output_cols = [
        "morph_file",
        "config_name",
        "axon_id",
        "terminal_id",
        "cluster_size",
        "x",
        "y",
        "z",
    ]

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
            if len(axon.points) < MIN_AXON_POINTS:
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

            for config_name, config in clustering.parameters.items():
                axon_group = group.loc[group["axon_id"] == axon_id].copy(deep=True)
                axon_group["config_name"] = config_name
                suffix = f"_{config_name}_{axon_id}"
                clustering_kwargs = {
                    "atlas": atlas,
                    "wmr": wmr,
                    "config": config,
                    "config_name": config_name,
                    "morph": morph,
                    "axon": axon,
                    "axon_id": axon_id,
                    "group_name": group_name,
                    "group": axon_group,
                    "nodes": nodes,
                    "edges": edges,
                    "directed_graph": directed_graph,
                    "output_cols": output_cols,
                    "clustered_morphologies_path": clustering.CLUSTERED_MORPHOLOGIES_DIRNAME,
                    "trunk_morphologies_path": clustering.TRUNK_MORPHOLOGIES_DIRNAME,
                    "tuft_morphologies_path": clustering.TUFT_MORPHOLOGIES_DIRNAME,
                    "figure_path": clustering.FIGURE_DIRNAME,
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
                    export_tuft_morph_dir=clustering.TUFT_MORPHOLOGIES_DIRNAME if debug else None,
                    config_name=config_name,
                )

                # Create the clustered morphology
                clustered_morph, trunk_morph = create_clustered_morphology(
                    morph,
                    group_name,
                    kept_path,
                    sections_to_add,
                    suffix=suffix,
                )

                # Compute trunk properties
                trunk_props.extend(
                    compute_trunk_properties(trunk_morph, group_name, axon_id, config_name),
                )

                # Export the trunk and clustered morphologies
                morph_paths["clustered"].append(
                    (
                        group_name,
                        config_name,
                        axon_id,
                        export_morph(
                            clustering.CLUSTERED_MORPHOLOGIES_PATHS_FILENAME,
                            group_name,
                            clustered_morph,
                            "clustered",
                            suffix=suffix,
                        ),
                    ),
                )
                morph_paths["trunks"].append(
                    (
                        group_name,
                        config_name,
                        axon_id,
                        export_morph(
                            clustering.TRUNK_MORPHOLOGIES_PATHS_FILENAME,
                            group_name,
                            trunk_morph,
                            "trunk",
                            suffix=suffix,
                        ),
                    ),
                )

                # Plot the clusters
                if debug:
                    plot_clusters(
                        morph,
                        clustered_morph,
                        axon_group,
                        group_name,
                        cluster_df,
                        clustering.FIGURE_DIRNAME
                        / f"{Path(group_name).with_suffix('').name}{suffix}.html",
                    )

    # Export long-range trunk properties
    clustering.trunk_props_df = pd.DataFrame(
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
    ).sort_values(["morph_file", "config_name", "axon_id"])

    # Export tuft properties
    clustering.cluster_props_df = pd.DataFrame(
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
    ).sort_values(["morph_file", "config_name", "axon_id"])

    # Plot cluster properties
    if debug:
        clustering.plot_cluster_properties()

    # Export the terminals
    clustering.clustered_terminals = (
        pd.DataFrame(all_terminal_points, columns=output_cols)
        .fillna({"cluster_size": 1})
        .astype({"cluster_size": int})
        .sort_values(["morph_file", "config_name", "axon_id", "terminal_id"])
    )
    # TODO: Check if some terminals are missing here and if can remove the merge below
    # clustering.clustered_terminals = pd.merge(
    #     clustering.clustered_terminals,
    #     terminals.groupby(
    #         ["morph_file", "axon_id", "cluster_id", "config"]
    #     ).size().rename("cluster_size"),
    #     left_on=["morph_file", "axon_id", "terminal_id"],
    #     right_on=["morph_file", "axon_id", "cluster_id"],
    #     how="left",
    # )

    # Export morphology paths
    clustering.clustered_morph_paths = pd.DataFrame(
        morph_paths["clustered"],
        columns=["morph_file", "config_name", "axon_id", "morph_path"],
    ).sort_values(["morph_file", "config_name", "axon_id"])

    clustering.trunk_morph_paths = pd.DataFrame(
        morph_paths["trunks"],
        columns=["morph_file", "config_name", "axon_id", "morph_path"],
    ).sort_values(["morph_file", "config_name", "axon_id"])

    if debug:
        clustering.tuft_morph_paths = pd.DataFrame(
            morph_paths["tufts"],
            columns=["morph_file", "config_name", "axon_id", "cluster_id", "morph_path"],
        ).sort_values(["morph_file", "config_name", "axon_id", "cluster_id"])

    clustering.save()

    return clustering
