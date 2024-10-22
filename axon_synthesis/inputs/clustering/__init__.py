"""Cluster the terminal points of a morphology to define a main truk and a set of tufts."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import json
import logging
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar

import pandas as pd
from attrs import evolve

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.base_path_builder import FILE_SELECTION
from axon_synthesis.base_path_builder import BasePathBuilder
from axon_synthesis.inputs.clustering import extract_terminals
from axon_synthesis.inputs.clustering.from_barcodes import (
    compute_clusters as clusters_from_barcodes,
)
from axon_synthesis.inputs.clustering.from_brain_regions import (
    compute_clusters as clusters_from_brain_regions,
)
from axon_synthesis.inputs.clustering.from_sphere_parents import (
    compute_clusters as clusters_from_sphere_parents,
)
from axon_synthesis.inputs.clustering.from_spheres import compute_clusters as clusters_from_spheres
from axon_synthesis.inputs.clustering.plot import plot_cluster_properties
from axon_synthesis.inputs.clustering.plot import plot_clusters
from axon_synthesis.inputs.clustering.utils import DefaultValidatingValidator
from axon_synthesis.inputs.clustering.utils import clusters_basic_tuft
from axon_synthesis.inputs.clustering.utils import compute_shortest_paths
from axon_synthesis.inputs.clustering.utils import create_clustered_morphology
from axon_synthesis.inputs.clustering.utils import export_morph
from axon_synthesis.inputs.clustering.utils import reduce_clusters
from axon_synthesis.inputs.trunk_properties import compute_trunk_properties
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.typing import Self
from axon_synthesis.utils import COORDS_COLS
from axon_synthesis.utils import MorphNameAdapter
from axon_synthesis.utils import ParallelConfig
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import get_morphology_paths
from axon_synthesis.utils import load_morphology
from axon_synthesis.utils import neurite_to_graph
from axon_synthesis.utils import parallel_evaluator
from axon_synthesis.utils import setup_logger
from axon_synthesis.utils import temp_dir
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from morphio import PointLevel

LOGGER = logging.getLogger(__name__)

MIN_AXON_POINTS = 5

OUTPUT_COLS = [
    "morphology",
    "morph_file",
    "config_name",
    "axon_id",
    "terminal_id",
    "size",
    *COORDS_COLS,
]

CLUSTERING_FUNCS = {
    "basic": clusters_basic_tuft,
    "sphere": clusters_from_spheres,
    "sphere_parents": clusters_from_sphere_parents,
    "barcode": clusters_from_barcodes,
    "brain_regions": clusters_from_brain_regions,
}


def nodes_to_terminals_mapping(graph, shortest_paths):
    """Map nodes to terminals."""
    node_to_terminals = defaultdict(set)
    for node_id, parent_ids in shortest_paths.items():
        if not graph.nodes[node_id]["is_terminal"]:
            continue
        for j in parent_ids:
            node_to_terminals[j].add(node_id)
    return node_to_terminals


class Clustering(BasePathBuilder):
    """The class to store Clustering data."""

    _filenames: ClassVar[dict] = {
        "CLUSTERING_CONFIGURATIONS_FILENAME": "clustering_configurations.json",
        "FIGURE_DIRNAME": "figures",
        "CLUSTERED_MORPHOLOGIES_DIRNAME": "clustered_morphologies",
        "CLUSTERED_MORPHOLOGIES_PATHS_FILENAME": "clustered_morphologies_paths.csv",
        "CLUSTERED_TERMINALS_FILENAME": "clustered_terminals.csv",
        "TERMINALS_DIRNAME": "clustered_terminals",
        "TRUNK_MORPHOLOGIES_DIRNAME": "trunk_morphologies",
        "TRUNK_MORPHOLOGIES_PATHS_FILENAME": "trunk_morphologies_paths.csv",
        "TRUNK_PROPS_FILENAME": "trunk_properties.json",
        "TUFT_MORPHOLOGIES_DIRNAME": "tuft_morphologies",
        "TUFT_MORPHOLOGIES_PATHS_FILENAME": "tuft_morphologies_paths.csv",
        "TUFT_PROPS_FILENAME": "tuft_properties.json",
        "TUFT_PROPS_PLOT_FILENAME": "tuft_properties.pdf",
    }
    _optional_keys: ClassVar[set[str]] = {
        "CLUSTERED_TERMINALS_FILENAME",
        "CLUSTERED_MORPHOLOGIES_DIRNAME",
        "CLUSTERED_MORPHOLOGIES_PATHS_FILENAME",
        "FIGURE_DIRNAME",
        "TERMINALS_DIRNAME",
        "TRUNK_MORPHOLOGIES_DIRNAME",
        "TRUNK_MORPHOLOGIES_PATHS_FILENAME",
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
    PARAM_SCHEMA_VALIDATOR = DefaultValidatingValidator(PARAM_SCHEMA)  # type: ignore[operator]

    def __init__(self, path: FileType, parameters: dict, **kwargs):
        """The Clustering constructor.

        Args:
            path: The base path used to build the relative paths.
            parameters: The parameters used for clustering.
            **kwargs: The keyword arguments are passed to the base constructor.
        """
        super().__init__(path, **kwargs)

        if kwargs.get("create", False):
            self.create_tree()

        # Process parameters
        parameters = deepcopy(parameters)
        self.PARAM_SCHEMA_VALIDATOR.validate(parameters)
        self._parameters = parameters

        # Clustering results
        self.clustered_terminals: pd.DataFrame | None = None
        self.clustered_morph_paths: pd.DataFrame | None = None
        self.trunk_properties: pd.DataFrame | None = None
        self.trunk_morph_paths: pd.DataFrame | None = None
        self.tuft_properties: pd.DataFrame | None = None
        self.tuft_morph_paths: pd.DataFrame | None = None

    @property
    def parameters(self):
        """Return the parameters used for clustering."""
        return self._parameters

    def create_tree(self):
        """Create the associated directories."""
        self.path.mkdir(parents=True, exist_ok=True)
        for k, v in self:
            if k.endswith("_DIRNAME"):
                v.mkdir(parents=True, exist_ok=True)

    def plot_cluster_properties(self):
        """Plot cluster properties."""
        if self.tuft_properties is not None:
            plot_cluster_properties(self.tuft_properties, self.TUFT_PROPS_PLOT_FILENAME)
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
        if self.trunk_properties is not None:
            with self.TRUNK_PROPS_FILENAME.open(mode="w", encoding="utf-8") as f:
                json.dump(self.trunk_properties.to_dict("records"), f, indent=4)
            LOGGER.info("Exported trunk properties to %s", self.TRUNK_PROPS_FILENAME)

        # Export tuft properties
        if self.tuft_properties is not None:
            with self.TUFT_PROPS_FILENAME.open(mode="w", encoding="utf-8") as f:
                json.dump(self.tuft_properties.to_dict("records"), f, indent=4)
            LOGGER.info("Exported tuft properties to %s", self.TUFT_PROPS_FILENAME)

        # Export the terminals
        if self.clustered_terminals is not None:
            self.clustered_terminals.to_csv(self.CLUSTERED_TERMINALS_FILENAME, index=False)
            LOGGER.info("Exported cluster terminals to %s", self.CLUSTERED_TERMINALS_FILENAME)

        # Export morphology paths
        if self.clustered_morph_paths is not None:
            self.clustered_morph_paths.to_csv(
                self.CLUSTERED_MORPHOLOGIES_PATHS_FILENAME, index=False
            )
            LOGGER.info(
                "Exported clustered morphologies paths to %s",
                self.CLUSTERED_MORPHOLOGIES_PATHS_FILENAME,
            )

        # Export trunk morphology paths
        if self.trunk_morph_paths is not None:
            self.trunk_morph_paths.to_csv(self.TRUNK_MORPHOLOGIES_PATHS_FILENAME, index=False)
            LOGGER.info(
                "Exported trunk morphologies paths to %s",
                self.TRUNK_MORPHOLOGIES_PATHS_FILENAME,
            )

        # Export trunk morphology paths
        if self.tuft_morph_paths is not None:
            self.tuft_morph_paths.to_csv(self.TUFT_MORPHOLOGIES_PATHS_FILENAME, index=False)
            LOGGER.info(
                "Exported tuft morphologies paths to %s",
                self.TUFT_MORPHOLOGIES_PATHS_FILENAME,
            )

        # Export the clustering configurations so they can can be retrieved afterwards
        with self.CLUSTERING_CONFIGURATIONS_FILENAME.open(mode="w", encoding="utf-8") as f:
            json.dump(self.parameters, f, indent=4)
        LOGGER.info("Exported clustering parameters to %s", self.CLUSTERING_CONFIGURATIONS_FILENAME)

    @classmethod
    def load(
        cls,
        path: FileType,
        file_selection: FILE_SELECTION = FILE_SELECTION.NONE,
        *,
        allow_missing: bool = False,
        **kwargs,
    ) -> Self:
        """Load the clustering data from a given directory."""
        path = Path(path)
        paths = cls.build_default_paths(path)

        # Import the clustering configurations
        with paths["CLUSTERING_CONFIGURATIONS_FILENAME"].open(encoding="utf-8") as f:
            parameters = json.load(f)

        # Create the object
        obj = cls(path, parameters)

        obj.update_from_dict("trunk_properties_file", "TRUNK_PROPS_FILENAME", kwargs)
        obj.update_from_dict("tuft_properties_file", "TUFT_PROPS_FILENAME", kwargs)

        # Load data if they exist
        if file_selection <= FILE_SELECTION.REQUIRED_ONLY:
            try:
                obj.assert_exists(file_selection=FILE_SELECTION.REQUIRED_ONLY)
            except FileNotFoundError:
                if not allow_missing:
                    raise
            else:
                with obj.TRUNK_PROPS_FILENAME.open(encoding="utf-8") as f:
                    obj.trunk_properties = pd.read_json(
                        obj.TRUNK_PROPS_FILENAME, dtype={"morphology": str, "population_id": str}
                    )
                with obj.TUFT_PROPS_FILENAME.open(encoding="utf-8") as f:
                    obj.tuft_properties = pd.read_json(
                        f, dtype={"morphology": str, "population_id": str}
                    )
        if file_selection <= FILE_SELECTION.ALL or file_selection == FILE_SELECTION.OPTIONAL_ONLY:
            try:
                obj.assert_exists(file_selection=FILE_SELECTION.OPTIONAL_ONLY)
            except FileNotFoundError:
                if not allow_missing:
                    raise
            else:
                obj.clustered_terminals = pd.read_csv(
                    obj.CLUSTERED_TERMINALS_FILENAME, dtype={"morphology": str}
                )
                obj.clustered_morph_paths = pd.read_csv(obj.CLUSTERED_MORPHOLOGIES_PATHS_FILENAME)
                obj.trunk_morph_paths = pd.read_csv(obj.TRUNK_MORPHOLOGIES_PATHS_FILENAME)
                obj.tuft_morph_paths = pd.read_csv(obj.TUFT_MORPHOLOGIES_PATHS_FILENAME)

        return obj


def extract_morph_name_from_filename(df, file_col="morph_file", name_col="morphology"):
    """Add a 'morphology' column in the given DataFrame computed from the 'morph_file' column."""
    df[name_col] = df[file_col].apply(lambda x: Path(x).stem)
    return df


def export_clusters(
    clustering, trunk_props, cluster_props, all_terminal_points, morph_paths, *, debug=False
):
    """Export cluster data."""
    # Export long-range trunk properties
    clustering.trunk_properties = pd.DataFrame(
        trunk_props,
        columns=[
            "morphology",
            "morph_file",
            "config_name",
            "axon_id",
            "atlas_region_id",
            "mean_segment_lengths",
            "std_segment_lengths",
            "mean_segment_meander_angles",
            "std_segment_meander_angles",
        ],
    ).sort_values(["morphology", "config_name", "axon_id"])

    # Export tuft properties
    clustering.tuft_properties = pd.DataFrame(
        cluster_props,
        columns=[
            "morphology",
            "morph_file",
            "config_name",
            "axon_id",
            "tuft_id",
            "center_coords",
            "common_ancestor_id",
            "common_ancestor_x",
            "common_ancestor_y",
            "common_ancestor_z",
            "path_distance",
            "radial_distance",
            "path_length",
            "max_path_extent",
            "size",
            "orientation",
            "mean_tuft_length",
            "atlas_region_id",
            "population_id",
            "barcode",
        ],
    ).sort_values(["morphology", "config_name", "axon_id"])

    # Plot cluster properties
    if debug:
        clustering.plot_cluster_properties()

    # Export the terminals
    clustering.clustered_terminals = (
        pd.DataFrame(all_terminal_points, columns=OUTPUT_COLS)
        .fillna({"size": 1})
        .astype({"size": int})
        .sort_values(["morphology", "config_name", "axon_id", "terminal_id"])
    )
    # TODO: Check if some terminals are missing here and if can remove the merge below
    # clustering.clustered_terminals = pd.merge(
    #     clustering.clustered_terminals,
    #     terminals.groupby(
    #         ["morph_file", "axon_id", "tuft_id", "config"]
    #     ).size().rename("size"),
    #     left_on=["morph_file", "axon_id", "terminal_id"],
    #     right_on=["morph_file", "axon_id", "tuft_id"],
    #     how="left",
    # )

    # Export morphology paths
    clustering.clustered_morph_paths = pd.DataFrame(
        morph_paths["clustered"],
        columns=["morphology", "morph_file", "config_name", "axon_id", "morph_path"],
    ).sort_values(["morphology", "config_name", "axon_id"])

    clustering.trunk_morph_paths = pd.DataFrame(
        morph_paths["trunks"],
        columns=["morphology", "morph_file", "config_name", "axon_id", "morph_path"],
    ).sort_values(["morphology", "config_name", "axon_id"])

    if debug:
        clustering.tuft_morph_paths = pd.DataFrame(
            morph_paths["tufts"],
            columns=["morphology", "morph_file", "config_name", "axon_id", "tuft_id", "morph_path"],
        ).sort_values(["morphology", "config_name", "axon_id", "tuft_id"])

    # TODO: Should the 'use_ancestors' mode be moved here from the graph creation step?


def cluster_one_morph(
    morph_path: FileType,
    clustering: Clustering,
    atlas: AtlasHelper | None,
    wmr: WhiteMatterRecipe | None,
    projection_pop_numbers: pd.DataFrame | None,
    bouton_density: float | None,
    *,
    morph_name: str | None = None,
    debug: bool = False,
    rng: SeedType = None,
    parallel_config: ParallelConfig | None = None,
):
    """Run clustering on one morphology."""
    parallel_config = ParallelConfig() if parallel_config is None else parallel_config

    morph_name = Path(morph_path).name if morph_name is None else morph_name
    morph_path = str(morph_path)

    morph_custom_logger = MorphNameAdapter(LOGGER, extra={"morph_name": morph_name})

    pts = extract_terminals.process_morph(morph_path, morph_name)
    if len(pts) == 0:
        morph_custom_logger.warning("The morphology has no axon point")
        return ClusteringResult([], [], [], {})
    terminals = pd.DataFrame(
        pts,
        columns=["morphology", "morph_file", "axon_id", "terminal_id", "section_id", *COORDS_COLS],
    )
    terminals[["config", "tuft_id"]] = None, -1

    brain_regions = atlas.brain_regions if atlas is not None else None

    all_terminal_points: list[tuple] = []
    cluster_props: list[tuple] = []
    trunk_props: list[tuple] = []
    morph_paths: MutableMapping[str, list] = defaultdict(list)

    morph_custom_logger.debug("%s points", len(terminals))

    # Load the morphology
    morph = load_morphology(morph_path)

    # Get the source brain region
    atlas_region_id = brain_regions.lookup(morph.soma.center) if brain_regions is not None else None

    # Run the clustering function on each axon
    for axon_id, axon in enumerate(get_axons(morph)):
        # Create a graph for each axon and compute the shortest paths from the soma to all
        # terminals
        nodes, edges, directed_graph = neurite_to_graph(axon)
        shortest_paths = compute_shortest_paths(directed_graph)
        node_to_terminals = nodes_to_terminals_mapping(directed_graph, shortest_paths)
        axon_custom_logger = MorphNameAdapter(
            LOGGER, extra={"morph_name": morph_name, "axon_id": axon_id}
        )

        for config_name, config in clustering.parameters.items():
            clustering_method = config["method"]
            if clustering_method == "brain_regions" and (atlas is None or wmr is None):
                msg = (
                    "The atlas and wmr can not be None when the clustering method is "
                    "'brain_regions'."
                )
                raise ValueError(msg)
            if len(axon.points) < MIN_AXON_POINTS:
                axon_custom_logger.warning(
                    "The axon %s of %s is clustered with basic algorithm because it has only "
                    "%s points while at least 5 points are needed for other clustering "
                    "algorithms",
                    axon_id,
                    morph_name,
                    len(axon.points),
                )
                clustering_method = "basic"
            axon_group = terminals.loc[terminals["axon_id"] == axon_id].copy(deep=True)
            axon_group["config_name"] = config_name
            axon_group = axon_group.merge(
                nodes.reset_index().rename(columns={"id": "graph_node_id"})[
                    ["section_id", "graph_node_id"]
                ],
                on="section_id",
                how="left",
            )
            suffix = f"_{config_name}_{axon_id}"
            clustering_kwargs = {
                "atlas": atlas,
                "wmr": wmr,
                "config": config,
                "config_name": config_name,
                "morph": morph,
                "axon": axon,
                "axon_id": axon_id,
                "group_name": morph_name,
                "group_path": morph_path,
                "group": axon_group,
                "nodes": nodes,
                "edges": edges,
                "directed_graph": directed_graph,
                "shortest_paths": shortest_paths,
                "node_to_terminals": node_to_terminals,
                "output_cols": OUTPUT_COLS,
                "clustered_morphologies_path": clustering.CLUSTERED_MORPHOLOGIES_DIRNAME,
                "trunk_morphologies_path": clustering.TRUNK_MORPHOLOGIES_DIRNAME,
                "tuft_morphologies_path": clustering.TUFT_MORPHOLOGIES_DIRNAME,
                "figure_path": clustering.FIGURE_DIRNAME,
                "nb_workers": parallel_config.nb_processes,
                "debug": debug,
                "logger": axon_custom_logger,
            }
            new_terminal_points, tuft_ids = CLUSTERING_FUNCS[clustering_method](  # type: ignore[operator]
                **clustering_kwargs
            )

            # Add the cluster to the final points
            all_terminal_points.extend(new_terminal_points)

            # Propagate cluster IDs
            axon_group["tuft_id"] = tuft_ids

            axon_custom_logger.info(
                "%s (axon %s): %s points after merge",
                morph_name,
                axon_id,
                len(new_terminal_points),
            )

            cluster_df = pd.DataFrame(new_terminal_points, columns=OUTPUT_COLS)

            # Reduce clusters to one section
            sections_to_add: MutableMapping[int, PointLevel] = defaultdict(list)
            kept_path = reduce_clusters(
                axon_group,
                morph_path,
                morph_name,
                morph,
                axon,
                axon_id,
                cluster_df,
                directed_graph,
                nodes,
                sections_to_add,
                morph_paths,
                cluster_props,
                shortest_paths,
                bouton_density,
                brain_regions,
                atlas_region_id,
                atlas.orientations if atlas is not None else None,
                projection_pop_numbers=projection_pop_numbers,
                export_tuft_morph_dir=clustering.TUFT_MORPHOLOGIES_DIRNAME if debug else None,
                config_name=config_name,
                rng=rng,
                logger=axon_custom_logger,
            )

            # Create the clustered morphology
            clustered_morph, trunk_morph = create_clustered_morphology(
                morph,
                morph_name,
                axon_id,
                kept_path,
                sections_to_add,
                suffix=suffix,
            )

            # Compute trunk properties
            trunk_props.append(
                compute_trunk_properties(
                    trunk_morph,
                    str(morph_name),
                    str(morph_path),
                    axon_id,
                    config_name,
                    atlas_region_id,
                )
            )

            # Export the trunk and clustered morphologies
            morph_paths["clustered"].append(
                (
                    morph_name,
                    morph_path,
                    config_name,
                    axon_id,
                    export_morph(
                        clustering.CLUSTERED_MORPHOLOGIES_DIRNAME,
                        morph_name,
                        clustered_morph,
                        "clustered",
                        suffix=suffix,
                    ),
                ),
            )
            morph_paths["trunks"].append(
                (
                    morph_name,
                    morph_path,
                    config_name,
                    axon_id,
                    export_morph(
                        clustering.TRUNK_MORPHOLOGIES_DIRNAME,
                        morph_name,
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
                    morph_name,
                    cluster_df,
                    clustering.FIGURE_DIRNAME
                    / f"{Path(str(morph_name)).with_suffix('').name}{suffix}.html",
                )
                axon_group.to_csv(
                    clustering.TERMINALS_DIRNAME
                    / f"terminals_{Path(str(morph_name)).with_suffix('').name}{suffix}.csv"
                )
                cluster_df.to_csv(
                    clustering.TERMINALS_DIRNAME
                    / f"clusters_{Path(str(morph_name)).with_suffix('').name}{suffix}.csv"
                )

    return ClusteringResult(trunk_props, cluster_props, all_terminal_points, morph_paths)


def _wrapper(data: dict, **kwargs: dict) -> dict:
    """Wrap process_morph() for parallel computation."""
    all_kwargs = {**data, **kwargs}
    trunk_props, cluster_props, all_terminal_points, morph_paths = cluster_one_morph(**all_kwargs)

    return {
        "trunk_props": trunk_props,
        "cluster_props": cluster_props,
        "terminal_points": all_terminal_points,
        "morph_paths": morph_paths,
    }


class ClusteringResult:
    """Class to store clustering result for one morphology."""

    TRUNK_PROPS_FILENAME = "trunk_props"
    CLUSTER_PROPS_FILENAME = "cluster_props"
    ALL_TERMINAL_POINTS_FILENAME = "all_terminal_points"
    MORPH_PATH_FILENAME = "morph_paths"

    def __init__(self, trunk_props, cluster_props, all_terminal_points, morph_paths):
        """The ClusteringResult constructor."""
        self.trunk_props = trunk_props
        self.cluster_props = cluster_props
        self.all_terminal_points = all_terminal_points
        self.morph_paths = morph_paths

    def __getitem__(self, name):
        """Behave like a dict."""
        return getattr(self, name)

    def __iter__(self):
        """Make the class iterable."""
        yield self.trunk_props
        yield self.cluster_props
        yield self.all_terminal_points
        yield self.morph_paths

    def save(self, **kwargs):
        """Save internals to a temporary directory."""
        tmpdir = temp_dir(**kwargs)
        dir_path = Path(tmpdir.name)
        with (dir_path / self.TRUNK_PROPS_FILENAME).open(mode="wb") as f:
            pickle.dump(self.trunk_props, f)
        with (dir_path / self.CLUSTER_PROPS_FILENAME).open(mode="wb") as f:
            pickle.dump(self.cluster_props, f)
        with (dir_path / self.ALL_TERMINAL_POINTS_FILENAME).open(mode="wb") as f:
            pickle.dump(self.all_terminal_points, f)
        with (dir_path / self.MORPH_PATH_FILENAME).open(mode="wb") as f:
            pickle.dump(self.morph_paths, f)
        return tmpdir

    @classmethod
    def load(cls, path: FileType) -> Self:
        """Load internals from a directory."""
        dir_path = Path(path)
        with (dir_path / cls.TRUNK_PROPS_FILENAME).open(mode="rb") as f:
            trunk_props = pickle.load(f)  # noqa: S301
        with (dir_path / cls.CLUSTER_PROPS_FILENAME).open(mode="rb") as f:
            cluster_props = pickle.load(f)  # noqa: S301
        with (dir_path / cls.ALL_TERMINAL_POINTS_FILENAME).open(mode="rb") as f:
            all_terminal_points = pickle.load(f)  # noqa: S301
        with (dir_path / cls.MORPH_PATH_FILENAME).open(mode="rb") as f:
            morph_paths = pickle.load(f)  # noqa: S301
        return cls(trunk_props, cluster_props, all_terminal_points, morph_paths)


# This might be useful to optimize Dask communication
# @dask_serialize.register(ClusteringResult)
# def serialize(res: ClusteringResult) -> tuple[dict, list[bytes]]:
#     header = {}
#     tmpdir = res.save()
#     LOGGER.critical("Saved to %s", tmpdir.name)
#     frames = [str(tmpdir.name).encode()]
#     return header, frames

# @dask_deserialize.register(ClusteringResult)
# def deserialize(header: dict, frames: list[bytes]) -> ClusteringResult:
#     LOGGER.critical("Load from %s", frames[0].decode())
#     return ClusteringResult.load(frames[0].decode())


def cluster_morphologies(
    morph_dir: FileType,
    clustering_parameters: dict,
    output_path: FileType,
    *,
    atlas: AtlasHelper | None,
    wmr: WhiteMatterRecipe | None,
    pop_neuron_numbers: pd.DataFrame | None,
    bouton_density: float | None,
    debug: bool = False,
    rng: SeedType = None,
    parallel_config: ParallelConfig | None = None,
) -> Clustering:
    """Compute the cluster of all morphologies of the given directory."""
    if parallel_config is None:
        parallel_config = ParallelConfig()

    clustering = Clustering(output_path, clustering_parameters, create=True)

    if clustering.path.exists():
        LOGGER.warning(
            "The '%s' folder already exists, the new morphologies will be added to it",
            clustering.path,
        )

    LOGGER.info(
        "Clustering morphologies using the following configuration: %s",
        clustering.parameters,
    )

    projection_pop_numbers = (
        wmr.projection_targets.merge(
            pop_neuron_numbers[
                ["pop_raw_name", "atlas_region_volume", "pop_neuron_numbers"]
            ].drop_duplicates(),
            left_on="target_population_name",
            right_on="pop_raw_name",
            how="left",
            suffixes=("", "_target"),
        )
        if wmr is not None and wmr.projection_targets is not None and pop_neuron_numbers is not None
        else None
    )

    morphologies = get_morphology_paths(morph_dir)

    if len(morphologies) == 0:
        LOGGER.error("No morphology file found in '%s'", morph_dir)
        return clustering
    LOGGER.info("Found %s morphology files in '%s'", len(morphologies), morph_dir)
    parallel_config_tmp = evolve(
        parallel_config,
        nb_processes=min(len(morphologies), parallel_config.nb_processes, os.cpu_count() or 0),
    )

    # TODO: Should use map_partitions and load the atlas and other data from the workers
    results = parallel_evaluator(
        morphologies,
        _wrapper,
        parallel_config_tmp,
        [
            ["trunk_props", None],
            ["cluster_props", None],
            ["terminal_points", None],
            ["morph_paths", None],
        ],
        func_kwargs={
            "clustering": clustering,
            "atlas": atlas,
            "wmr": wmr,
            "projection_pop_numbers": projection_pop_numbers,
            "bouton_density": bouton_density,
            "debug": debug,
            "rng": rng,
            "parallel_config": parallel_config_tmp,
        },
        progress_bar=False,
        startup_func=partial(setup_logger, level=logging.getLevelName(LOGGER.getEffectiveLevel())),
    )

    trunk_props = results["trunk_props"].dropna().explode().dropna().tolist()
    cluster_props = results["cluster_props"].dropna().explode().dropna().tolist()
    all_terminal_points = results["terminal_points"].dropna().explode().dropna().tolist()
    paths = results["morph_paths"].dropna().apply(pd.Series).dropna()
    morph_paths = {col: paths[col].explode().tolist() for col in paths.columns}

    export_clusters(
        clustering, trunk_props, cluster_props, all_terminal_points, morph_paths, debug=debug
    )

    return clustering
