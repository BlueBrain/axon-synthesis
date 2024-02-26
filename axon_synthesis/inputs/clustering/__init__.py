"""Cluster the terminal points of a morphology to define a main truk and a set of tufts."""
import json
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar

import pandas as pd

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
from axon_synthesis.inputs.clustering.utils import compute_shortest_paths
from axon_synthesis.inputs.clustering.utils import create_clustered_morphology
from axon_synthesis.inputs.clustering.utils import export_morph
from axon_synthesis.inputs.clustering.utils import reduce_clusters
from axon_synthesis.inputs.trunk_properties import compute_trunk_properties
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.typing import Self
from axon_synthesis.utils import COORDS_COLS
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import load_morphology
from axon_synthesis.utils import neurite_to_graph
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from morphio import PointLevel

LOGGER = logging.getLogger(__name__)

MIN_AXON_POINTS = 5

OUTPUT_COLS = [
    "morph_file",
    "config_name",
    "axon_id",
    "terminal_id",
    "size",
    *COORDS_COLS,
]

CLUSTERING_FUNCS = {
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
    ) -> Self:
        """Load the clustering data from a given directory."""
        path = Path(path)
        paths = cls.build_paths(path)

        # Import the clustering configurations
        with paths["CLUSTERING_CONFIGURATIONS_FILENAME"].open(encoding="utf-8") as f:
            parameters = json.load(f)

        # Create the object
        obj = cls(path, parameters)

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
    clustering.trunk_properties = extract_morph_name_from_filename(
        pd.DataFrame(
            trunk_props,
            columns=[
                "morph_file",
                "config_name",
                "axon_id",
                "atlas_region_id",
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
    )

    # Export tuft properties
    clustering.tuft_properties = extract_morph_name_from_filename(
        pd.DataFrame(
            cluster_props,
            columns=[
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
                "size",
                "orientation",
                "mean_tuft_length",
                "population_id",
                "barcode",
            ],
        ).sort_values(["morph_file", "config_name", "axon_id"])
    )

    # Plot cluster properties
    if debug:
        clustering.plot_cluster_properties()

    # Export the terminals
    clustering.clustered_terminals = extract_morph_name_from_filename(
        pd.DataFrame(all_terminal_points, columns=OUTPUT_COLS)
        .fillna({"size": 1})
        .astype({"size": int})
        .sort_values(["morph_file", "config_name", "axon_id", "terminal_id"])
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
        columns=["morph_file", "config_name", "axon_id", "morph_path"],
    ).sort_values(["morph_file", "config_name", "axon_id"])

    clustering.trunk_morph_paths = pd.DataFrame(
        morph_paths["trunks"],
        columns=["morph_file", "config_name", "axon_id", "morph_path"],
    ).sort_values(["morph_file", "config_name", "axon_id"])

    if debug:
        clustering.tuft_morph_paths = pd.DataFrame(
            morph_paths["tufts"],
            columns=["morph_file", "config_name", "axon_id", "tuft_id", "morph_path"],
        ).sort_values(["morph_file", "config_name", "axon_id", "tuft_id"])

    # TODO: Should the 'use_ancestors' mode be moved here from the graph creation step?


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
    nb_workers: int = 1,
    rng: SeedType = None,
) -> Clustering:
    """Compute the cluster of all morphologies of the given directory."""
    clustering = Clustering(output_path, clustering_parameters, create=True)

    if clustering.path.exists():
        LOGGER.warning(
            "The '%s' folder already exists, the new morpholgies will be added to it",
            clustering.path,
        )

    LOGGER.info(
        "Clustering morphologies using the following configuration: %s",
        clustering.parameters,
    )

    brain_regions = atlas.brain_regions if atlas is not None else None
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

    terminals = extract_terminals.process_morphologies(morph_dir)
    terminals[["config", "tuft_id"]] = None, -1

    all_terminal_points: list[tuple]
    cluster_props: list[tuple]
    trunk_props: list[tuple]
    all_terminal_points, cluster_props, trunk_props = [], [], []
    morph_paths: MutableMapping[str, list] = defaultdict(list)

    for group_name, group in terminals.groupby("morph_file"):
        # group_name = str(group_name)

        # TODO: Parallelize this loop?
        LOGGER.debug("%s: %s points", group_name, len(group))

        # Load the morphology
        morph = load_morphology(group_name)

        # Get the source brain region
        atlas_region_id = brain_regions.lookup(morph.soma.center) if atlas is not None else None

        # Run the clustering function on each axon
        for axon_id, axon in enumerate(get_axons(morph)):
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
            shortest_paths = compute_shortest_paths(directed_graph)
            node_to_terminals = nodes_to_terminals_mapping(directed_graph, shortest_paths)

            for config_name, config in clustering.parameters.items():
                if config["method"] == "brain_regions" and (atlas is None or wmr is None):
                    msg = (
                        "The atlas and wmr can not be None when the clustering method is "
                        "'brain_regions'."
                    )
                    raise ValueError(msg)
                axon_group = group.loc[group["axon_id"] == axon_id].copy(deep=True)
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
                    "group_name": group_name,
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
                    "nb_workers": nb_workers,
                    "debug": debug,
                }
                new_terminal_points, tuft_ids = CLUSTERING_FUNCS[config["method"]](  # type: ignore[operator]
                    **clustering_kwargs
                )

                # Add the cluster to the final points
                all_terminal_points.extend(new_terminal_points)

                # Propagate cluster IDs
                axon_group["tuft_id"] = tuft_ids
                # terminals.loc[axon_group.index, "tuft_id"] = axon_group["tuft_id"]

                LOGGER.info(
                    "%s (axon %s): %s points after merge",
                    group_name,
                    axon_id,
                    len(new_terminal_points),
                )

                cluster_df = pd.DataFrame(new_terminal_points, columns=OUTPUT_COLS)

                # Reduce clusters to one section
                sections_to_add: MutableMapping[int, PointLevel] = defaultdict(list)
                kept_path = reduce_clusters(
                    axon_group,
                    group_name,
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
                    compute_trunk_properties(
                        trunk_morph, str(group_name), axon_id, config_name, atlas_region_id
                    ),
                )

                # Export the trunk and clustered morphologies
                morph_paths["clustered"].append(
                    (
                        group_name,
                        config_name,
                        axon_id,
                        export_morph(
                            clustering.CLUSTERED_MORPHOLOGIES_DIRNAME,
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
                            clustering.TRUNK_MORPHOLOGIES_DIRNAME,
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
                        / f"{Path(str(group_name)).with_suffix('').name}{suffix}.html",
                    )

    export_clusters(
        clustering, trunk_props, cluster_props, all_terminal_points, morph_paths, debug=debug
    )

    return clustering
