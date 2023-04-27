"""Cluster the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import json
import logging
from collections import defaultdict
from pathlib import Path

import luigi
import luigi_tools
import networkx as nx
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from neurom import load_morphology

from axon_synthesis.atlas import load as load_atlas
from axon_synthesis.config import Config
from axon_synthesis.create_dataset import FetchWhiteMatterRecipe
from axon_synthesis.PCSF.clustering.from_barcodes import compute_clusters as clusters_from_barcodes
from axon_synthesis.PCSF.clustering.from_brain_regions import (
    compute_clusters as clusters_from_brain_regions,
)
from axon_synthesis.PCSF.clustering.from_sphere_parents import (
    compute_clusters as clusters_from_sphere_parents,
)
from axon_synthesis.PCSF.clustering.from_spheres import compute_clusters as clusters_from_spheres
from axon_synthesis.PCSF.clustering.plot import plot_cluster_properties
from axon_synthesis.PCSF.clustering.plot import plot_clusters
from axon_synthesis.PCSF.clustering.utils import DefaultValidatingValidator
from axon_synthesis.PCSF.clustering.utils import create_clustered_morphology
from axon_synthesis.PCSF.clustering.utils import export_morph
from axon_synthesis.PCSF.clustering.utils import reduce_clusters
from axon_synthesis.PCSF.extract_terminals import ExtractTerminals
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import neurite_to_graph
from axon_synthesis.white_matter_recipe import load_WMR_data

logger = logging.getLogger(__name__)


class ClusteringOutputLocalTarget(TaggedOutputLocalTarget):
    """Target for clustering outputs."""

    __prefix = "clustering"  # pylint: disable=unused-private-member


_CLUSTERING_PARAM_SCHEMA = {
    "type": "array",
    "items": {
        "oneOf": [
            {
                # For 'sphere' clustering mode
                "additionalProperties": False,
                "properties": {
                    "clustering_mode": {
                        "type": "string",
                        "enum": ["sphere"],
                    },
                    "clustering_distance": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "default": 100,
                    },
                    "clustering_number": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 20,
                    },
                },
                "required": [
                    "clustering_mode",
                ],
                "type": "object",
            },
            {
                # For 'sphere_parents' clustering mode
                "additionalProperties": False,
                "properties": {
                    "clustering_mode": {
                        "type": "string",
                        "enum": ["sphere_parents"],
                    },
                    "clustering_distance": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "default": 100,
                    },
                    "max_path_clustering_distance": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                },
                "required": [
                    "clustering_mode",
                ],
                "type": "object",
            },
            {
                # For 'barcode' clustering mode
                "additionalProperties": False,
                "properties": {
                    "clustering_mode": {
                        "type": "string",
                        "enum": ["barcode"],
                    },
                },
                "required": [
                    "clustering_mode",
                ],
                "type": "object",
            },
            {
                # For 'brain_regions' clustering mode
                "additionalProperties": False,
                "properties": {
                    "clustering_mode": {
                        "type": "string",
                        "enum": ["brain_regions"],
                    },
                    "wm_unnesting": {
                        "type": "boolean",
                        "default": True,
                    },
                },
                "required": [
                    "clustering_mode",
                ],
                "type": "object",
            },
        ],
    },
}
_CLUSTERING_PARAM_SCHEMA_VALIDATOR = DefaultValidatingValidator(_CLUSTERING_PARAM_SCHEMA)


class ClusterTerminals(luigi_tools.task.WorkflowTask):
    """Task to cluster the terminals."""

    terminals_path = luigi.OptionalPathParameter(
        description=":str: Path to the terminals CSV file.",
        default=None,
        exists=True,
    )
    clustering_parameters = luigi.ListParameter(
        description=":list: A JSON list of parameter sets used to compute the clusters.",
        schema=_CLUSTERING_PARAM_SCHEMA_VALIDATOR,
    )
    plot_debug = luigi.BoolParameter(
        description=(
            ":bool: If set to True, each group will create an interactive figure so it is possible "
            "to check the clustering parameters."
        ),
        default=False,
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )
    export_tuft_morphs = luigi.BoolParameter(
        description=":bool: If set to True, each tuft will be exported as a morphology.",
        default=False,
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )
    nb_workers = luigi.IntParameter(
        default=1, description=":int: Number of jobs used by parallel tasks."
    )

    # Attributes that are populated in the run() method
    atlas = None
    brain_regions = None
    config = None
    region_map = None
    wm_recipe = None
    wm_populations = None
    wm_projections = None
    wm_targets = None
    wm_fractions = None
    wm_interaction_strengths = None
    projection_targets = None

    def requires(self):
        return {
            "terminals": ExtractTerminals(),
            "WMR": FetchWhiteMatterRecipe(),
        }

    def load_atlas(self):
        """Get the atlas data."""
        self.atlas, self.brain_regions, self.region_map = load_atlas(
            str(self.config.atlas_path),
            self.config.atlas_region_filename,
            self.config.atlas_hierarchy_filename,
        )

    def load_WMR(self):
        """Get the white matter recipe data."""
        (
            self.wm_populations,
            self.wm_projections,
            self.wm_targets,
            self.projection_targets,
            self.wm_fractions,
            self.wm_interaction_strengths,
        ) = load_WMR_data(
            self.input()["WMR"]["wm_populations"].pathlib_path,
            self.input()["WMR"]["wm_projections"].pathlib_path,
            self.input()["WMR"]["wm_targets"].pathlib_path,
            self.input()["WMR"]["wm_projection_targets"].pathlib_path,
            self.input()["WMR"]["wm_fractions"].pathlib_path,
            self.input()["WMR"]["wm_interaction_strengths"].pathlib_path,
        )

    def run(self):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        self.config = Config()
        clustering_parameters = luigi.freezing.recursively_unfreeze(self.clustering_parameters)
        print("############################################################################")
        print(clustering_parameters)
        print("############################################################################")

        # Load terminals
        terminals = pd.read_csv(self.terminals_path or self.input()["terminals"].path)
        terminals["config"] = None

        # Create output directories
        self.output()["figures"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["trunk_morphologies"].mkdir(parents=True, exist_ok=True, is_dir=True)
        if self.export_tuft_morphs:
            self.output()["tuft_morphologies"].mkdir(parents=True, exist_ok=True, is_dir=True)

        # Get atlas data
        self.load_atlas()

        # Load data from WMR
        self.load_WMR()

        all_terminal_points = []
        cluster_props = []
        output_cols = ["morph_file", "axon_id", "terminal_id", "x", "y", "z", "config"]

        morph_paths = defaultdict(list)

        # Drop soma terminals and add them to the final points
        soma_centers_mask = terminals["axon_id"] == -1
        soma_centers = terminals.loc[soma_centers_mask].copy()
        all_terminal_points.extend(soma_centers[output_cols].to_records(index=False).tolist())
        terminals.drop(soma_centers.index, inplace=True)
        terminals["cluster_id"] = -1

        clustering_funcs = {
            "sphere": clusters_from_spheres,
            "sphere_parents": clusters_from_sphere_parents,
            "barcode": clusters_from_barcodes,
            "brain_regions": clusters_from_brain_regions,
        }

        for group_name, group in terminals.groupby("morph_file"):
            logger.debug("%s: %s points", group_name, len(group))

            # Load the morphology
            morph = load_morphology(group_name)

            # Select the soma center
            soma_center = soma_centers.loc[
                soma_centers["morph_file"] == group_name, ["x", "y", "z"]
            ].values[0]

            # Get the list of axons of the morphology
            axons = get_axons(morph)

            # Run the clustering function on each axon
            for axon_id, axon in enumerate(axons):
                for config in clustering_parameters:
                    axon_group = group.loc[group["axon_id"] == axon_id]
                    (
                        new_terminal_points,
                        cluster_ids,
                        _,
                    ) = clustering_funcs[config["clustering_mode"]](
                        self,
                        config,
                        axon,
                        axon_id,
                        group_name,
                        axon_group,
                        output_cols,
                        soma_center,
                    )
                    group.loc[axon_group.index, "cluster_id"] = cluster_ids

                    # Add the cluster to the final points
                    all_terminal_points.extend(new_terminal_points)

            # Propagate cluster IDs
            terminals.loc[group.index, "cluster_id"] = group["cluster_id"]

            logger.info("%s: %s points after merge", group_name, len(new_terminal_points))

            cluster_df = pd.DataFrame(new_terminal_points, columns=output_cols)

            # Create a graph for each axon and compute the shortest paths from the soma to all
            # terminals
            directed_graphes = {}
            shortest_paths = {}
            for axon_id, axon in enumerate(axons):
                _, __, directed_graph = neurite_to_graph(axon)
                directed_graphes[axon_id] = directed_graph
                sections_to_add = defaultdict(list)
                shortest_paths[axon_id] = nx.single_source_shortest_path(directed_graph, -1)

            # Reduce clusters to one section
            kept_path = reduce_clusters(
                group,
                group_name,
                morph,
                axons,
                cluster_df,
                directed_graphes,
                sections_to_add,
                morph_paths,
                cluster_props,
                shortest_paths,
                export_tuft_morph_dir=self.output()["tuft_morphologies"].pathlib_path
                if self.export_tuft_morphs
                else None,
            )

            # Create the clustered morphology
            clustered_morph, trunk_morph = create_clustered_morphology(
                morph, group_name, kept_path, sections_to_add
            )

            # Export the trunk and clustered morphologies
            morph_paths["clustered"].append(
                (
                    group_name,
                    export_morph(
                        self.output()["morphologies"].pathlib_path,
                        group_name,
                        clustered_morph,
                        "clustered",
                    ),
                )
            )
            morph_paths["trunks"].append(
                (
                    group_name,
                    export_morph(
                        self.output()["trunk_morphologies"].pathlib_path,
                        group_name,
                        trunk_morph,
                        "trunk",
                    ),
                )
            )

            # Plot the clusters
            if self.plot_debug:
                plot_clusters(
                    morph,
                    clustered_morph,
                    group,
                    group_name,
                    cluster_df,
                    self.output()["figures"].pathlib_path
                    / f"{Path(group_name).with_suffix('').name}.html",
                )

        # Export tuft properties
        cluster_props_df = pd.DataFrame(
            cluster_props,
            columns=[
                "morph_file",
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
        with self.output()["tuft_properties"].open(mode="w") as f:
            json.dump(cluster_props_df.to_dict("records"), f, indent=4)

        # Plot cluster properties
        if self.plot_debug:
            plot_cluster_properties(
                cluster_props_df, self.output()["figures"].pathlib_path / "tuft_properties.pdf"
            )

        # Export the terminals
        new_terminals = pd.DataFrame(all_terminal_points, columns=output_cols)
        new_terminals = pd.merge(
            new_terminals,
            terminals.groupby(["morph_file", "axon_id", "cluster_id"])
            .size()
            .rename("cluster_size"),
            left_on=["morph_file", "axon_id", "terminal_id"],
            right_on=["morph_file", "axon_id", "cluster_id"],
            how="left",
        )
        new_terminals["cluster_size"] = new_terminals["cluster_size"].fillna(1).astype(int)
        new_terminals.sort_values(["morph_file", "axon_id", "terminal_id"], inplace=True)
        new_terminals.to_csv(self.output()["terminals"].path, index=False)

        # Export morphology paths
        pd.DataFrame(morph_paths["clustered"], columns=["morph_file", "morph_path"]).to_csv(
            self.output()["morphology_paths"].path, index=False
        )
        pd.DataFrame(morph_paths["trunks"], columns=["morph_file", "morph_path"]).to_csv(
            self.output()["trunk_morphology_paths"].path, index=False
        )
        if self.export_tuft_morphs:
            pd.DataFrame(
                morph_paths["tufts"], columns=["morph_file", "axon_id", "cluster_id", "morph_path"]
            ).to_csv(self.output()["tuft_morphology_paths"].path, index=False)

    def output(self):
        targets = {
            "figures": ClusteringOutputLocalTarget("figures", create_parent=True),
            "morphologies": ClusteringOutputLocalTarget("morphologies", create_parent=True),
            "morphology_paths": ClusteringOutputLocalTarget(
                "morphology_paths.csv", create_parent=True
            ),
            "trunk_morphologies": ClusteringOutputLocalTarget(
                "trunk_morphologies", create_parent=True
            ),
            "trunk_morphology_paths": ClusteringOutputLocalTarget(
                "trunk_morphology_paths.csv", create_parent=True
            ),
            "terminals": ClusteringOutputLocalTarget("clustered_terminals.csv", create_parent=True),
            "tuft_properties": ClusteringOutputLocalTarget(
                "tuft_properties.json", create_parent=True
            ),
        }
        if self.export_tuft_morphs:
            targets["tuft_morphologies"] = ClusteringOutputLocalTarget(
                "tuft_morphologies", create_parent=True
            )
            targets["tuft_morphology_paths"] = ClusteringOutputLocalTarget(
                "tuft_morphology_paths.csv", create_parent=True
            )
        return targets
