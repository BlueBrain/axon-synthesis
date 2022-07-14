"""Cluster the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import luigi
import luigi_tools
import networkx as nx
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from morphio import IterType
from morphio import PointLevel
from neurom import COLS
from neurom import NeuriteType
from neurom import load_morphology
from neurom.core import Morphology
from neurom.morphmath import section_length

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
from axon_synthesis.PCSF.clustering.utils import common_path
from axon_synthesis.PCSF.clustering.utils import create_clustered_morphology
from axon_synthesis.PCSF.clustering.utils import create_tuft_morphology
from axon_synthesis.PCSF.clustering.utils import get_barcode
from axon_synthesis.PCSF.extract_terminals import ExtractTerminals
from axon_synthesis.utils import cols_from_json
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import neurite_to_graph

logger = logging.getLogger(__name__)


class ClusteringOutputLocalTarget(TaggedOutputLocalTarget):
    """Target for clustering outputs."""

    __prefix = "clustering"  # pylint: disable=unused-private-member


class ClusterTerminals(luigi_tools.task.WorkflowTask):
    """Task to cluster the terminals."""

    terminals_path = luigi.OptionalPathParameter(
        description="Path to the terminals CSV file.",
        default=None,
        exists=True,
    )
    clustering_distance = luigi.NumericalParameter(
        description="The distance used to cluster the points.",
        var_type=float,
        default=100,
        min_value=0,
        max_value=sys.float_info.max,
    )
    max_path_clustering_distance = luigi.OptionalNumericalParameter(
        description=(
            "The maximum path distance used to cluster the points in 'sphere_parents' mode."
        ),
        var_type=float,
        default=None,
        min_value=0,
        max_value=sys.float_info.max,
    )
    clustering_number = luigi.NumericalParameter(
        description="The min number of points to define a cluster in 'sphere' mode.",
        var_type=int,
        default=20,
        min_value=1,
        max_value=sys.float_info.max,
    )
    clustering_mode = luigi.ChoiceParameter(
        description="The method used to define a cluster.",
        choices=["sphere", "sphere_parents", "barcode", "brain_regions"],
        default="sphere",
    )
    wm_unnesting = luigi.BoolParameter(
        description=(
            "If set to True, the brain regions are unnested up to the ones present in the WMR."
        ),
        default=True,
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )
    plot_debug = luigi.BoolParameter(
        description=(
            "If set to True, each group will create an interactive figure so it is possible to "
            "check the clustering parameters."
        ),
        default=False,
        parsing=luigi.BoolParameter.EXPLICIT_PARSING,
    )
    nb_workers = luigi.IntParameter(
        default=-1, description=":int: Number of jobs used by parallel tasks."
    )

    # Attributes that are populated in the run() method
    atlas = None
    brain_regions = None
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

    def load_WMR(self):
        """Get the white matter recipe data."""
        self.wm_populations = pd.read_csv(self.input()["WMR"]["wm_populations"].pathlib_path)
        self.wm_populations = cols_from_json(self.wm_populations, ["atlas_region", "filters"])
        self.wm_projections = pd.read_csv(self.input()["WMR"]["wm_projections"].pathlib_path)
        self.wm_projections = cols_from_json(
            self.wm_projections, ["mapping_coordinate_system", "targets", "atlas_region", "filters"]
        )
        self.wm_targets = pd.read_csv(self.input()["WMR"]["wm_targets"].pathlib_path)
        self.wm_targets = cols_from_json(self.wm_targets, ["target"])
        self.projection_targets = pd.read_csv(
            self.input()["WMR"]["wm_projection_targets"].pathlib_path
        )
        self.projection_targets = cols_from_json(
            self.projection_targets,
            ["targets", "atlas_region", "filters", "target", "topographical_mapping"],
        )
        with self.input()["WMR"]["wm_fractions"].pathlib_path.open("r", encoding="utf-8") as f:
            self.wm_fractions = json.load(f)
        with self.input()["WMR"]["wm_interaction_strengths"].pathlib_path.open(
            "r", encoding="utf-8"
        ) as f:
            self.wm_interaction_strengths = json.load(f)

    def run(self):
        config = Config()

        # Load terminals
        terminals = pd.read_csv(self.terminals_path or self.input()["terminals"].path)

        # Create output directories
        self.output()["figures"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies"].mkdir(parents=True, exist_ok=True, is_dir=True)

        # Get atlas data
        self.atlas, self.brain_regions, self.region_map = load_atlas(
            str(config.atlas_path),
            config.atlas_region_filename,
            config.atlas_hierarchy_filename,
        )

        # Load data from WMR
        self.load_WMR()

        all_terminal_points = []
        cluster_props = []
        output_cols = ["morph_file", "axon_id", "terminal_id", "x", "y", "z"]

        # Drop soma terminals and add them to the final points
        soma_centers_mask = terminals["axon_id"] == -1
        soma_centers = terminals.loc[soma_centers_mask].copy()
        all_terminal_points.extend(soma_centers[output_cols].to_records(index=False).tolist())
        terminals.drop(soma_centers.index, inplace=True)
        terminals["cluster_id"] = -1

        for group_name, group in terminals.groupby("morph_file"):
            logger.debug("%s: %s points", group_name, len(group))

            # Load the morphology
            morph = load_morphology(group_name)
            axon_morph = Morphology(morph)
            for i in axon_morph.root_sections:
                if i.type != NeuriteType.axon:
                    axon_morph.delete_section(i)

            # Select the soma center
            soma_center = soma_centers.loc[
                soma_centers["morph_file"] == group_name, ["x", "y", "z"]
            ].values[0]

            # Get the list of axons of the morphology
            axons = get_axons(morph)

            # Run the clustering function on each axon
            for axon_id, axon in enumerate(axons):

                if self.clustering_mode == "sphere":
                    clustering_func = clusters_from_spheres
                elif self.clustering_mode == "sphere_parents":
                    clustering_func = clusters_from_sphere_parents
                elif self.clustering_mode == "barcode":
                    clustering_func = clusters_from_barcodes
                elif self.clustering_mode == "brain_regions":
                    clustering_func = clusters_from_brain_regions

                axon_group = group.loc[group["axon_id"] == axon_id]
                (new_terminal_points, cluster_ids, _,) = clustering_func(
                    self,
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

            # Replace terminals by the cluster centers and create sections from common ancestors
            # to cluster centers
            kept_path = set()
            for (axon_id, cluster_id), cluster in group.groupby(["axon_id", "cluster_id"]):
                # Skip the root cluster
                if (cluster.cluster_id == 0).any():
                    continue

                # Compute the common ancestor of the nodes
                cluster_common_path = common_path(
                    directed_graphes[axon_id],
                    cluster["section_id"].tolist(),
                    shortest_paths=shortest_paths[axon_id],
                )
                if len(cluster) == 1 and len(cluster_common_path) > 2:
                    common_ancestor_shift = -2
                else:
                    common_ancestor_shift = -1
                common_ancestor = cluster_common_path[common_ancestor_shift]
                common_section = morph.section(common_ancestor)

                kept_path = kept_path.union(cluster_common_path)

                # Create a morphology for the current tuft and compute its barcode
                tuft_morph, tuft_ancestor = create_tuft_morphology(
                    axon_morph,
                    set(cluster["section_id"]),
                    common_ancestor,
                    cluster_common_path[:common_ancestor_shift],
                    shortest_paths[axon_id],
                )
                tuft_barcode = get_barcode(tuft_morph)

                # Compute tuft orientation
                cluster_center = cluster_df.loc[
                    (cluster_df["axon_id"] == axon_id) & (cluster_df["terminal_id"] == cluster_id),
                    ["x", "y", "z"],
                ].values[0]
                tuft_orientation = cluster_center - tuft_ancestor.points[-1]
                tuft_orientation /= np.linalg.norm(tuft_orientation)

                # Resize the common section used as root (the root section is 1um)
                new_root_section = tuft_morph.root_sections[0]
                new_root_section.points = np.vstack(
                    [
                        new_root_section.points[-1] - tuft_orientation,
                        new_root_section.points[-1],
                    ]
                )
                new_root_section.diameters = np.repeat(new_root_section.diameters[1], 2)

                # Add tuft category data
                path_distance = sum(
                    section_length(i.points) for i in common_section.iter(IterType.upstream)
                )
                radial_distance = np.linalg.norm(
                    axons[axon_id].points[0, COLS.XYZ] - common_section.points[-1]
                )
                path_length = sum(section_length(i.points) for i in common_section.iter())

                cluster_props.append(
                    (
                        group_name,
                        axon_id,
                        cluster_id,
                        cluster_center.tolist(),
                        common_ancestor,
                        tuft_ancestor.points[-1].tolist(),
                        path_distance,
                        radial_distance,
                        path_length,
                        len(cluster),
                        tuft_orientation.tolist(),
                        np.array(tuft_barcode).tolist(),
                    )
                )

                # Continue if the cluster has only one node
                if len(cluster) == 1:
                    continue

                # Create a new section from the common ancestor to the center of the cluster
                sections_to_add[common_section.id].append(
                    PointLevel(
                        [
                            common_section.points[-1],
                            cluster_df.loc[
                                (cluster_df["axon_id"] == cluster["axon_id"].iloc[0])
                                & (cluster_df["terminal_id"] == cluster_id),
                                ["x", "y", "z"],
                            ].values[0],
                        ],
                        [0, 0],
                    )
                )

            # Create the clustered morphology
            clustered_morph = create_clustered_morphology(
                morph, group_name, kept_path, sections_to_add
            )

            # Export the clustered morphology
            morph_path = (
                self.output()["morphologies"].pathlib_path
                / f"{Path(group_name).with_suffix('').name}.asc"
            )
            clustered_morph.write(str(morph_path))

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

    def output(self):
        return {
            "figures": ClusteringOutputLocalTarget("figures", create_parent=True),
            "morphologies": ClusteringOutputLocalTarget("morphologies", create_parent=True),
            "terminals": ClusteringOutputLocalTarget("clustered_terminals.csv", create_parent=True),
            "tuft_properties": ClusteringOutputLocalTarget(
                "tuft_properties.json", create_parent=True
            ),
        }
