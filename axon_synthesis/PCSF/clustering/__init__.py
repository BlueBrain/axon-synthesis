"""Cluster the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import json
import logging
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

# import dask.distributed
import luigi
import luigi_tools
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# from bluepyparallel import evaluate
# from bluepyparallel import init_parallel_factory
from data_validation_framework.target import TaggedOutputLocalTarget
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool import resampling
from morphio import IterType
from morphio import PointLevel
from morphio.mut import Morphology as MorphIoMorphology
from neurom import COLS
from neurom import NeuriteType
from neurom import load_morphology
from neurom.core import Morphology
from neurom.morphmath import section_length
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

# from scipy import stats
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import connected_components
# from scipy.spatial import KDTree
# from scipy.spatial.distance import pdist
from tmd.io.conversion import convert_morphio_trees

# from tmd.io.io import load_neuron_from_morphio
# from tmd.Topology.analysis import barcode_bin_centers
# from tmd.Topology.analysis import histogram_horizontal
# from tmd.Topology.analysis import histogram_stepped
from tmd.Topology.methods import tree_to_property_barcode
from tmd.Topology.persistent_properties import PersistentAngles

from axon_synthesis.atlas import load as load_atlas
from axon_synthesis.config import Config
from axon_synthesis.create_dataset import FetchWhiteMatterRecipe

# from axon_synthesis.geometry import voxel_intersection
from axon_synthesis.PCSF.clustering.from_barcodes import compute_clusters as clusters_from_barcodes
from axon_synthesis.PCSF.clustering.from_brain_regions import (
    compute_clusters as clusters_from_brain_regions,
)
from axon_synthesis.PCSF.clustering.from_sphere_parents import (
    compute_clusters as clusters_from_sphere_parents,
)
from axon_synthesis.PCSF.clustering.from_spheres import compute_clusters as clusters_from_spheres
from axon_synthesis.PCSF.clustering.utils import common_path
from axon_synthesis.PCSF.extract_terminals import ExtractTerminals
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import neurite_to_graph
from axon_synthesis.white_matter_recipe import load as load_wmr
from axon_synthesis.white_matter_recipe import process as process_wmr

# from tmd.view.plot import barcode as plot_barcode


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

    def run(self):
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        config = Config()
        terminals = pd.read_csv(self.terminals_path or self.input()["terminals"].path)

        # Get atlas data
        self.atlas, self.brain_regions, self.region_map = load_atlas(
            str(config.atlas_path),
            config.atlas_region_filename,
            config.atlas_hierarchy_filename,
        )

        # Get the white matter recipe
        self.wm_recipe = load_wmr(self.input()["WMR"].pathlib_path)

        # Process the white matter recipe
        (
            self.wm_populations,
            self.wm_projections,
            self.wm_targets,
            self.wm_fractions,
            self.wm_interaction_strengths,
            self.projection_targets,
        ) = process_wmr(
            self.wm_recipe,
            self.region_map,
            False,
            True,
            "",
        )

        self.output()["figures"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies"].mkdir(parents=True, exist_ok=True, is_dir=True)

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

            # Soma center
            soma_center = soma_centers.loc[
                soma_centers["morph_file"] == group_name, ["x", "y", "z"]
            ].values[0]

            # Cluster each axon
            axons = get_axons(morph)

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

            # Replace terminals by the cluster centers and create sections from common ancestors
            # to cluster centers
            directed_graphes = {}
            shortest_paths = {}
            for axon_id, axon in enumerate(axons):
                _, __, directed_graph = neurite_to_graph(axon)
                directed_graphes[axon_id] = directed_graph
                sections_to_add = defaultdict(list)
                kept_path = None
                shortest_paths[axon_id] = nx.single_source_shortest_path(directed_graph, -1)
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

                if kept_path is None:
                    kept_path = set(cluster_common_path)
                else:
                    kept_path = kept_path.union(cluster_common_path)

                # Get the current tuft barcode
                tuft_sections = set(
                    j
                    for terminal_id, path in shortest_paths[axon_id].items()
                    if terminal_id in set(cluster["section_id"])
                    for j in path
                ).difference(cluster_common_path[:common_ancestor_shift])
                tuft_morph = Morphology(axon_morph)

                tuft_ancestor = tuft_morph.section(common_ancestor)

                for i in tuft_morph.sections:
                    if i.id not in tuft_sections:
                        # if i is tuft_ancestor:
                        #     import pdb

                        #     pdb.set_trace()

                        tuft_morph.delete_section(i.morphio_section, recursive=False)

                for sec in list(tuft_ancestor.iter(IterType.upstream)):
                    if sec is tuft_ancestor:
                        continue
                    tuft_morph.delete_section(sec, recursive=False)

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

                # Compute the barcode
                tmd_axon = list(
                    convert_morphio_trees(MorphIoMorphology(tuft_morph).as_immutable())
                )[axon_id]
                tuft_barcode, _ = tree_to_property_barcode(
                    tmd_axon,
                    lambda tree: tree.get_point_path_distances(),
                    # lambda tree: tree.get_point_radial_distances(point=tuft_morph.soma.center),
                    property_class=PersistentAngles,
                )

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

            # Create a new morphology with the kept path and add new sections to cluster centers
            clustered_morph = Morphology(
                deepcopy(morph),
                name=f"Clustered {Path(group_name).with_suffix('').name}",
            )

            for axon, new_axon in zip(morph.neurites, clustered_morph.neurites):
                if axon.type != NeuriteType.axon:
                    continue

                root = axon.root_node
                new_root = new_axon.root_node

                assert np.array_equal(root.points, new_root.points), "The axons where messed up!"

                for sec in new_root.children:
                    clustered_morph.delete_section(sec.morphio_section)

                current_sections = [(root, new_root)]

                # Add kept sections
                while current_sections:
                    current_section, current_new_section = current_sections.pop()
                    for child in current_section.children:
                        if child.id in kept_path:
                            new_section = PointLevel(
                                child.points[:, COLS.XYZ].tolist(),
                                (child.points[:, COLS.R] * 2).tolist(),
                            )
                            current_sections.append(
                                (child, current_new_section.append_section(new_section))
                            )

                    if current_section.id in sections_to_add:
                        for new_sec in sections_to_add[current_section.id]:
                            current_new_section.append_section(new_sec)

            # Export the clustered morphology
            morph_path = (
                self.output()["morphologies"].pathlib_path
                / f"{Path(group_name).with_suffix('').name}.asc"
            )
            clustered_morph.write(str(morph_path))

            # Plot the clusters
            if self.plot_debug:

                plotted_morph = Morphology(
                    resampling.resample_linear_density(morph, 0.005),
                    name=Path(group_name).with_suffix("").name,
                )
                fig_builder = NeuronBuilder(
                    plotted_morph, "3d", line_width=4, title=f"{plotted_morph.name}"
                )

                x, y, z = group[["x", "y", "z"]].values.T
                node_trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker={"size": 3, "color": "black"},
                    name="Morphology nodes",
                )
                x, y, z = cluster_df[["x", "y", "z"]].values.T
                cluster_trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker={"size": 5, "color": "red"},
                    name="Cluster centers",
                )
                cluster_lines = [
                    [
                        [
                            i["x"],
                            cluster_df.loc[cluster_df["terminal_id"] == i["cluster_id"], "x"].iloc[
                                0
                            ],
                            None,
                        ],
                        [
                            i["y"],
                            cluster_df.loc[cluster_df["terminal_id"] == i["cluster_id"], "y"].iloc[
                                0
                            ],
                            None,
                        ],
                        [
                            i["z"],
                            cluster_df.loc[cluster_df["terminal_id"] == i["cluster_id"], "z"].iloc[
                                0
                            ],
                            None,
                        ],
                    ]
                    for i in group.to_dict("records")
                    if i["cluster_id"] >= 0
                ]
                edge_trace = go.Scatter3d(
                    x=[j for i in cluster_lines for j in i[0]],
                    y=[j for i in cluster_lines for j in i[1]],
                    z=[j for i in cluster_lines for j in i[2]],
                    hoverinfo="none",
                    mode="lines",
                    line={
                        "color": "green",
                        "width": 4,
                    },
                    name="Morphology nodes to cluster",
                )

                # Build the clustered morph figure
                clustered_builder = NeuronBuilder(
                    clustered_morph,
                    "3d",
                    line_width=4,
                    title=f"Clustered {clustered_morph.name}",
                )

                # Create the figure from the traces
                fig = make_subplots(
                    cols=2,
                    specs=[[{"is_3d": True}, {"is_3d": True}]],
                    subplot_titles=("Node clusters", "Clustered morphology"),
                )

                morph_data = fig_builder.get_figure()["data"]
                fig.add_traces(morph_data, rows=[1] * len(morph_data), cols=[1] * len(morph_data))
                fig.add_trace(node_trace, row=1, col=1)
                fig.add_trace(edge_trace, row=1, col=1)
                fig.add_trace(cluster_trace, row=1, col=1)

                clustered_morph_data = clustered_builder.get_figure()["data"]
                fig.add_traces(
                    clustered_morph_data,
                    rows=[1] * len(clustered_morph_data),
                    cols=[2] * len(clustered_morph_data),
                )
                fig.add_trace(cluster_trace, row=1, col=2)

                # Export figure
                filepath = str(
                    self.output()["figures"].pathlib_path
                    / f"{Path(group_name).with_suffix('').name}.html"
                )
                fig.write_html(filepath)

                add_camera_sync(filepath)

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
            with PdfPages(self.output()["figures"].pathlib_path / "tuft_properties.pdf") as pdf:
                ax = cluster_props_df.plot.scatter(
                    x="path_distance",
                    y="cluster_size",
                    title="Cluster size vs path distance",
                    legend=True,
                )
                ax.set_yscale("log")
                pdf.savefig()
                plt.close()

                ax = cluster_props_df.plot.scatter(
                    x="radial_distance",
                    y="cluster_size",
                    title="Cluster size vs radial distance",
                    legend=True,
                )
                ax.set_yscale("log")
                pdf.savefig()
                plt.close()

                ax = (
                    plt.scatter(
                        x=cluster_props_df["radial_distance"],
                        y=(
                            cluster_props_df["cluster_center_coords"].apply(np.array)
                            - cluster_props_df["common_ancestor_coords"]
                        ).apply(np.linalg.norm),
                    )
                    .get_figure()
                    .gca()
                )
                ax.set_title("Cluster radial length vs radial distance")
                pdf.savefig()
                plt.close()

                ax = cluster_props_df.plot.scatter(
                    x="cluster_size",
                    y="path_length",
                    title="Path length vs cluster size",
                    legend=True,
                )
                pdf.savefig()
                plt.close()

                ax = cluster_props_df.plot.scatter(
                    x="path_distance",
                    y="path_length",
                    title="Path length vs path distance",
                    legend=True,
                )
                pdf.savefig()
                plt.close()

                ax = cluster_props_df.plot.scatter(
                    x="radial_distance",
                    y="path_length",
                    title="Path length vs radial distance",
                    legend=True,
                )
                pdf.savefig()
                plt.close()

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
