"""Cluster the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import json
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import luigi
import luigi_tools
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from data_validation_framework.target import TaggedOutputLocalTarget
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from morphio import IterType
from morphio import PointLevel
from morph_tool import resampling
from neurom import COLS
from neurom import NeuriteType
from neurom import load_morphology
from neurom.core import Morphology
from neurom.morphmath import section_length
from plotly_helper.neuron_viewer import NeuronBuilder
from plotly.subplots import make_subplots
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from tmd.io.conversion import convert_morphio_trees
from tmd.io.io import load_neuron_from_morphio
from tmd.Topology.analysis import barcode_bin_centers
from tmd.Topology.analysis import histogram_horizontal
from tmd.Topology.analysis import histogram_stepped
from tmd.Topology.methods import tree_to_property_barcode
from tmd.Topology.persistent_properties import PersistentAngles
from tmd.view.plot import barcode as plot_barcode

from PCSF.extract_terminals import ExtractTerminals
from utils import add_camera_sync
from utils import get_axons
from utils import neurite_to_graph

logger = logging.getLogger(__name__)


class ClusteringOutputLocalTarget(TaggedOutputLocalTarget):
    __prefix = "clustering"


class ClusterTerminals(luigi_tools.task.WorkflowTask):
    terminals_path = luigi_tools.parameter.OptionalPathParameter(
        description="Path to the terminals CSV file.",
        default=None,
        exists=True,
    )
    clustering_distance = luigi.NumericalParameter(
        description="The distance used to cluster the points.",
        var_type=float,
        default=100,
        min_value=0,
        max_value=float("inf"),
    )
    max_path_clustering_distance = luigi_tools.parameter.OptionalNumericalParameter(
        description=(
            "The maximum path distance used to cluster the points in 'sphere_parents' mode."
        ),
        var_type=float,
        default=None,
        min_value=0,
        max_value=float("inf"),
    )
    clustering_number = luigi.NumericalParameter(
        description="The min number of points to define a cluster in 'sphere' mode.",
        var_type=int,
        default=20,
        min_value=1,
        max_value=float("inf"),
    )
    clustering_mode = luigi.ChoiceParameter(
        description="The method used to define a cluster.",
        choices=["sphere", "sphere_parents", "barcode"],
        default="sphere",
    )
    plot_debug = luigi.BoolParameter(
        description=(
            "If set to True, each group will create an interactive figure so it is possible to "
            "check the clustering parameters."
        ),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
    )

    def requires(self):
        return ExtractTerminals()

    def clusters_from_spheres(self, axon, axon_id, group_name, group, output_cols, _):
        """The points must be inside the ball to be merged."""
        new_terminal_points = []

        # Get the pairs of terminals closer to the given distance
        tree = KDTree(group[["x", "y", "z"]].values)
        pairs = tree.query_pairs(self.clustering_distance)

        # Get the connected components
        adjacency_matrix = np.zeros((len(group), len(group)))
        if pairs:
            adjacency_matrix[tuple(np.array(list(pairs)).T.tolist())] = 1
        np.fill_diagonal(adjacency_matrix, 1)
        graph = csr_matrix(adjacency_matrix)
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        # Define clusters from these components
        cluster_labels, cluster_sizes = np.unique(labels, return_counts=True)
        big_clusters = cluster_labels[cluster_sizes >= self.clustering_number]
        group_with_label = group.reset_index()
        group_with_label["cluster_label"] = labels
        group_with_label["distance"] = -1.0
        clusters = group_with_label.loc[
            group_with_label["cluster_label"].isin(big_clusters)
        ].groupby("cluster_label")

        # Check clusters
        real_clusters = []
        for cluster_label, cluster in clusters:
            # Check that the potential cluster is a real one (at least 'clustering_number'
            # points must be close to the center)
            distances, indices = tree.query(
                cluster[["x", "y", "z"]].mean().values,
                k=len(group),
            )
            cluster_mask = np.isin(indices, cluster.index)
            cluster_indices = indices[cluster_mask]
            cluster_distances = distances[cluster_mask]
            if (
                np.count_nonzero(cluster_distances <= self.clustering_distance)
                < self.clustering_number
            ):
                continue

            # Mark the cluster as a real one
            real_clusters.append((cluster_label, cluster))
            group_with_label.loc[cluster_indices, "distance"] = cluster_distances

        # Sort clusters by size
        real_clusters = sorted(real_clusters, key=lambda x: len(x[1]))

        added_clusters = []

        # Merge points from clusters
        new_terminal_id = group["terminal_id"].max() + 10
        for cluster_index, (
            real_cluster_label,
            real_cluster,
        ) in enumerate(real_clusters):

            # Let at least 4 points in the graph
            points_not_clustered = group_with_label.loc[
                ~group_with_label["cluster_label"].isin(
                    [i[0] for i in real_clusters[: cluster_index + 1]]
                )
            ]
            if len(points_not_clustered) + cluster_index + 1 <= 3:
                points_in_current_cluster = group_with_label.loc[
                    group_with_label["cluster_label"] == real_cluster_label
                ].sort_values("distance", ascending=False)
                removed_indices = points_in_current_cluster.index[
                    : max(0, 3 - len(points_not_clustered) - cluster_index)
                ]
                logger.warning(
                    f"{group_name}: not enought points, removing {removed_indices.tolist()} "
                    f"from the cluster {real_cluster_label}"
                )
                actual_cluster = points_in_current_cluster.loc[
                    points_in_current_cluster.index.difference(removed_indices)
                ]

                # Mark the points that will not be merged to keep at least 4 points in the graph
                group_with_label.loc[removed_indices, "cluster_label"] = (
                    group_with_label["cluster_label"].max() + np.arange(len(removed_indices)) + 1
                )
            else:
                actual_cluster = real_cluster

            cluster_center = actual_cluster[["x", "y", "z"]].mean().values

            # Add the merged point
            first_element = actual_cluster.iloc[0]
            new_terminal_points.append(
                [
                    first_element["morph_file"],
                    first_element["axon_id"],
                    new_terminal_id,
                ]
                + cluster_center.tolist()
            )
            new_terminal_id += 1
            added_clusters.append(real_cluster_label)

        # Add non merged points
        new_terminal_points.extend(
            group_with_label.loc[
                ~group_with_label["cluster_label"].isin(added_clusters), output_cols
            ].values.tolist()
        )

        return new_terminal_points, group["cluster_id"]

    @staticmethod
    def common_path(graph, nodes, source=None, shortest_paths=None):
        """Compute the common paths of the given nodes.

        Source should be given only if the graph if undirected.
        Shortest paths can be given if they were already computed before.

        .. warning:: The graph must have only one component.
        """
        if not isinstance(graph, nx.DiGraph) and source is None and shortest_paths is None:
            raise ValueError(
                "Either the source or the pre-computed shortest paths must be provided when using "
                "an undirected graph."
            )

        if shortest_paths is None:
            if isinstance(graph, nx.DiGraph):
                try:
                    sources = [k for k, v in graph.in_degree if v == 0]
                    if len(sources) > 1:
                        raise RuntimeError("Several roots found in the directed graph.")
                    source = sources[0]
                except IndexError:
                    raise RuntimeError("Could not find the root of the directed graph.")
            shortest_paths = nx.single_source_shortest_path(graph, source)

        # Compute the common ancestor
        common_nodes = set(shortest_paths[nodes[0]])
        for i in nodes[1:]:
            common_nodes.intersection_update(set(shortest_paths[i]))
        common_nodes = list(common_nodes)
        common_path = [i for i in shortest_paths[nodes[0]] if i in common_nodes]

        return common_path

    @staticmethod
    def nodes_to_terminals_mapping(graph, source=None, shortest_paths=None):
        if (source is None) == (shortest_paths is None):
            raise ValueError("At least 'source' or 'shortest_paths' must be given but not both.")
        elif shortest_paths is None:
            shortest_paths = nx.single_source_shortest_path(graph, source)
        node_to_terminals = defaultdict(set)
        for node_id, parent_ids in shortest_paths.items():
            if not graph.nodes[node_id]["is_terminal"]:
                continue
            for j in parent_ids:
                node_to_terminals[j].add(node_id)
        return node_to_terminals

    def clusters_from_sphere_parents(self, axon, axon_id, group_name, group, output_cols, _):
        """All parents up to the common ancestor must be inside the sphere to be merged."""
        if self.max_path_clustering_distance is None:
            self.max_path_clustering_distance = self.clustering_distance

        # Get the complete morphology
        new_terminal_points = []
        nodes, edges, directed_graph = neurite_to_graph(axon)
        graph = nx.Graph(directed_graph)
        terminal_ids = nodes.loc[nodes["is_terminal"]].index

        pair_paths = dict(i for i in nx.all_pairs_shortest_path(graph) if i[0] in terminal_ids)

        # Get the pairs of terminals closer to the given distance
        terminal_nodes = nodes.loc[terminal_ids, ["x", "y", "z"]]
        terminal_tree = KDTree(terminal_nodes.values)
        terminal_pairs = terminal_tree.query_pairs(self.clustering_distance)

        # Initialize cluster IDs
        nodes["cluster_id"] = -1

        # The root can not be part of a cluster so it is given a specific cluster ID
        nodes.loc[-1, "cluster_id"] = nodes["cluster_id"].max() + 1
        group.loc[group["terminal_id"] == 0, "cluster_id"] = nodes.loc[-1, "cluster_id"]

        # Check that the paths between each pair do not exceed the given distance
        for a, b in terminal_pairs:
            term_a = terminal_nodes.iloc[a].name
            term_b = terminal_nodes.iloc[b].name
            if term_a == -1 or term_b == -1:
                continue
            try:
                path = pair_paths[term_a][term_b]
            except KeyError:
                path = pair_paths[term_b][term_a]
            path_points = nodes.loc[path]

            if pdist(path_points[["x", "y", "z"]].values).max() > self.max_path_clustering_distance:
                # Skip if a point on the path exceeds the clustering distance
                continue

            # TODO: Do not cluster the terminals if they are in different regions?
            # Or if the path between them goes too far inside another region?

            # Add points to clusters
            term_a_cluster_id = nodes.loc[term_a, "cluster_id"]
            term_b_cluster_id = nodes.loc[term_b, "cluster_id"]

            # If term_a is already in a cluster
            if term_a_cluster_id != -1:
                # If term_b is also already in a cluster
                if term_b_cluster_id != -1:
                    # Transfert all terminals from the cluster of term_b to the one of term_a
                    nodes.loc[
                        nodes["cluster_id"] == term_b_cluster_id, "cluster_id"
                    ] = term_a_cluster_id
                else:
                    # Add term_b to the cluster of term_a
                    nodes.loc[term_b, "cluster_id"] = term_a_cluster_id
            else:
                # If term_b is already in a cluster
                if term_b_cluster_id != -1:
                    # Add term_a to the cluster of term_b
                    nodes.loc[term_a, "cluster_id"] = term_b_cluster_id
                else:
                    # Create new cluster
                    nodes.loc[[term_a, term_b], "cluster_id"] = nodes["cluster_id"].max() + 1

        # Create cluster IDs for not clustered terminals
        not_clustered_mask = (nodes["is_terminal"]) & (nodes["cluster_id"] == -1)
        nodes.loc[not_clustered_mask, "cluster_id"] = (
            nodes.loc[not_clustered_mask].reset_index(drop=True).index
            + nodes["cluster_id"].max()
            + 1
        )

        clusters = nodes.loc[nodes["is_terminal"]].groupby("cluster_id")
        sorted_clusters = sorted(
            [(k, g) for k, g in clusters], key=lambda x: x[1].size, reverse=True
        )

        new_terminal_id = 1
        paths_from_root = nx.single_source_shortest_path(graph, -1)
        node_to_terminals = self.nodes_to_terminals_mapping(graph, shortest_paths=paths_from_root)

        # Ensure there are at least 4 clusters
        for num_cluster, (cluster_id, cluster) in enumerate(sorted_clusters):
            # Compute the common ancestor
            common_path = self.common_path(
                directed_graph,
                cluster.index.tolist(),
                shortest_paths=paths_from_root,
            )
            common_ancestors = [common_path[-1]]

            # Compute the number of missing clusters
            missing_points = max(
                0, 3 - (len(sorted_clusters[num_cluster + 1 :]) + len(new_terminal_points))
            )
            is_root = -1 in cluster.index
            if is_root:
                # The root node can not be splitted
                missing_points = 0

            # Split the cluster if needed in order to get at least 3 points
            common_ancestor_ind = 0
            while 0 < len(common_ancestors) <= missing_points:
                if common_ancestor_ind >= len(common_ancestors):
                    break
                sub_ancestors = edges.loc[
                    edges["source"] == common_ancestors[common_ancestor_ind], "target"
                ].tolist()
                if sub_ancestors:
                    common_ancestors.pop(common_ancestor_ind)
                    common_ancestors.extend(sub_ancestors)
                else:
                    common_ancestor_ind += 1

            # Add the points of the cluster
            for common_ancestor in common_ancestors:
                terminals_with_current_ancestor = cluster.loc[
                    set(node_to_terminals[common_ancestor]).intersection(cluster.index)
                ]
                new_terminal_points.append(
                    [
                        group_name,
                        axon_id,
                        new_terminal_id if not is_root else 0,
                    ]
                    + terminals_with_current_ancestor[["x", "y", "z"]].mean().tolist()
                )
                if not is_root:
                    group.loc[
                        group["section_id"].isin(terminals_with_current_ancestor.index),
                        "cluster_id",
                    ] = new_terminal_id
                    new_terminal_id += 1

        return new_terminal_points, group["cluster_id"]

    def barcode_mins(self, barcode, nb_bins=100, threshold=0.1):
        """Compute min values of a barcode."""
        bin_centers, data = barcode_bin_centers(barcode, num_bins=nb_bins)

        # Gaussian kernel to smooth distribution of bars
        kde = stats.gaussian_kde(data)
        minimas = []

        # Compute first and second derivatives
        der1 = np.gradient(kde(bin_centers))
        der2 = np.gradient(der1)

        # Compute minimas of distribution
        while len(minimas) == 0:
            minimas = np.where(abs(der1) < threshold * np.max(abs(der1)))[0]
            minimas = minimas[der2[minimas] > 0]
            threshold *= 2.0  # if threshold was too small, increase and retry

        def _get_min_indices(mins, der):
            # Compute where the derivative crosses the X axis
            der_zero_intervals = np.append(
                np.insert(
                    np.where(np.diff(np.sign(der)))[0],
                    0,
                    -len(der) * 2,
                ),
                len(der) * 2,
            )

            # Find in which interval is each value
            zero_interval_indices = np.digitize(mins, der_zero_intervals)
            _tmp = pd.DataFrame(
                {
                    "min_indices": mins,
                    "interval_idx": zero_interval_indices,
                }
            ).groupby("interval_idx")

            # Get the median value
            return _tmp.quantile(interpolation="higher")["min_indices"].astype(int).values

        # Keep one minimum per der1 and der2 interval
        min_indices = _get_min_indices(minimas, der1)
        min_indices = _get_min_indices(min_indices, der2)
        min_positions = bin_centers[min_indices]

        return min_indices, min_positions, bin_centers, der1, der2

    def clusters_from_barcodes(self, group_name, group, output_cols, soma_center):
        """The points must be inside the ball to be merged."""
        new_terminal_points = []

        # Get the complete morphology
        morph = load_morphology(group_name)
        # axons = [i for i in neuron.neurites if i.type == NeuriteType.axon]

        neuron = load_neuron_from_morphio(group_name)
        origin = neuron.soma.get_center()
        nb_bins = 100

        for axon in neuron.axon:
            barcode, bars_to_points = tree_to_property_barcode(
                axon,
                # lambda tree: tree.get_point_path_distances(),
                lambda tree: tree.get_point_radial_distances(point=origin),
            )

            min_indices, min_positions, bin_centers, der1, der2 = self.barcode_mins(
                barcode,
                nb_bins,
            )

            # Plot
            if self.plot_debug:
                fig, (ax_barcode, ax_hist, ax_der) = plt.subplots(1, 3, figsize=(12, 9))

                # Plot barcode
                plt.sca(ax_barcode)
                plot_barcode(barcode, new_fig=False)
                ax_barcode.vlines(
                    min_positions,
                    0,
                    ax_barcode.get_ylim()[1],
                    color="red",
                    linestyle="--",
                    label="In-between tufts",
                )

                # Plot histograms
                plt.sca(ax_hist)
                hist_data_horizontal = histogram_horizontal(barcode, num_bins=nb_bins)
                ax_hist.plot(
                    hist_data_horizontal[0][:-1],
                    hist_data_horizontal[1],
                    color="orange",
                    alpha=0.7,
                    label="Histogram horizontal",
                )
                hist_data_stepped = histogram_stepped(barcode)
                ax_hist.plot(
                    hist_data_stepped[0][:-1],
                    hist_data_stepped[1],
                    color="blue",
                    alpha=0.7,
                    label="Histogram stepped",
                )
                ax_hist.vlines(
                    min_positions,
                    0,
                    np.max(np.concatenate([hist_data_horizontal[1], hist_data_stepped[1]])),
                    color="red",
                    linestyle="--",
                    label="In-between tufts",
                )
                ax_hist.legend()

                # Plot derivatives
                ax_der.plot(bin_centers, der1, color="orange", alpha=0.7, label="1st derivative")
                ax_der.plot(bin_centers, der2, color="blue", alpha=0.7, label="2nd derivative")
                ax_der.vlines(
                    min_positions,
                    np.min(np.concatenate([der1, der2])),
                    np.max(np.concatenate([der1, der2])),
                    color="red",
                    linestyle="--",
                    label="In-between tufts",
                )
                ax_der.legend()

                plt.show()

            group["radial_dist"] = np.linalg.norm(group[["x", "y", "z"]] - soma_center, axis=1)
            min_positions = np.append(
                np.insert(min_positions, 0, 0), group["radial_dist"].max() + 1
            )
            terminal_intervals = np.digitize(group["radial_dist"], min_positions)
            clusters = []

            for num_interval, interval in enumerate(zip(min_positions[:-1], min_positions[1:])):
                cluster_terminals = group.loc[terminal_intervals == num_interval + 1]
                terminal_parents = defaultdict(list)
                crossing_sections = set()
                for term_sec in cluster_terminals["section_id"].values:
                    for sec in morph.section(term_sec).iter(IterType.upstream):
                        if np.linalg.norm(sec.points[-1] - soma_center) < interval[0]:
                            break
                        elif np.linalg.norm(sec.points[0] - soma_center) <= interval[0]:
                            crossing_sections.add(sec.id)
                        terminal_parents[sec.id].append(term_sec)
                if not crossing_sections:
                    crossing_sections.add(min(terminal_parents.keys()))

                raise NotImplemented("This mode is not implemented yet.")
                for sec in crossing_sections:
                    print(sec)

        return new_terminal_points

    def run(self):
        terminals = pd.read_csv(self.terminals_path or self.input().path)

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
            logger.debug(f"{group_name}: {len(group)} points")

            # Load the morphology
            morph = load_morphology(group_name)
            axon_morph = Morphology(morph)
            for i in axon_morph.root_sections:
                if i.type != NeuriteType.axon:
                    axon_morph.delete_section(i)

            # Soma center
            soma_center = (
                soma_centers.loc[soma_centers["morph_file"] == group_name, ["x", "y", "z"]].values[0]
            )

            # Cluster each axon
            axons = get_axons(morph)

            for axon_id, axon in enumerate(axons):

                if self.clustering_mode == "sphere":
                    clustering_func = self.clusters_from_spheres
                elif self.clustering_mode == "sphere_parents":
                    clustering_func = self.clusters_from_sphere_parents
                elif self.clustering_mode == "barcode":
                    clustering_func = self.clusters_from_barcodes

                axon_group = group.loc[group["axon_id"] == axon_id]
                new_terminal_points, cluster_ids = clustering_func(
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

            logger.info(f"{group_name}: {len(new_terminal_points)} points after merge")

            cluster_df = pd.DataFrame(new_terminal_points, columns=output_cols)

            # Replace terminals by the cluster centers and create sections from common ancestors
            # to cluster centers
            nodes, edges, directed_graph = neurite_to_graph(axon)
            sections_to_add = defaultdict(list)
            kept_path = None
            shortest_paths = nx.single_source_shortest_path(directed_graph, -1)
            for (axon_id, cluster_id), cluster in group.groupby(["axon_id", "cluster_id"]):
                # Skip the root cluster
                if (cluster.cluster_id == 0).any():
                    continue

                # Compute the common ancestor of the nodes
                common_path = self.common_path(
                    directed_graph,
                    cluster["section_id"].tolist(),
                    shortest_paths=shortest_paths,
                )
                if len(cluster) == 1 and len(common_path) > 2:
                    common_ancestor_shift = -2
                else:
                    common_ancestor_shift = -1
                common_ancestor = common_path[common_ancestor_shift]
                common_section = morph.section(common_ancestor)

                if kept_path is None:
                    kept_path = set(common_path)
                else:
                    kept_path = kept_path.union(common_path)

                # Get the current tuft barcode
                tuft_sections = set(
                    [
                        j
                        for terminal_id, path in shortest_paths.items()
                        if terminal_id in set(cluster["section_id"])
                        for j in path
                    ]
                ).difference(common_path[:common_ancestor_shift])
                tuft_morph = Morphology(axon_morph)

                tuft_ancestor = tuft_morph.section(common_ancestor)

                for i in tuft_morph.sections:
                    if i.id not in tuft_sections:
                        if i is tuft_ancestor:
                            import pdb
                            pdb.set_trace()
                        tuft_morph.delete_section(i.morphio_section, recursive=False)

                for sec in list(tuft_ancestor.iter(IterType.upstream)):
                    if sec is tuft_ancestor:
                        continue
                    tuft_morph.delete_section(sec, recursive=False)

                # Compute tuft orientation
                cluster_center = cluster_df.loc[
                    (cluster_df["axon_id"] == axon_id)
                    & (cluster_df["terminal_id"] == cluster_id),
                    ["x", "y", "z"]
                ].values[0]
                tuft_orientation = cluster_center - tuft_ancestor.points[-1]
                tuft_orientation /= np.linalg.norm(tuft_orientation)

                # Resize the common section used as root (the root section is 1um)
                new_root_section = tuft_morph.root_sections[0]
                new_root_section.points = np.vstack([new_root_section.points[-1] - tuft_orientation, new_root_section.points[-1]])
                new_root_section.diameters = np.repeat(new_root_section.diameters[1], 2)

                # Compute the barcode
                tmd_axon = list(convert_morphio_trees(Morphology(tuft_morph)))[axon_id]
                tuft_barcode, _ = tree_to_property_barcode(
                    tmd_axon,
                    lambda tree: tree.get_point_path_distances(),
                    # lambda tree: tree.get_point_radial_distances(point=tuft_morph.soma.center),
                    property_class=PersistentAngles,
                )

                # Add tuft category data
                path_distance = sum([section_length(i.points) for i in common_section.iter(IterType.upstream)])
                radial_distance = np.linalg.norm(axons[axon_id].points[0, COLS.XYZ] - common_section.points[-1])
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
                    clustered_morph, "3d", line_width=4, title=f"Clustered {clustered_morph.name}"
                )

                # Create the figure from the traces
                fig = make_subplots(
                    cols=2,
                    specs=[[{"is_3d": True}, {"is_3d": True}]],
                    subplot_titles=("Node clusters", "Clustered morphology")
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

                ax = plt.scatter(
                    x=cluster_props_df["radial_distance"],
                    y=(
                        cluster_props_df["cluster_center_coords"].apply(np.array)
                        - cluster_props_df["common_ancestor_coords"]
                    ).apply(np.linalg.norm)
                ).get_figure().gca()
                ax.set_title("Cluster radial length vs radial distance")
                pdf.savefig()
                plt.close()


        # Export the terminals
        new_terminals = pd.DataFrame(all_terminal_points, columns=output_cols)
        new_terminals = pd.merge(
            new_terminals,
            terminals.groupby(["morph_file", "axon_id", "cluster_id"]).size().rename("cluster_size"),
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
            "tuft_properties": ClusteringOutputLocalTarget("tuft_properties.json", create_parent=True),
        }
