"""Cluster the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import logging
import time
from collections import defaultdict
from pathlib import Path

import luigi
import luigi_tools
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from morphio import IterType
from morph_tool import resampling
from neurom import NeuriteType
from neurom import load_morphology
from neurom.core import Morphology
from plotly_helper.neuron_viewer import NeuronBuilder
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from tmd.io.io import load_neuron_from_morphio
from tmd.Topology.analysis import barcode_bin_centers
from tmd.Topology.analysis import histogram_horizontal
from tmd.Topology.analysis import histogram_stepped
from tmd.Topology.methods import tree_to_property_barcode
from tmd.view.plot import barcode as plot_barcode

from PCSF.extract_terminals import ExtractTerminals

logger = logging.getLogger(__name__)


class ClusterTerminals(luigi_tools.task.WorkflowTask):
    terminals_path = luigi_tools.parameter.OptionalPathParameter(
        description="Path to the terminals CSV file.",
        default=None,
        exists=True,
    )
    output_dataset = luigi.Parameter(
        description="Output dataset file.", default="clustered_terminals.csv"
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
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING
    )

    def requires(self):
        return ExtractTerminals()

    def clusters_from_spheres(self, group_name, group, output_cols):
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

        return new_terminal_points

    def clusters_from_sphere_parents(self, group_name, group, output_cols):
        """All parents up to the common ancestor must be inside the sphere to be merged."""
        if self.max_path_clustering_distance is None:
            self.max_path_clustering_distance = self.clustering_distance

        # Get the complete morphology
        neuron = load_morphology(group_name)
        axons = [i for i in neuron.neurites if i.type == NeuriteType.axon]
        new_terminal_points = []
        # if group_name == "out_curated/CheckNeurites/data/AA0411.asc":
        #     import pdb
        #     pdb.set_trace()
        # else:
        #     return []
        for axon_id, axon in enumerate(axons):
            graph_nodes = []
            graph_edges = []
            for section in axon.iter_sections():
                is_terminal = not bool(section.children)
                if section.parent is None:
                    graph_nodes.append((-1, *section.points[0, :3], True))
                    graph_edges.append((-1, section.id))

                graph_nodes.append((section.id, *section.points[-1, :3], is_terminal))

                for child in section.children:
                    graph_edges.append((section.id, child.id))

            nodes = pd.DataFrame(graph_nodes, columns=["id", "x", "y", "z", "is_terminal"])
            nodes.set_index("id", inplace=True)
            terminal_ids = nodes.loc[nodes["is_terminal"]].index

            edges = pd.DataFrame(graph_edges, columns=["source", "target"])
            graph = nx.from_pandas_edgelist(edges)
            pair_paths = dict(i for i in nx.all_pairs_shortest_path(graph) if i[0] in terminal_ids)

            # Get the pairs of terminals closer to the given distance
            terminal_nodes = nodes.loc[terminal_ids, ["x", "y", "z"]]
            terminal_tree = KDTree(terminal_nodes.values)
            terminal_pairs = terminal_tree.query_pairs(self.clustering_distance)

            # Check that the paths between each pair do not exceed the given distance
            nodes["cluster_id"] = -1
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

            new_terminal_id = 0
            paths_from_root = nx.single_source_shortest_path(graph, -1)
            lengths_from_root = nx.single_source_shortest_path_length(graph, -1)
            point_to_terminals = defaultdict(list)
            for point_id, parent_ids in paths_from_root.items():
                if not nodes.loc[point_id, "is_terminal"]:
                    continue
                for j in parent_ids:
                    point_to_terminals[j].append(point_id)

            # Ensure there are at least 4 clusters
            for num_cluster, (cluster_id, cluster) in enumerate(sorted_clusters):
                # Compute the common ancestor
                base_path = set(paths_from_root[cluster.index[0]])
                for i in cluster.index[1:]:
                    base_path.intersection_update(set(paths_from_root[i]))
                base_path = list(base_path)
                common_ancestors = [base_path[np.argmax([lengths_from_root[i] for i in base_path])]]

                # Compute the number of missing clusters
                missing_points = max(
                    0, 3 - (len(sorted_clusters[num_cluster + 1 :]) + len(new_terminal_points))
                )

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
                    terminals_with_current_ancestor = cluster.loc[set(point_to_terminals[common_ancestor]).intersection(cluster.index)]
                    new_terminal_points.append(
                        [
                            group_name,
                            axon_id,
                            new_terminal_id,
                        ]
                        + terminals_with_current_ancestor[["x", "y", "z"]].mean().tolist()
                    )
                    group.loc[group["section_id"].isin(terminals_with_current_ancestor.index), "cluster_id"] = new_terminal_id
                    new_terminal_id += 1

        return new_terminal_points

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
            threshold *= 2.  # if threshold was too small, increase and retry

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

            min_indices, min_positions, bin_centers, der1, der2 = self.barcode_mins(barcode, nb_bins)

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

            soma_pos = np.array([soma_center["x"], soma_center["y"], soma_center["z"]])
            group["radial_dist"] = np.linalg.norm(
                group[["x", "y", "z"]] - soma_pos,
                axis=1
            )
            min_positions = np.append(np.insert(min_positions, 0, 0), group["radial_dist"].max() + 1)
            terminal_intervals = np.digitize(group["radial_dist"], min_positions)
            clusters = []

            for num_interval, interval in enumerate(zip(min_positions[:-1], min_positions[1:])):
                cluster_terminals = group.loc[terminal_intervals == num_interval + 1]
                terminal_parents = defaultdict(list)
                crossing_sections = set()
                for term_sec in cluster_terminals["section_id"].values:
                    for sec in morph.section(term_sec).iter(IterType.upstream):
                        if np.linalg.norm(sec.points[-1] - soma_pos) < interval[0]:
                            break
                        elif np.linalg.norm(sec.points[0] - soma_pos) <= interval[0]:
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

        all_terminal_points = []
        output_cols = ["morph_file", "axon_id", "terminal_id", "x", "y", "z"]

        # Drop soma terminals
        soma_centers_mask = (terminals["axon_id"] == -1)
        soma_centers = terminals.loc[soma_centers_mask].copy()
        all_terminal_points.extend(
            soma_centers[output_cols].to_records(index=False).tolist()
        )
        terminals.drop(soma_centers.index, inplace=True)
        terminals["cluster_id"] = -1

        if self.plot_debug:
            old_backend = matplotlib.get_backend()
            matplotlib.use("TkAgg")

        for group_name, group in terminals.groupby("morph_file"):
            logger.debug(f"{group_name}: {len(group)} points")

            if self.clustering_mode == "sphere":
                new_terminal_points = self.clusters_from_spheres(group_name, group, output_cols)
            elif self.clustering_mode == "sphere_parents":
                new_terminal_points = self.clusters_from_sphere_parents(
                    group_name, group, output_cols
                )
            elif self.clustering_mode == "barcode":
                new_terminal_points = self.clusters_from_barcodes(
                    group_name,
                    group,
                    output_cols,
                    soma_centers.loc[soma_centers["morph_file"] == group_name].to_dict("records")[0],
                )

            all_terminal_points.extend(new_terminal_points)

            logger.info(f"{group_name}: {len(new_terminal_points)} points after merge")

            if self.plot_debug:
                plot_df = pd.DataFrame(new_terminal_points, columns=output_cols)

                neuron = load_morphology(group_name)
                neuron = Morphology(
                    resampling.resample_linear_density(neuron, 0.001),
                    name=Path(group_name).with_suffix("").name,
                )
                fig_builder = NeuronBuilder(neuron, "3d", line_width=4, title=f"{neuron.name}")

                x, y, z = group[["x", "y", "z"]].values.T
                node_trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker={"size": 3, "color": "black"},
                    name="Morphology nodes",
                )
                x, y, z = plot_df[["x", "y", "z"]].values.T
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
                        [i["x"], plot_df.loc[plot_df["terminal_id"] == i["cluster_id"], "x"].iloc[0], None],
                        [i["y"], plot_df.loc[plot_df["terminal_id"] == i["cluster_id"], "y"].iloc[0], None],
                        [i["z"], plot_df.loc[plot_df["terminal_id"] == i["cluster_id"], "z"].iloc[0], None],
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
                fig = go.Figure()
                fig.add_traces(fig_builder.get_figure()["data"])
                fig.add_trace(node_trace)
                fig.add_trace(cluster_trace)
                fig.add_trace(edge_trace)

                # Export figure
                filepath = self.output().pathlib_path.parent / f"clustering/{Path(group_name).with_suffix('').name}.html"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(filepath))

        if self.plot_debug:
            matplotlib.use(old_backend)

        # Export the terminals
        new_terminals = pd.DataFrame(all_terminal_points, columns=output_cols)
        new_terminals.to_csv(self.output().path, index=False)

    def output(self):
        return luigi_tools.target.OutputLocalTarget(self.output_dataset, create_parent=True)
