"""Cluster the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import logging
import time
from pathlib import Path

import luigi
import luigi_tools
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from neurom import NeuriteType
from neurom import load_morphology
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

from PCSF.extract_terminals import ExtractTerminals

logger = logging.getLogger(__name__)


class ClusterTerminals(luigi_tools.task.WorkflowTask):
    terminals_path = luigi.Parameter(description="Path to the terminals CSV file.", default=None)
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
    clustering_number = luigi.NumericalParameter(
        description="The min number of points to define a cluster.",
        var_type=int,
        default=20,
        min_value=1,
        max_value=float("inf"),
    )
    clustering_mode = luigi.ChoiceParameter(
        description="The method used to define a cluster.",
        choices=["sphere", "sphere_parents"],
        default="",
    )
    plot_debug = luigi.BoolParameter(
        description=(
            "If set to True, each group will create an interactive figure so it is possible to "
            "check the clustering parameters."
        )
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
            cluster_center = cluster[["x", "y", "z"]].mean().values
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
            real_clusters.append((cluster_label, cluster, cluster_center))
            group_with_label.loc[cluster_indices, "distance"] = cluster_distances

        # Sort clusters by size
        real_clusters = sorted(real_clusters, key=lambda x: len(x[1]))

        added_clusters = []

        # Merge points from clusters
        new_terminal_id = group["terminal_id"].max() + 10
        for cluster_index, (
            real_cluster_label,
            real_cluster,
            cluster_center,
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
            digraph = nx.from_pandas_edgelist(edges, create_using=nx.DiGraph)
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

                if pdist(path_points[["x", "y", "z"]].values).max() > self.clustering_distance:
                    # Continue if a point on the path exceeds the clustering distance
                    continue

                # Add points to clusters
                term_a_cluster_id = nodes.loc[term_a, "cluster_id"]
                term_b_cluster_id = nodes.loc[term_b, "cluster_id"]
                if term_a_cluster_id != -1:
                    # If term_a is already in a cluster
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

            for num_cluster, (cluster_id, cluster) in enumerate(sorted_clusters):
                # Compute the common ancestor
                base_path = set(paths_from_root[cluster.index[0]])
                for i in cluster.index[1:]:
                    base_path.intersection_update(set(paths_from_root[i]))
                base_path = list(base_path)
                common_ancestors = [base_path[np.argmax([lengths_from_root[i] for i in base_path])]]

                # Split the cluster if needed in order to get at least 4 points
                missing_points = max(
                    0, 3 - (len(sorted_clusters[num_cluster + 1 :]) + len(new_terminal_points))
                )
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

                # Add points
                for common_ancestor in common_ancestors:
                    new_terminal_points.append(
                        [
                            group_name,
                            axon_id,
                            new_terminal_id,
                        ]
                        + nodes.loc[common_ancestor, ["x", "y", "z"]].tolist()
                    )
                    new_terminal_id += 1

        return new_terminal_points

    def run(self):
        terminals = pd.read_csv(self.terminals_path or self.input().path)
        output_file = Path(self.output().path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        all_terminal_points = []
        output_cols = ["morph_file", "axon_id", "terminal_id", "x", "y", "z"]

        # Drop soma terminals
        soma_centers = (terminals["axon_id"] == -1)
        all_terminal_points.extend(
            terminals.loc[soma_centers, output_cols].to_records(index=False).tolist()
        )
        terminals.drop(terminals.loc[soma_centers].index, inplace=True)

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

            all_terminal_points.extend(new_terminal_points)

            logger.info(f"{group_name}: {len(new_terminal_points)} points after merge")

            if self.plot_debug:
                plot_df = pd.DataFrame(new_terminal_points, columns=output_cols)
                fig = plt.figure(figsize=(12, 9))
                ax = fig.gca(projection="3d")
                x, y, z = group[["x", "y", "z"]].values.T
                ax.scatter(x, y, z, c="black", s=5)
                x, y, z = plot_df[["x", "y", "z"]].values.T
                ax.scatter(x, y, z, c="red", s=2)
                plt.show()
                time.sleep(1)

        if self.plot_debug:
            matplotlib.use(old_backend)

        # Export the terminals
        new_terminals = pd.DataFrame(all_terminal_points, columns=output_cols)
        new_terminals.to_csv(output_file, index=False)

    def output(self):
        return luigi_tools.target.OutputLocalTarget(self.output_dataset)
