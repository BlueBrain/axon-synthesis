"""Cluster the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import sys
from pathlib import Path

import luigi
import luigi_tools
import matplotlib
import neurom
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from neurom import load_neuron
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from PCSF.extract_terminals import ExtractTerminals


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
        left_op=luigi.parameter.operator.lt,
    )
    clustering_number = luigi.NumericalParameter(
        description="The min number of points to define a cluster.",
        var_type=int,
        default=20,
        min_value=0,
        max_value=float("inf"),
        left_op=luigi.parameter.operator.lt,
    )
    plot_debug = luigi.BoolParameter(
        description=(
            "If set to True, each group will create aa interactive figure so it is possible to "
            "check the clustering parameters."
        )
    )

    def requires(self):
        return ExtractTerminals()

    def run(self):
        terminals = pd.read_csv(self.terminals_path or self.input().path)
        output_file = Path(self.output().path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        new_terminal_points = []

        if self.plot_debug:
            old_backend = matplotlib.get_backend()
            matplotlib.use("TkAgg")

        for group_name, group in terminals.groupby("morph_file"):
            # Get the pairs of points closer to the given distance
            tree = KDTree(group[["x", "y", "z"]].values)
            pairs = tree.query_pairs(self.clustering_distance)

            # Get the connected components
            adjacency_matrix = np.zeros((len(group), len(group)))
            adjacency_matrix[tuple(np.array(list(pairs)).T.tolist())] = 1
            graph = csr_matrix(adjacency_matrix)
            n_components, labels = connected_components(
                csgraph=graph, directed=False, return_labels=True
            )

            # Define clusters from these components
            cluster_labels, cluster_sizes = np.unique(labels, return_counts=True)
            big_clusters = cluster_labels[cluster_sizes >= self.clustering_number]
            group_with_label = group.copy()
            group_with_label["cluster_label"] = labels
            clusters = group_with_label.loc[
                group_with_label["cluster_label"].isin(big_clusters)
            ].groupby("cluster_label")

            # Merge points from clusters
            new_terminal_id = group["terminal_id"].max() + 10
            real_clusters = []
            for cluster_label, cluster in clusters:
                # Check that the potential cluster is a real one (most of the points must be close
                # to the center)
                cluster_center = cluster[["x", "y", "z"]].mean().values
                distances, indices = tree.query(
                    cluster[["x", "y", "z"]].mean().values,
                    k=min(self.clustering_number, len(cluster)),
                    distance_upper_bound=self.clustering_distance,
                )
                cluster_indices = indices[~np.isinf(distances)]
                if len(cluster_indices) < self.clustering_number:
                    continue

                # Mark the cluster as a real one
                real_clusters.append(cluster_label)

                # Add the point
                first_index = cluster.index[0]
                new_terminal_points.append(
                    [
                        cluster.loc[first_index, "morph_file"],
                        cluster.loc[first_index, "axon_id"],
                        new_terminal_id,
                    ]
                    + cluster_center.tolist()
                )
                new_terminal_id += 1

            # Add non merged points
            new_terminal_points.extend(
                group.loc[~group_with_label["cluster_label"].isin(real_clusters)].values.tolist()
            )

            if self.plot_debug:
                fig = plt.figure(figsize=(12, 9))
                ax = fig.gca(projection="3d")
                x, y, z, colors = group_with_label[["x", "y", "z", "cluster_label"]].values.T
                ax.scatter(x, y, z, c=colors)
                plt.show()

        if self.plot_debug:
            matplotlib.use(old_backend)

        # Export the terminals
        new_terminals = pd.DataFrame(new_terminal_points, columns=terminals.columns)
        new_terminals.to_csv(output_file, index=False)

    def output(self):
        return luigi.LocalTarget(self.output_dataset)
