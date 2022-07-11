"""Cluster the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import logging
import time
from pathlib import Path

import luigi
import luigi_tools
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

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
            "If set to True, each group will create an interactive figure so it is possible to "
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
        output_cols = ["morph_file", "axon_id", "terminal_id", "x", "y", "z"]

        if self.plot_debug:
            old_backend = matplotlib.get_backend()
            matplotlib.use("TkAgg")

        for group_name, group in terminals.groupby("morph_file"):
            logger.debug(f"{group_name}: {len(group)} points")

            # Get the pairs of points closer to the given distance
            tree = KDTree(group[["x", "y", "z"]].values)
            pairs = tree.query_pairs(self.clustering_distance)

            # Get the connected components
            adjacency_matrix = np.zeros((len(group), len(group)))
            adjacency_matrix[tuple(np.array(list(pairs)).T.tolist())] = 1
            graph = csr_matrix(adjacency_matrix)
            _, labels = connected_components(
                csgraph=graph, directed=False, return_labels=True
            )

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
                _,
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
                        group_with_label["cluster_label"].max()
                        + np.arange(len(removed_indices))
                        + 1
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

            logger.info(
                f"{group_name}: "
                f"""{
                    len(real_clusters) + len(
                        group_with_label.loc[
                            ~group_with_label["cluster_label"].isin(added_clusters)
                        ]
                    )
                } points after merge"""
            )

            if self.plot_debug:
                fig = plt.figure(figsize=(12, 9))
                ax = fig.gca(projection="3d")
                x, y, z, colors = group_with_label[["x", "y", "z", "cluster_label"]].values.T
                ax.scatter(x, y, z, c=colors, cmap="prism")
                plt.show()
                time.sleep(1)

        if self.plot_debug:
            matplotlib.use(old_backend)

        # Export the terminals
        new_terminals = pd.DataFrame(new_terminal_points, columns=output_cols)
        new_terminals.to_csv(output_file, index=False)

    def output(self):
        return luigi.LocalTarget(self.output_dataset)
