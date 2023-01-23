"""Clustering from spheres."""
import json
import logging

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def compute_clusters(task, config, axon, axon_id, group_name, group, output_cols, soma_center):
    """The points must be inside the ball to be merged."""
    # pylint: disable=too-many-locals
    # pylint: disable=unused-argument
    new_terminal_points = []

    config_str = json.dumps(config)

    # Get the pairs of terminals closer to the given distance
    tree = KDTree(group[["x", "y", "z"]].values)
    pairs = tree.query_pairs(config["clustering_distance"])

    # Get the connected components
    adjacency_matrix = np.zeros((len(group), len(group)))
    if pairs:
        adjacency_matrix[tuple(np.array(list(pairs)).T.tolist())] = 1
    np.fill_diagonal(adjacency_matrix, 1)
    graph = csr_matrix(adjacency_matrix)
    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    # Define clusters from these components
    cluster_labels, cluster_sizes = np.unique(labels, return_counts=True)
    big_clusters = cluster_labels[cluster_sizes >= config["clustering_number"]]
    group_with_label = group.reset_index()
    group_with_label["cluster_label"] = labels
    group_with_label["distance"] = -1.0
    clusters = group_with_label.loc[group_with_label["cluster_label"].isin(big_clusters)].groupby(
        "cluster_label"
    )

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
            np.count_nonzero(cluster_distances <= config["clustering_distance"])
            < config["clustering_number"]
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
                "%s: not enough points, removing %s from the cluster %s",
                group_name,
                removed_indices.tolist(),
                real_cluster_label,
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
            + [config_str]
        )
        new_terminal_id += 1
        added_clusters.append(real_cluster_label)

    # Add non merged points
    new_terminal_points.extend(
        group_with_label.loc[
            ~group_with_label["cluster_label"].isin(added_clusters), output_cols
        ].values.tolist()
    )

    return new_terminal_points, group["cluster_id"], []
