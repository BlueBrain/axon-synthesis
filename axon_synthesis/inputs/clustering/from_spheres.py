"""Clustering from spheres."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import logging

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from axon_synthesis.utils import COORDS_COLS

logger = logging.getLogger(__name__)


def compute_clusters(
    config, config_name, axon_id, group_name, group_path, group, output_cols, **_kwargs
):
    """The points must be inside the ball to be merged."""
    # pylint: disable=too-many-locals
    new_terminal_points = []

    # Get the pairs of terminals closer to the given distance
    tree = KDTree(group[COORDS_COLS].to_numpy())
    pairs = tree.query_pairs(config["sphere_radius"])

    # Get the connected components
    adjacency_matrix = np.zeros((len(group), len(group)))
    if pairs:
        adjacency_matrix[tuple(np.array(list(pairs)).T.tolist())] = 1
    np.fill_diagonal(adjacency_matrix, 1)
    graph = csr_matrix(adjacency_matrix)
    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    # Define clusters from these components
    cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    big_clusters = cluster_ids[cluster_sizes >= config["min_size"]]
    group_with_label = group.reset_index()
    group_with_label["tuft_id"] = labels
    group_with_label["distance"] = -1.0
    group_with_label["config_name"] = config_name
    clusters = group_with_label.loc[group_with_label["tuft_id"].isin(big_clusters)].groupby(
        "tuft_id",
    )

    # Check clusters
    real_clusters = []
    for cluster_id, cluster in clusters:
        # Check that the potential cluster is a real one (at least 'min_size'
        # points must be close to the center)
        distances, indices = tree.query(
            cluster[COORDS_COLS].mean().to_numpy(),
            k=len(group),
        )
        cluster_mask = np.isin(indices, cluster.index)
        cluster_indices = indices[cluster_mask]
        cluster_distances = distances[cluster_mask]
        if np.count_nonzero(cluster_distances <= config["sphere_radius"]) < config["min_size"]:
            continue

        # Mark the cluster as a real one
        real_clusters.append((cluster_id, cluster))
        group_with_label.loc[cluster_indices, "distance"] = cluster_distances

    # Sort clusters by size
    real_clusters = sorted(real_clusters, key=lambda x: len(x[1]))

    added_clusters = []

    # Merge points from clusters
    for cluster_index, (
        real_cluster_id,
        real_cluster,
    ) in enumerate(real_clusters):
        # Let at least 4 points in the graph
        points_not_clustered = group_with_label.loc[
            ~group_with_label["tuft_id"].isin([i[0] for i in real_clusters[: cluster_index + 1]])
        ]
        if len(points_not_clustered) + cluster_index + 1 <= 3:
            points_in_current_cluster = group_with_label.loc[
                group_with_label["tuft_id"] == real_cluster_id
            ].sort_values("distance", ascending=False)
            removed_indices = points_in_current_cluster.index[
                : max(0, 3 - len(points_not_clustered) - cluster_index)
            ]
            logger.warning(
                "%s: not enough points, removing %s from the cluster %s",
                group_name,
                removed_indices.tolist(),
                real_cluster_id,
            )
            actual_cluster = points_in_current_cluster.loc[
                points_in_current_cluster.index.difference(removed_indices)
            ]

            # Mark the points that will not be merged to keep at least 4 points in the graph
            group_with_label.loc[removed_indices, "tuft_id"] = (
                group_with_label["tuft_id"].max() + np.arange(len(removed_indices)) + 1
            )
        else:
            actual_cluster = real_cluster

        cluster_center = actual_cluster[COORDS_COLS].mean().to_numpy()

        # Add the merged point
        new_terminal_points.append(
            [
                group_name,
                group_path,
                config_name,
                axon_id,
                real_cluster_id,
                len(actual_cluster),
                *cluster_center.tolist(),
            ]
        )
        added_clusters.append(real_cluster_id)

    # Add non merged points
    not_added_mask = ~group_with_label["tuft_id"].isin(added_clusters)
    group_with_label.loc[not_added_mask, "tuft_id"] = sorted(
        set(range(len(added_clusters) + len(group_with_label.loc[not_added_mask]))).difference(
            added_clusters,
        ),
    )
    group_with_label.loc[not_added_mask, "terminal_id"] = group_with_label.loc[
        not_added_mask,
        "tuft_id",
    ]
    group_with_label["size"] = 1
    new_terminal_points.extend(
        group_with_label.loc[not_added_mask, output_cols].to_numpy().tolist()
    )

    return new_terminal_points, group_with_label.set_index("index")["tuft_id"]
