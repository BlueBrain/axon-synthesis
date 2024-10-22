"""Clustering from sphere parents."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import networkx as nx
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

from axon_synthesis.inputs.clustering.utils import common_path
from axon_synthesis.utils import COORDS_COLS


def _check_cluster(a, b, nodes, terminal_nodes, pair_paths, cluster_ids, max_path_distance) -> None:
    """Check that 2 points are in the same cluster."""
    term_a = terminal_nodes.iloc[a].name
    term_b = terminal_nodes.iloc[b].name
    if term_a == -1 or term_b == -1:
        return
    try:
        path = pair_paths[term_a][term_b]
    except KeyError:
        path = pair_paths[term_b][term_a]

    if pdist(nodes.loc[path, COORDS_COLS].to_numpy()).max() > max_path_distance:
        # Skip if a point on the path exceeds the clustering distance
        return

    # TODO: Do not cluster the terminals if they are in different regions?
    # Or if the path between them goes too far inside another region?

    # Add points to clusters
    term_a_cluster_id = cluster_ids.loc[term_a]
    term_b_cluster_id = cluster_ids.loc[term_b]

    # If term_a is already in a cluster
    if term_a_cluster_id != -1:
        # If term_b is also already in a cluster
        if term_b_cluster_id != -1:
            # Transfer all terminals from the cluster of term_b to the one of term_a
            cluster_ids.loc[cluster_ids == term_b_cluster_id] = term_a_cluster_id
        else:
            # Add term_b to the cluster of term_a
            cluster_ids.loc[term_b] = term_a_cluster_id
    else:  # noqa: PLR5501
        # If term_b is already in a cluster
        if term_b_cluster_id != -1:
            # Add term_a to the cluster of term_b
            cluster_ids.loc[term_a] = term_b_cluster_id
        else:
            # Create new cluster
            cluster_ids.loc[[term_a, term_b]] = cluster_ids.max() + 1


def compute_clusters(  # noqa: PLR0913
    config,
    config_name,
    axon_id,
    group_name,
    group_path,
    group,
    nodes,
    edges,
    directed_graph,
    shortest_paths,
    node_to_terminals,
    **_kwargs,
):
    """All parents up to the common ancestor must be inside the sphere to be merged."""
    sphere_radius = config["sphere_radius"]
    max_path_distance = config.get("max_path_distance", sphere_radius)

    # Get the complete morphology
    new_terminal_points: list[list] = []

    graph: nx.Graph = nx.Graph(directed_graph)
    terminals = nodes.loc[nodes["is_terminal"]]

    pair_paths = dict(i for i in nx.all_pairs_shortest_path(graph) if i[0] in set(terminals.index))

    # Get the pairs of terminals closer to the given distance
    terminal_tree = KDTree(terminals[COORDS_COLS].to_numpy())
    terminal_pairs = terminal_tree.query_pairs(sphere_radius)

    # Initialize cluster IDs
    cluster_ids = pd.Series(-1, index=nodes.index.to_numpy())

    # The root can not be part of a cluster so it is given a specific cluster ID
    cluster_ids.loc[-1] = cluster_ids.max() + 1  # type: ignore[call-overload]
    group.loc[group["terminal_id"] == 0, "tuft_id"] = cluster_ids.loc[-1]

    # Check that the paths between each pair do not exceed the given distance
    for a, b in terminal_pairs:
        _check_cluster(a, b, nodes, terminals, pair_paths, cluster_ids, max_path_distance)

    # Create cluster IDs for not clustered terminals
    not_clustered_mask = (nodes["is_terminal"]) & (cluster_ids == -1)
    cluster_ids.loc[not_clustered_mask] = (
        cluster_ids.loc[not_clustered_mask].reset_index(drop=True).index + cluster_ids.max() + 1
    )

    # Reset cluster IDs to consecutive values
    nodes["tuft_id"] = cluster_ids.map(
        {v: k for k, v in enumerate(sorted(cluster_ids.unique()), start=-1)},
    )

    # Groupby points by cluster IDs
    clusters = nodes.loc[nodes["is_terminal"]].groupby("tuft_id")
    sorted_clusters = sorted(clusters, key=lambda x: x[1].size, reverse=True)

    new_terminal_id = 1

    # Ensure there are at least 4 clusters
    for num_cluster, (_, cluster) in enumerate(sorted_clusters):
        # Compute the common ancestor
        cluster_common_path = common_path(
            directed_graph,
            cluster.index.to_list(),
            shortest_paths=shortest_paths,
        )
        common_ancestors = [cluster_common_path[-1]]

        # Compute the number of missing clusters
        missing_points = max(
            0,
            3 - (len(sorted_clusters[num_cluster + 1 :]) + len(new_terminal_points)),
        )
        is_root = -1 in cluster.index
        if is_root:
            # The root node can not be split
            missing_points = 0

        # Split the cluster if needed in order to get at least 3 points
        common_ancestor_ind = 0
        while 0 < len(common_ancestors) <= missing_points:
            if common_ancestor_ind >= len(common_ancestors):
                break
            sub_ancestors = edges.loc[
                edges["source"] == common_ancestors[common_ancestor_ind],
                "target",
            ].tolist()
            if sub_ancestors:
                common_ancestors.pop(common_ancestor_ind)
                common_ancestors.extend(sub_ancestors)
            else:
                common_ancestor_ind += 1

        # Add the points of the cluster
        for common_ancestor in common_ancestors:
            terminals_with_current_ancestor = cluster.loc[
                list(set(node_to_terminals[common_ancestor]).intersection(cluster.index))
            ].sort_index()
            new_terminal_points.append(
                [
                    group_name,
                    group_path,
                    config_name,
                    axon_id,
                    new_terminal_id if not is_root else 0,
                    len(terminals_with_current_ancestor),
                    *terminals_with_current_ancestor[COORDS_COLS].mean().tolist(),
                ],
            )
            if not is_root:
                group.loc[
                    group["graph_node_id"].isin(terminals_with_current_ancestor.index),
                    "tuft_id",
                ] = new_terminal_id
                new_terminal_id += 1

    return new_terminal_points, group["tuft_id"]
