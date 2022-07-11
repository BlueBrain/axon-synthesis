"""Some clustering utils."""
import networkx as nx


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
                # pylint: disable=raise-missing-from
                raise RuntimeError("Could not find the root of the directed graph.")
        shortest_paths = nx.single_source_shortest_path(graph, source)

    # Compute the common ancestor
    common_nodes = set(shortest_paths[nodes[0]])
    for i in nodes[1:]:
        common_nodes.intersection_update(set(shortest_paths[i]))
    common_nodes = list(common_nodes)
    shortest_common_path = [i for i in shortest_paths[nodes[0]] if i in common_nodes]

    return shortest_common_path
