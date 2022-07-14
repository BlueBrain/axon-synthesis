"""Some clustering utils."""
import networkx as nx
from morphio.mut import Morphology as MorphIoMorphology
from tmd.io.conversion import convert_morphio_trees
from tmd.Topology.methods import tree_to_property_barcode
from tmd.Topology.persistent_properties import PersistentAngles


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


def get_barcode(morph, metric="path_distance", tree_index=0):
    """Compute the barcode of the given morphology."""
    tmd_axon = list(convert_morphio_trees(MorphIoMorphology(morph).as_immutable()))[tree_index]
    tuft_barcode, _ = tree_to_property_barcode(
        tmd_axon,
        lambda tree: tree.get_point_path_distances()
        if metric == "path_distance"
        else tree.get_point_radial_distances(point=morph.soma.center),
        property_class=PersistentAngles,
    )
