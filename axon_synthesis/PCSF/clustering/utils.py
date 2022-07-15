"""Some utils for clustering."""
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
from morphio import IterType
from morphio import PointLevel
from morphio.mut import Morphology as MorphIoMorphology
from neurom import COLS
from neurom import NeuriteType
from neurom.core import Morphology
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


def create_tuft_morphology(
    morph, tuft_section_ids, common_ancestor, cluster_common_path, shortest_paths
):
    """Create a new morphology containing only the given tuft."""
    tuft_morph = Morphology(morph)
    for i in tuft_morph.root_sections:
        if i.type != NeuriteType.axon:
            tuft_morph.delete_section(i)

    tuft_sections = set(
        j
        for terminal_id, path in shortest_paths.items()
        if terminal_id in tuft_section_ids
        for j in path
    ).difference(cluster_common_path)

    tuft_ancestor = tuft_morph.section(common_ancestor)

    for i in tuft_morph.sections:
        if i.id not in tuft_sections:
            tuft_morph.delete_section(i.morphio_section, recursive=False)

    for sec in list(tuft_ancestor.iter(IterType.upstream)):
        if sec is tuft_ancestor:
            continue
        tuft_morph.delete_section(sec, recursive=False)

    return tuft_morph, tuft_ancestor


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
    return tuft_barcode


def resize_root_section(tuft_morph, tuft_orientation, root_section_idx=0):
    """Resize the root section to 1um."""
    new_root_section = tuft_morph.root_sections[root_section_idx]
    new_root_section.points = np.vstack(
        [
            new_root_section.points[-1] - tuft_orientation,
            new_root_section.points[-1],
        ]
    )
    new_root_section.diameters = np.repeat(new_root_section.diameters[1], 2)


def create_clustered_morphology(morph, group_name, kept_path, sections_to_add):
    """Create a new morphology with the kept path and add new sections to cluster centers."""
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
    return clustered_morph
