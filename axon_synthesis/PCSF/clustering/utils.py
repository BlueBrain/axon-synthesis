"""Some utils for clustering."""
import logging
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
from neurom.morphmath import section_length
from tmd.io.conversion import convert_morphio_trees
from tmd.Topology.methods import tree_to_property_barcode
from tmd.Topology.persistent_properties import PersistentAngles

logger = logging.getLogger(__name__)


def export_morph(root_path, morph_name, morph, morph_type, suffix=""):
    """Export the given morphology to the given path."""
    morph_path = str(root_path / f"{Path(morph_name).with_suffix('').name}{suffix}.asc")
    logger.debug("Export %s morphology to %s", morph_type, morph_path)
    morph.write(morph_path)
    return morph_path


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


def tuft_morph_path(root_path, group_name, axon_id, cluster_id):
    """Create a tuft file path according to the group name, axon ID and cluster ID."""
    return root_path / f"{Path(group_name).with_suffix('').name}_{axon_id}_{cluster_id}.asc"


def reduce_clusters(
    group,
    group_name,
    morph,
    axons,
    cluster_df,
    directed_graphes,
    sections_to_add,
    morph_paths,
    cluster_props,
    shortest_paths,
    export_tuft_morph_dir=None,
):
    """Reduce clusters to one section from their common ancestors to their centers."""
    kept_path = set()
    for (axon_id, cluster_id), cluster in group.groupby(["axon_id", "cluster_id"]):
        # Skip the root cluster
        if (cluster.cluster_id == 0).any():
            continue

        # Compute the common ancestor of the nodes
        cluster_common_path = common_path(
            directed_graphes[axon_id],
            cluster["section_id"].tolist(),
            shortest_paths=shortest_paths[axon_id],
        )
        if len(cluster) == 1 and len(cluster_common_path) > 2:
            common_ancestor_shift = -2
        else:
            common_ancestor_shift = -1
        common_ancestor = cluster_common_path[common_ancestor_shift]
        common_section = morph.section(common_ancestor)

        kept_path = kept_path.union(cluster_common_path)

        # Create a morphology for the current tuft
        tuft_morph, tuft_ancestor = create_tuft_morphology(
            morph,
            set(cluster["section_id"]),
            common_ancestor,
            cluster_common_path[:common_ancestor_shift],
            shortest_paths[axon_id],
        )

        # Compute cluster center
        cluster_center = cluster_df.loc[
            (cluster_df["axon_id"] == axon_id) & (cluster_df["terminal_id"] == cluster_id),
            ["x", "y", "z"],
        ].values[0]

        # Compute tuft orientation
        tuft_orientation = cluster_center - tuft_ancestor.points[-1]
        tuft_orientation /= np.linalg.norm(tuft_orientation)

        # Resize the common section used as root (the root section is 1um)
        resize_root_section(tuft_morph, tuft_orientation)

        # Compute the barcode
        tuft_barcode = get_barcode(tuft_morph)

        if export_tuft_morph_dir is not None:
            # Export each tuft as a morphology
            morph_paths["tufts"].append(
                (
                    group_name,
                    axon_id,
                    cluster_id,
                    export_morph(
                        export_tuft_morph_dir,
                        group_name,
                        tuft_morph,
                        "tuft",
                        f"_{axon_id}_{cluster_id}",
                    ),
                )
            )

        # Add tuft category data
        path_distance = sum(
            section_length(i.points) for i in common_section.iter(IterType.upstream)
        )
        radial_distance = np.linalg.norm(
            axons[axon_id].points[0, COLS.XYZ] - common_section.points[-1]
        )
        path_length = sum(section_length(i.points) for i in common_section.iter())

        cluster_props.append(
            (
                group_name,
                axon_id,
                cluster_id,
                cluster_center.tolist(),
                common_ancestor,
                tuft_ancestor.points[-1].tolist(),
                path_distance,
                radial_distance,
                path_length,
                len(cluster),
                tuft_orientation.tolist(),
                np.array(tuft_barcode).tolist(),
            )
        )

        # Continue if the cluster has only one node
        if len(cluster) == 1:
            continue

        # Create a new section from the common ancestor to the center of the cluster
        sections_to_add[common_section.id].append(
            PointLevel(
                [
                    common_section.points[-1],
                    cluster_df.loc[
                        (cluster_df["axon_id"] == cluster["axon_id"].iloc[0])
                        & (cluster_df["terminal_id"] == cluster_id),
                        ["x", "y", "z"],
                    ].values[0],
                ],
                [0, 0],
            )
        )
    return kept_path


def create_clustered_morphology(morph, group_name, kept_path, sections_to_add):
    """Create a new morphology with the kept path and add new sections to cluster centers."""
    clustered_morph = Morphology(
        deepcopy(morph),
        name=f"Clustered {Path(group_name).with_suffix('').name}",
    )
    trunk_morph = Morphology(
        deepcopy(morph),
        name=f"Clustered {Path(group_name).with_suffix('').name}",
    )

    for axon, new_axon, trunk_axon in zip(
        morph.neurites, clustered_morph.neurites, trunk_morph.neurites
    ):
        if axon.type != NeuriteType.axon:
            continue

        root = axon.root_node
        new_root = new_axon.root_node
        new_trunk_root = trunk_axon.root_node

        assert np.array_equal(root.points, new_root.points), "The axons were messed up!"

        for sec in new_root.children:
            clustered_morph.delete_section(sec.morphio_section)
        for sec in new_trunk_root.children:
            trunk_morph.delete_section(sec.morphio_section)

        current_sections = [(root, new_root, new_trunk_root)]

        # Add kept sections
        while current_sections:
            current_section, current_new_section, current_trunk_section = current_sections.pop()
            for child in current_section.children:
                if child.id in kept_path:
                    new_section = PointLevel(
                        child.points[:, COLS.XYZ].tolist(),
                        (child.points[:, COLS.R] * 2).tolist(),
                    )
                    current_sections.append(
                        (
                            child,
                            current_new_section.append_section(new_section),
                            current_trunk_section.append_section(new_section),
                        )
                    )

            if current_section.id in sections_to_add:
                for new_sec in sections_to_add[current_section.id]:
                    current_new_section.append_section(new_sec)
    return clustered_morph, trunk_morph
