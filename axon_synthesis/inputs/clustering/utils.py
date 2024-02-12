"""Some utils for clustering."""
import logging
from collections.abc import Iterator
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from jsonschema import Draft7Validator
from jsonschema import ValidationError
from jsonschema import validators
from jsonschema.protocols import Validator
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
from voxcell import VoxcellError
from voxcell import VoxelData
from voxcell.voxel_data import OrientationField

from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import DEFAULT_POPULATION
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType

logger = logging.getLogger(__name__)


def export_morph(root_path, morph_name, morph, morph_type, suffix=""):
    """Export the given morphology to the given path."""
    morph_path = str(root_path / f"{Path(morph_name).with_suffix('').name}{suffix}.asc")
    logger.debug("Export %s morphology to %s", morph_type, morph_path)
    morph.write(morph_path)
    return morph_path


def common_path(graph, nodes, source=None, shortest_paths=None) -> list[int]:
    """Compute the common paths of the given nodes.

    Source should be given only if the graph if undirected.
    Shortest paths can be given if they were already computed before.

    .. warning:: The graph must have only one component.
    """
    if not isinstance(graph, nx.DiGraph) and source is None and shortest_paths is None:
        msg = (
            "Either the source or the pre-computed shortest paths must be provided when using "
            "an undirected graph."
        )
        raise ValueError(msg)

    if shortest_paths is None:
        if isinstance(graph, nx.DiGraph):
            try:
                sources = [k for k, v in graph.in_degree if v == 0]
                if len(sources) > 1:
                    msg = "Several roots found in the directed graph."
                    raise RuntimeError(msg)
                source = sources[0]
            except IndexError as exc:
                # pylint: disable=raise-missing-from
                msg = "Could not find the root of the directed graph."
                raise RuntimeError(msg) from exc
        shortest_paths = nx.single_source_shortest_path(graph, source)

    # Compute the common ancestor
    common_nodes: set[int] = set(shortest_paths[nodes[0]])
    for i in nodes[1:]:
        common_nodes.intersection_update(set(shortest_paths[i]))

    return [i for i in shortest_paths[nodes[0]] if i in common_nodes]


def create_tuft_morphology(
    morph,
    tuft_section_ids,
    common_ancestor,
    cluster_common_path,
    shortest_paths,
):
    """Create a new morphology containing only the given tuft."""
    tuft_morph = Morphology(morph)
    for i in tuft_morph.root_sections:
        if i.type != NeuriteType.axon:
            tuft_morph.delete_section(i)

    tuft_sections = {
        j
        for terminal_id, path in shortest_paths.items()
        if terminal_id in tuft_section_ids
        for j in path
    }.difference(cluster_common_path)

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
        ],
    )
    new_root_section.diameters = np.repeat(new_root_section.diameters[1], 2)


def tuft_morph_path(root_path, group_name, axon_id, tuft_id):
    """Create a tuft file path according to the group name, axon ID and cluster ID."""
    return root_path / f"{Path(group_name).with_suffix('').name}_{axon_id}_{tuft_id}.asc"


def compute_mean_tuft_length(
    volume,
    n_pot,
    fraction,
    strength,
    bouton_density,
):
    """Compute properties for the cluster."""
    n_act = n_pot * fraction
    n_tot = volume * strength
    n_syn_per = n_tot / n_act
    return n_syn_per / bouton_density


def compute_common_section_properties(
    first_axon_pt, section, n_pot, volume, fraction, strength, bouton_density
):
    """Compute basic properties of the given section."""
    path_distance = sum(section_length(i.points) for i in section.iter(IterType.upstream))
    radial_distance = np.linalg.norm(first_axon_pt - section.points[-1])
    path_length = sum(section_length(i.points) for i in section.iter())

    if all(i is not None for i in [n_pot, volume, fraction, strength, bouton_density]):
        mean_tuft_length = compute_mean_tuft_length(
            volume,
            n_pot,
            fraction,
            strength,
            bouton_density,
        )
    else:
        mean_tuft_length = None

    return path_distance, radial_distance, path_length, mean_tuft_length


def reduce_clusters(  # noqa: PLR0912, PLR0913
    group,
    group_name,
    morph,
    axon,
    axon_id,
    cluster_df,
    directed_graph,
    sections_to_add,
    morph_paths,
    cluster_props: list[tuple],
    shortest_paths,
    bouton_density: float | None,
    brain_regions: VoxelData | None = None,
    atlas_region_id: int | None = None,
    atlas_orientations: OrientationField | None = None,
    projection_pop_numbers: pd.DataFrame | None = None,
    export_tuft_morph_dir: FileType | None = None,
    config_name: str | None = None,
    rng: SeedType = None,
) -> set[int]:
    """Reduce clusters to one section from their common ancestors to their centers."""
    if not config_name:
        config_name = ""
    kept_path: set[int] = set()
    group = group.dropna(subset="tuft_id").astype({"tuft_id": int})

    root_point = axon.points[0, COLS.XYZ]

    if projection_pop_numbers is not None:
        source_projections = projection_pop_numbers.loc[
            projection_pop_numbers["atlas_region_id"] == atlas_region_id
        ]
    else:
        source_projections = pd.DataFrame(columns=["target_atlas_id"])

    for tuft_id, cluster in group.groupby("tuft_id"):
        # Skip the root cluster
        if (cluster.tuft_id == 0).any():
            continue

        # Compute the common ancestor of the nodes
        cluster_common_path = common_path(
            directed_graph,
            cluster["section_id"].tolist(),
            shortest_paths=shortest_paths,
        )
        common_ancestor_shift = -2 if len(cluster) == 1 and len(cluster_common_path) > 2 else -1
        common_ancestor = cluster_common_path[common_ancestor_shift]
        common_section = morph.section(common_ancestor)

        kept_path = kept_path.union(cluster_common_path)

        # TODO: The graph node IDs should be mapped to section IDs, currently it only workds for the
        # first neurite because the graph node IDs correspond to the section IDs of the first
        # neurite

        # Create a morphology for the current tuft
        tuft_morph, tuft_ancestor = create_tuft_morphology(
            morph,
            set(cluster["section_id"]),
            common_ancestor,
            cluster_common_path[:common_ancestor_shift],
            shortest_paths,
        )

        # Compute cluster center
        cluster_center = cluster_df.loc[
            cluster_df["terminal_id"] == tuft_id,
            COORDS_COLS,
        ].to_numpy()[0]

        # Compute tuft orientation
        tuft_orientation = cluster_center - tuft_ancestor.points[-1]
        tuft_orientation /= np.linalg.norm(tuft_orientation)
        if atlas_orientations is not None:
            try:
                tuft_orientation = np.dot(
                    tuft_orientation, atlas_orientations.lookup(tuft_ancestor.points[-1])[0].T
                )
            except VoxcellError:
                logger.exception(
                    "Could not retrieve the atlas orientation for %s at %s",
                    group_name,
                    tuft_ancestor.points[-1],
                )

        # Resize the common section used as root (the root section is 1um)
        resize_root_section(tuft_morph, tuft_orientation)

        # Compute the barcode
        tuft_barcode = get_barcode(tuft_morph)

        if export_tuft_morph_dir is not None:
            # Export each tuft as a morphology
            morph_paths["tufts"].append(
                (
                    group_name,
                    config_name,
                    axon_id,
                    tuft_id,
                    export_morph(
                        export_tuft_morph_dir,
                        group_name,
                        tuft_morph,
                        "tuft",
                        f"_{config_name}_{axon_id}_{tuft_id}",
                    ),
                ),
            )

        # Add tuft category data
        # TODO: Fix the WMR stuff
        try:
            target_atlas_region_id = brain_regions.lookup(  # type: ignore[union-attr]
                cluster_center
            )
        except:  # noqa: E722 ; pylint: disable=bare-except
            target_atlas_region_id = 0
        target_projection_number = source_projections.loc[
            source_projections["target_atlas_id"] == target_atlas_region_id
        ]
        if len(target_projection_number) > 0:
            target_tmp = target_projection_number.sample(random_state=rng)
            target_projection_number, volume, target_projection_name = target_tmp[
                ["pop_neuron_numbers", "atlas_region_volume_target", "target_projection_name"]
            ].to_numpy()[0]
        else:
            target_projection_number = None
            target_projection_name = DEFAULT_POPULATION
            fraction = None
            strength = None
            volume = None

        (
            path_distance,
            radial_distance,
            path_length,
            mean_tuft_length,
        ) = compute_common_section_properties(
            root_point,
            common_section,
            target_projection_number,
            volume,
            fraction,
            strength,
            bouton_density,
        )

        # TODO: Compute the cluster weights somewhere else

        cluster_props.append(
            (
                group_name,
                config_name,
                axon_id,
                tuft_id,
                cluster_center.tolist(),
                common_ancestor,
                *tuft_ancestor.points[-1].tolist(),
                path_distance,
                radial_distance,
                path_length,
                len(cluster),
                tuft_orientation.tolist(),
                mean_tuft_length,
                target_projection_name,
                np.array(tuft_barcode).tolist(),
            ),
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
                        (cluster_df["terminal_id"] == tuft_id),
                        COORDS_COLS,
                    ].to_numpy()[0],
                ],
                [0, 0],
            ),
        )
    return kept_path


def create_clustered_morphology(morph, group_name, kept_path, sections_to_add, suffix=None):
    """Create a new morphology with the kept path and add new sections to cluster centers."""
    if not suffix:
        suffix = ""
    clustered_morph = Morphology(
        morph,
        name=f"Clustered {Path(group_name).with_suffix('').name}{suffix}",
    )
    trunk_morph = Morphology(
        morph,
        name=f"Clustered trunk {Path(group_name).with_suffix('').name}{suffix}",
    )

    for axon, new_axon, trunk_axon in zip(
        morph.neurites,
        clustered_morph.neurites,
        trunk_morph.neurites,
    ):
        if axon.type != NeuriteType.axon:
            continue

        root = axon.root_node
        new_root = new_axon.root_node
        new_trunk_root = trunk_axon.root_node

        if not np.array_equal(root.points, new_root.points):
            msg = "The axons were messed up!"
            raise RuntimeError(msg)

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
                        ),
                    )

            if current_section.id in sections_to_add:
                for new_sec in sections_to_add[current_section.id]:
                    current_new_section.append_section(new_sec)
    return clustered_morph, trunk_morph


def extend_validator_with_default(validator_class) -> Validator:
    """Extend a validator to automatically set default values during validation."""
    _no_default = object()
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults_and_validate(
        validator, properties, instance, schema
    ) -> Iterator[ValidationError] | None:
        drop_if_empty = set()
        new_instance = deepcopy(instance)
        for prop, subschema in properties.items():
            if prop in new_instance:
                continue
            obj_type = subschema.get("type", "")
            default_value = subschema.get("default", _no_default)
            if default_value is not _no_default:
                new_instance.setdefault(prop, default_value)
            elif obj_type == "object":
                new_instance.setdefault(prop, {})
                drop_if_empty.add(prop)

        is_valid = True
        for error in validate_properties(
            validator,
            properties,
            new_instance,
            schema,
        ):
            is_valid = False
            yield error

        for prop in drop_if_empty:
            instance_prop = new_instance[prop]
            if isinstance(instance_prop, Mapping) and len(instance_prop) == 0:
                del new_instance[prop]

        if is_valid:
            instance.update(new_instance)

    return validators.extend(
        validator_class,
        {"properties": set_defaults_and_validate},
    )


DefaultValidatingValidator = extend_validator_with_default(Draft7Validator)
