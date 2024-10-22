"""Clustering from brain regions."""

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
from pathlib import Path

import dask.distributed
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bluepyparallel import evaluate
from bluepyparallel import init_parallel_factory
from plotly.subplots import make_subplots
from voxcell.math_utils import voxel_intersection

from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.utils import build_layout_properties
from axon_synthesis.utils import disable_loggers
from axon_synthesis.utils import neurite_to_graph

logger = logging.getLogger(__name__)


def merge_similar_regions(regions, sub_segments):
    """Merge similar consecutive region IDs and corresponding sub-segments."""
    regions = np.array(regions)
    sub_segments = np.array(sub_segments)

    if regions.size == 0:
        return {
            "brain_regions": np.zeros_like([], dtype=int),
            "sub_segments": np.zeros_like([], shape=(0, 6), dtype=float),
        }

    # Find transitions between regions
    transitions = regions[:-1] != regions[1:]

    transition_indices = np.nonzero(transitions)[0]
    region_transition_indices = np.append(transition_indices, len(regions) - 1)
    seg_transition_indices = np.insert(region_transition_indices, 0, 0)

    # Build segments by region
    left_index = seg_transition_indices[:-1].copy()
    right_index = seg_transition_indices[1:].copy()
    couple_idx = np.vstack([left_index, right_index]).T
    segment_couples = sub_segments[couple_idx]

    region_sub_segments = np.hstack(
        [segment_couples[:, 0, [3, 4, 5]], segment_couples[:, 1, [3, 4, 5]]],
    )
    # Fix first start point
    region_sub_segments[0, [0, 1, 2]] = segment_couples[0, 0, [0, 1, 2]]

    # Remove segments with zero length
    seg_lengths = np.linalg.norm(
        region_sub_segments[:, [0, 1, 2]] - region_sub_segments[:, [3, 4, 5]],
        axis=1,
    )
    region_sub_segments = region_sub_segments[seg_lengths > 0]
    region_indices = regions[region_transition_indices][seg_lengths > 0]

    return {
        "brain_regions": region_indices,
        "sub_segments": region_sub_segments,
    }


def segment_region_ids(row, brain_regions):
    """Split segments according to brain regions and get the regions IDs."""
    start_pt = [row["source_x"], row["source_y"], row["source_z"]]
    end_pt = [row["target_x"], row["target_y"], row["target_z"]]
    indices, sub_segments = voxel_intersection(
        [start_pt, end_pt],
        brain_regions,
        return_sub_segments=True,
    )
    regions = brain_regions.raw[tuple(indices.T.tolist())]

    return merge_similar_regions(regions, sub_segments)


def cut_edges(edges, nodes, brain_regions, nb_workers, group_name):
    """Cut edges according to brain regions."""
    # Initialize Dask cluster
    with disable_loggers("asyncio", "distributed", "distributed.worker"):
        if nb_workers > 1:
            cluster = dask.distributed.LocalCluster(n_workers=nb_workers, timeout="60s")
            parallel_factory = init_parallel_factory("dask_dataframe", address=cluster)
        else:
            parallel_factory = init_parallel_factory(None)

        # Compute region indices of each segment
        all_brain_regions = evaluate(
            edges,
            segment_region_ids,
            [
                ["brain_regions", None],
                ["sub_segments", None],
            ],
            parallel_factory=parallel_factory,
            func_args=[brain_regions],
        )

        # Close the Dask cluster if opened
        if nb_workers > 1:
            parallel_factory.shutdown()
            cluster.close()

    logger.debug("%s: Computed brain regions for %s segments", group_name, len(edges))

    nb_intersected_regions = all_brain_regions["brain_regions"].apply(len)
    cut_edge_mask = nb_intersected_regions >= 2
    one_region_mask = nb_intersected_regions == 1

    # Set brain regions to not cut edges
    edges["brain_region"] = -1
    edges.loc[one_region_mask, "brain_region"] = all_brain_regions.loc[
        one_region_mask,
        "brain_regions",
    ].apply(lambda x: x[0])

    # Select edges that have to be cut
    edges_to_cut = edges.loc[cut_edge_mask].join(
        all_brain_regions.loc[cut_edge_mask, ["brain_regions", "sub_segments"]],
    )

    # Split lists into rows
    if not edges_to_cut.empty:
        region_sub_edges = edges_to_cut["brain_regions"].apply(pd.Series).stack().astype(int)
        segment_sub_edges = (
            edges_to_cut["sub_segments"].apply(lambda x: pd.Series(x.tolist())).stack()
        )

        region_sub_edges = region_sub_edges.reset_index(name="brain_region")
        segment_sub_edges = segment_sub_edges.reset_index(name="sub_segment")

        # Split coordinates into new columns
        segment_sub_edges[
            ["source_x", "source_y", "source_z", "target_x", "target_y", "target_z"]
        ] = pd.DataFrame(
            segment_sub_edges["sub_segment"].tolist(),
            columns=[
                "source_x",
                "source_y",
                "source_z",
                "target_x",
                "target_y",
                "target_z",
            ],
            index=segment_sub_edges.index,
        )

        # Join regions to sub-segments
        segment_sub_edges = segment_sub_edges.merge(region_sub_edges, on=["level_0", "level_1"])

        # Join sub-segments to edges to keep initial values for first and last segment points
        segment_sub_edges = segment_sub_edges.merge(
            edges[
                [
                    "source",
                    "target",
                    "source_is_terminal",
                    "target_is_terminal",
                    "source_section_id",
                    "target_section_id",
                    "source_sub_segment_num",
                    "target_sub_segment_num",
                    "source_is_intermediate_pt",
                    "target_is_intermediate_pt",
                ]
            ],
            left_on="level_0",
            right_index=True,
            how="left",
        )

        # Find indices of first and last element of each group
        # head_index = segment_sub_edges["level_1"] == 0
        # tail_index = segment_sub_edges.groupby("level_0").tail(1).index
        intermediate_sources = segment_sub_edges.groupby("level_0").tail(-1)
        intermediate_targets = segment_sub_edges.groupby("level_0").head(-1)

        # Add intermediate points to nodes
        intermediate_target_nodes = intermediate_targets[
            [
                "level_0",
                "level_1",
                "target_x",
                "target_y",
                "target_z",
                "target_section_id",
                "target_sub_segment_num",
            ]
        ].copy()

        intermediate_target_nodes["is_terminal"] = False  # By definition they can't be terminals
        intermediate_target_nodes["is_intermediate_pt"] = True  # Also by definition
        intermediate_target_nodes.rename(
            columns={
                "target_x": "x",
                "target_y": "y",
                "target_z": "z",
                "target_section_id": "section_id",
                "target_sub_segment_num": "sub_segment_num",
            },
            inplace=True,
        )

        intermediate_target_nodes.reset_index(inplace=True)
        intermediate_target_nodes.index += nodes.index.max() + 1
        intermediate_target_nodes["new_index"] = intermediate_target_nodes.index

        # Set source and target indices
        segment_sub_edges.loc[intermediate_targets.index, "target"] = (
            segment_sub_edges.loc[intermediate_targets.index]
            .merge(
                intermediate_target_nodes[["level_0", "level_1", "new_index"]],
                on=["level_0", "level_1"],
            )["new_index"]
            .tolist()
        )
        segment_sub_edges.loc[intermediate_sources.index, "source"] = segment_sub_edges.loc[
            intermediate_targets.index,
            "target",
        ].tolist()

        # Set source_is_terminal and target_is_terminal attributes to False (the intermediate
        # points can not be terminals)
        segment_sub_edges.loc[intermediate_sources.index, "source_is_terminal"] = False
        segment_sub_edges.loc[intermediate_targets.index, "target_is_terminal"] = False

        # Fix sub-segment numbers
        # TODO: Vectorize the for loop
        intermediate_target_nodes["idx_shift"] = intermediate_target_nodes.groupby(
            "section_id",
        ).cumcount()
        intermediate_target_nodes["sub_segment_num"] += intermediate_target_nodes["idx_shift"]
        for i in intermediate_target_nodes.itertuples():
            nodes.loc[
                (nodes["section_id"] == i.section_id)
                & (nodes["sub_segment_num"] >= i.sub_segment_num),
                "sub_segment_num",
            ] += 1
        segment_sub_edges.loc[
            intermediate_sources.index,
            ["source_sub_segment_num", "target_sub_segment_num"],
        ] += 1

        # Build new DataFrames
        new_nodes = pd.concat(
            [
                nodes,
                intermediate_target_nodes[
                    [i for i in nodes.columns if i in intermediate_target_nodes.columns]
                ],
            ],
        )

        new_edges = pd.concat(
            [
                edges.drop(np.where(cut_edge_mask.values)[0]),
                segment_sub_edges[edges.columns],
            ],
            ignore_index=True,
        )
    else:
        new_nodes = nodes
        new_edges = edges

    return new_nodes, new_edges


def _find_wm_first_nested_region(region_id, wm_regions, region_map) -> int:
    """Find the first nested region ID that is available in the white matter recipe."""
    if region_id in wm_regions or region_id <= 0:
        return region_id

    ids = region_map.get(region_id, attr="id", with_ascendants=True)
    for i in ids[1:]:
        if i in wm_regions:
            return i

    return region_id


def compute_clusters(  # noqa: PLR0913
    atlas,
    wmr,
    config,
    config_name,
    axon,
    axon_id,
    group_name,
    group_path,
    group,
    nb_workers,
    tuft_morphologies_path,
    figure_path,
    *,
    debug=False,
    **_kwargs,
):
    """Gather points according to connected components in brain regions."""
    nodes, edges, _ = neurite_to_graph(axon, keep_section_segments=True)

    nodes["is_intermediate_pt"] = (
        nodes["sub_segment_num"]
        != nodes.merge(
            nodes.groupby("section_id")["sub_segment_num"].max(),
            left_on="section_id",
            right_index=True,
            suffixes=("", "_max"),
        )["sub_segment_num_max"]
    )

    edges = edges.join(nodes.add_prefix("source_"), on="source")
    edges = edges.join(nodes.add_prefix("target_"), on="target")

    # Cut edges according to brain regions
    new_nodes, new_edges = cut_edges(edges, nodes, atlas.brain_regions, nb_workers, group_name)

    # Normalize node indices
    new_nodes.sort_values(["section_id", "sub_segment_num"], inplace=True)
    new_nodes.index.name = "old_index"
    new_nodes.reset_index(inplace=True)
    new_nodes.index -= 1
    new_nodes.index.name = "id"
    new_nodes["new_index"] = new_nodes.index

    new_edges["source"] = new_edges.merge(
        new_nodes[["old_index", "new_index"]],
        left_on="source",
        right_on="old_index",
        how="left",
    )["new_index"]
    new_edges["target"] = new_edges.merge(
        new_nodes[["old_index", "new_index"]],
        left_on="target",
        right_on="old_index",
        how="left",
    )["new_index"]
    new_nodes.drop(columns=["new_index", "old_index"], inplace=True)

    # Normalize nested brain regions
    wm_regions = np.sort(wmr.populations["atlas_region_id"].unique())

    new_edges["wm_brain_region"] = new_edges["brain_region"].apply(
        _find_wm_first_nested_region,
        args=(wm_regions, atlas.region_map),
    )

    brain_region_attr = "wm_brain_region" if config["wm_unnesting"] else "brain_region"

    # Create a graph from these new nodes and edges
    graph = nx.from_pandas_edgelist(new_edges, create_using=nx.Graph)
    nx.set_node_attributes(graph, new_nodes.to_dict("index"))
    nx.set_edge_attributes(
        graph,
        new_edges.set_index(["source", "target"]).to_dict("index"),
    )

    # Get subgraphs of each brain region
    region_sub_graphs = {
        brain_region: graph.edge_subgraph(
            edges_component[["source", "target"]].to_records(index=False).tolist(),
        )
        for brain_region, edges_component in new_edges.groupby(brain_region_attr)
    }

    # Get connected components of each brain region
    region_components = {
        brain_region: list(nx.connected_components(region_sub_graphs[brain_region]))
        for brain_region, sub_graph in region_sub_graphs.items()
    }

    region_component_subgraphs = {
        brain_region: [region_sub_graphs[brain_region].subgraph(comp) for comp in components]
        for brain_region, components in region_components.items()
    }

    region_acronyms = {
        brain_region: atlas.region_map.get(brain_region, attr="acronym")
        for brain_region in region_components
        if brain_region > 0
    }

    # Create a cluster ID for each component
    group_nodes = (
        group.reset_index()
        .merge(
            new_nodes.loc[~new_nodes["is_intermediate_pt"]].reset_index()[["section_id", "id"]],
            on="section_id",
        )
        .set_index("index")
    )
    group_nodes = (
        group_nodes.reset_index()
        .merge(
            new_edges[["target", brain_region_attr]],
            left_on="id",
            right_on="target",
        )
        .set_index("index")
    )

    cluster_id = 0
    # cluster_ids = {}
    for components in region_components.values():  # pylint: disable=unused-variable
        for component in components:
            group_nodes.loc[group_nodes["id"].isin(list(component)), "tuft_id"] = cluster_id
            cluster_id = group_nodes["tuft_id"].max() + 1

    group["tuft_id"] = group_nodes["tuft_id"]

    new_terminal_points = []
    # new_terminal_id = group["terminal_id"].max() + 1
    for new_terminal_id, (cluster_id, i) in enumerate(group.groupby("tuft_id")):
        new_terminal_points.append(
            [
                group_name,
                group_path,
                config_name,
                axon_id,
                new_terminal_id if cluster_id != -1 else 0,
                len(i),
                *i[COORDS_COLS].mean().tolist(),
            ],
        )

    if debug:
        plot(
            region_component_subgraphs,
            region_acronyms,
            str(figure_path / f"{group_name}_region_clusters.html"),
        )

        tuft_brain_region_path = tuft_morphologies_path / f"{group_name}_{axon_id}.csv"
        logger.debug("Export tuft brain regions to %s", tuft_brain_region_path)
        group_nodes["region_acronym"] = group_nodes["wm_brain_region"].map(region_acronyms)
        # group_nodes["tuft_morph_path"] = group_nodes.apply(
        #     lambda row: task.tuft_morph_path(group_name, axon_id, row["tuft_id"]), axis=1
        # )
        # group_nodes.loc[group_nodes["tuft_id"] == 0, "tuft_morph_path"] = None
        group_nodes.to_csv(tuft_brain_region_path)

    return new_terminal_points, group["tuft_id"]


def plot(region_component_subgraphs, region_acronyms, filepath):
    """Plot the resulting nodes and edges."""
    rng = np.random.default_rng()

    total_num = sum(len(i) for i in region_component_subgraphs.values())
    all_colors = np.arange(total_num).tolist()
    rng.shuffle(all_colors)
    x = []
    y = []
    z = []
    color = []
    acronym = []
    # annotations = []
    for region, subgraphs in region_component_subgraphs.items():
        acr = region_acronyms.get(region, "UNKNOWN")
        for subgraph in subgraphs:
            tmp = all_colors.pop()
            for start_node, end_node in subgraph.edges:
                edge_data = subgraph.get_edge_data(start_node, end_node)
                x.extend(
                    [
                        edge_data["source_x"],
                        edge_data["target_x"],
                        None,
                    ],
                )
                y.extend(
                    [
                        edge_data["source_y"],
                        edge_data["target_y"],
                        None,
                    ],
                )
                z.extend(
                    [
                        edge_data["source_z"],
                        edge_data["target_z"],
                        None,
                    ],
                )
                color.extend([tmp, tmp, tmp])
                acronym.extend([acr, acr, acr])
    edge_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={
            "color": color,
            "width": 4,
            "colorscale": "hsv",
        },
        name="Morphology nodes to cluster",
        hovertext=acronym,
    )

    fig = make_subplots(
        cols=1,
        specs=[[{"is_3d": True}]],
        subplot_titles=("Region clusters"),
    )

    fig.add_trace(edge_trace, row=1, col=1)

    layout_props = build_layout_properties(np.stack([x, y, z]).T, 0.1)

    fig.update_scenes(layout_props)
    fig.update_layout(title=Path(filepath).stem)

    # Export figure
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(filepath)
