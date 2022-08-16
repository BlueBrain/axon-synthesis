"""Some utils for the AxonSynthesis package."""
import json
import logging
import re
from contextlib import contextmanager

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from neurom import NeuriteType


def fill_diag(mat, val=1):
    """Fill the diagonal of the given matrix."""
    np.fill_diagonal(mat, val)
    return mat


def cols_to_json(df, cols):
    """Transform the given columns from Python objects to JSON strings."""
    df = df.copy(deep=False)
    for col in cols:
        df[col] = df[col].map(json.dumps)
    return df


def cols_from_json(df, cols):
    """Transform the given columns to Python objects from JSON strings."""
    df = df.copy(deep=False)
    for col in cols:
        df[col] = df[col].map(json.loads)
    return df


def get_layers(atlas, brain_regions, pos):
    """Get layer data."""
    # TODO: get layer from the region names?
    names, ids = atlas.get_layers()
    layers = np.zeros_like(brain_regions.raw, dtype="uint8")
    layer_mapping = {}
    for layer_id, (ids_set, layer) in enumerate(zip(ids, names)):
        layer_mapping[layer_id] = layer
        layers[np.isin(brain_regions.raw, list(ids_set))] = layer_id + 1
    layers = brain_regions.with_data(layers)
    return layers.lookup(pos, outer_value=0)


def add_camera_sync(fig_path):
    """Update the HTML file to synchronize the cameras between the two plots."""
    with open(fig_path, encoding="utf-8") as f:
        tmp = f.read()
        fig_id = re.match('.*id="([^ ]*)" .*', tmp, flags=re.DOTALL).group(1)

    js = f"""
    <script>
    var gd = document.getElementById('{fig_id}');
    var isUnderRelayout = false

    gd.on('plotly_relayout', () => {{
      console.log('relayout', isUnderRelayout)
      if (!isUnderRelayout) {{
        Plotly.relayout(gd, 'scene2.camera', gd.layout.scene.camera)
          .then(() => {{ isUnderRelayout = false }}  )
      }}

      isUnderRelayout = true;
    }})
    </script>
    """

    with open(fig_path, "w", encoding="utf-8") as f:
        f.write(tmp.replace("</body>", js + "</body>"))


def get_axons(morph):
    """Get axons of the given morphology."""
    return [i for i in morph.neurites if i.type == NeuriteType.axon]


def neurite_to_graph(neurite, graph_cls=nx.DiGraph, keep_section_segments=False, **graph_kwargs):
    """Transform a neurite into a graph."""
    graph_nodes = []
    graph_edges = []
    node_id = -1
    last_pts = {None: -1}
    for section in neurite.iter_sections():
        is_terminal = not bool(section.children)

        if section.parent is None:
            # Add first point of the root section
            graph_nodes.append((node_id, *section.points[0, :3], True, -1, 0))
            last_pt = last_pts[None]
        else:
            last_pt = last_pts[section.parent.id]

        # Add segment points
        if keep_section_segments:
            pts = section.points[1:, :3]
        else:
            pts = section.points[-1:, :3]
        len_pts = len(pts) - 1

        for num, i in enumerate(pts.tolist()):
            node_id = node_id + 1
            graph_nodes.append((node_id, *i, num == len_pts and is_terminal, section.id, num))
            graph_edges.append((last_pt, node_id))
            last_pt = node_id

        last_pts[section.id] = last_pt

    nodes = pd.DataFrame(
        graph_nodes, columns=["id", "x", "y", "z", "is_terminal", "section_id", "sub_segment_num"]
    )
    nodes.set_index("id", inplace=True)

    edges = pd.DataFrame(graph_edges, columns=["source", "target"])
    edges = edges.sort_values(
        ["source", "target"],
    ).reset_index(drop=True)

    graph = nx.from_pandas_edgelist(edges, create_using=graph_cls, **graph_kwargs)
    nx.set_node_attributes(graph, nodes[["x", "y", "z", "is_terminal"]].to_dict("index"))

    return nodes, edges, graph


def neurite_to_graph_old(neurite, graph_cls=nx.DiGraph, **graph_kwargs):
    """Transform a neurite into a graph."""
    graph_nodes = []
    graph_edges = []
    for section in neurite.iter_sections():
        is_terminal = not bool(section.children)
        if section.parent is None:
            graph_nodes.append((-1, *section.points[0, :3], True))
            graph_edges.append((-1, section.id))

        graph_nodes.append((section.id, *section.points[-1, :3], is_terminal))

        for child in section.children:
            graph_edges.append((section.id, child.id))

    nodes = pd.DataFrame(graph_nodes, columns=["id", "x", "y", "z", "is_terminal"])
    nodes.set_index("id", inplace=True)

    edges = pd.DataFrame(graph_edges, columns=["source", "target"])
    graph = nx.from_pandas_edgelist(edges, create_using=graph_cls, **graph_kwargs)
    nx.set_node_attributes(graph, nodes[["x", "y", "z", "is_terminal"]].to_dict("index"))

    return nodes, edges, graph


def append_section_recursive(source, target):
    """Append a target section to a source."""

    current_sections = [(source, target)]

    # Add kept sections
    while current_sections:
        source_parent, target_child = current_sections.pop()
        source_child = source_parent.append_section(target_child)
        for child in target_child.children:
            current_sections.append((source_child, child))
        #     if child.id in kept_path:
        #         new_section = PointLevel(
        #             child.points[:, COLS.XYZ].tolist(),
        #             (child.points[:, COLS.R] * 2).tolist(),
        #         )
        #         current_sections.append(
        #             (current_child, current_child.append_section(new_section))
        #         )

        # if current_parent.id in sections_to_add:
        #     for new_sec in sections_to_add[current_parent.id]:
        #         current_child.append_section(new_sec)


@contextmanager
def disable_loggers(*logger_names):
    """
    A context manager that will prevent any logging messages triggered during the body from being
    processed.

    Args:
        *logger_names (str): The names of the loggers to be disabled.
    """

    if not logger_names:
        loggers = [logging.root]
    else:
        loggers = [logging.getLogger(i) for i in logger_names]

    disabled_loggers = [(i, i.disabled) for i in loggers]

    try:
        for i, _ in disabled_loggers:
            i.disabled = True
        yield
    finally:
        for i, j in disabled_loggers:
            i.disabled = j


@contextmanager
def use_matplotlib_backend(new_backend):
    """
    A context manager that will set a new temporary backend to matplotlib then restore the old one.

    Args:
        new_backend (str): The name of the backend to use in this context.
    """
    old_backend = matplotlib.get_backend()
    matplotlib.use(new_backend)
    try:
        yield
    finally:
        matplotlib.use(old_backend)
