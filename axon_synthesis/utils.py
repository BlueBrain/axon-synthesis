"""Some utils for the AxonSynthesis package."""
import json
import logging
import re
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
from neurom import NeuriteType


def setup_logger(level="info", prefix="", suffix=""):
    """Setup application logger."""
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(
        format=prefix + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + suffix,
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=levels[level],
    )

    if levels[level] >= logging.INFO:  # pragma: no cover
        logging.getLogger("distributed").level = max(
            logging.getLogger("distributed").level,
            logging.WARNING,
        )


def fill_diag(mat, val=1):
    """Fill the diagonal of the given matrix."""
    np.fill_diagonal(mat, val)
    return mat


def cols_to_json(df, cols):
    """Transform the given columns from Python objects to JSON strings."""
    df = df.copy(deep=False)  # noqa: PD901
    for col in cols:
        df[col] = df[col].map(json.dumps)
    return df


def cols_from_json(df, cols):
    """Transform the given columns to Python objects from JSON strings."""
    df = df.copy(deep=False)  # noqa: PD901
    for col in cols:
        df[col] = df[col].map(json.loads)
    return df


def get_layers(atlas, pos):
    """Get layer data."""
    # TODO: get layer from the region names?
    names, ids = atlas.get_layers()
    layers = np.zeros_like(atlas.brain_regions.raw, dtype="uint8")
    layer_mapping = {}
    for layer_id, (ids_set, layer) in enumerate(zip(ids, names)):
        layer_mapping[layer_id] = layer
        layers[np.isin(atlas.brain_regions.raw, list(ids_set))] = layer_id + 1
    layers = atlas.brain_regions.with_data(layers)
    return layers.lookup(pos, outer_value=0)


def add_camera_sync(fig_path):
    """Update the HTML file to synchronize the cameras between the two plots."""
    with Path(fig_path).open(encoding="utf-8") as f:
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

    with Path(fig_path).open("w", encoding="utf-8") as f:
        f.write(tmp.replace("</body>", js + "</body>"))


def get_axons(morph):
    """Get axons of the given morphology."""
    return [i for i in morph.neurites if i.type == NeuriteType.axon]


def neurite_to_graph(neurite, graph_cls=nx.DiGraph, *, keep_section_segments=False, **graph_kwargs):
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
        pts = section.points[1:, :3] if keep_section_segments else section.points[-1:, :3]
        len_pts = len(pts) - 1

        for num, i in enumerate(pts.tolist()):
            node_id = node_id + 1
            graph_nodes.append((node_id, *i, num == len_pts and is_terminal, section.id, num))
            graph_edges.append((last_pt, node_id))
            last_pt = node_id

        last_pts[section.id] = last_pt

    nodes = pd.DataFrame(
        graph_nodes,
        columns=["id", "x", "y", "z", "is_terminal", "section_id", "sub_segment_num"],
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

        graph_edges.extend((section.id, child.id) for child in section.children)

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
        current_sections.extend((source_child, child) for child in target_child.children)


@contextmanager
def disable_loggers(*logger_names):
    """A context manager that will disable logging messages triggered during the body.

    Args:
        *logger_names (str): The names of the loggers to be disabled.
    """
    loggers = [logging.root] if not logger_names else [logging.getLogger(i) for i in logger_names]

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
    """A context manager to set a new temporary backend to matplotlib then restore the old one.

    Args:
        new_backend (str): The name of the backend to use in this context.
    """
    old_backend = mpl.get_backend()
    mpl.use(new_backend)
    try:
        yield
    finally:
        mpl.use(old_backend)


def recursive_to_str(data):
    """Cast all Path objects into str objects in a given dict."""
    new_data = deepcopy(data)
    for k, v in new_data.items():
        if isinstance(v, dict):
            new_data[k] = recursive_to_str(v)
        elif isinstance(v, Path):
            new_data[k] = str(v)
    return new_data
