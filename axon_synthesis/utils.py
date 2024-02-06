"""Some utils for the AxonSynthesis package."""
import collections.abc
import inspect
import json
import logging
import re
import warnings
from collections.abc import Callable
from collections.abc import MutableMapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from neurom import NeuriteType

from axon_synthesis.constants import COORDS_COLS


class MorphNameAdapter(logging.LoggerAdapter):
    """Add the morphology name and optionally the axon ID to the log entries."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, object]
    ) -> tuple[str, MutableMapping[str, object]]:
        """Add extra information to the log entry."""
        if self.extra is not None:
            header = f"morphology {self.extra['morph_name']}"
            if "axon_id" in self.extra:
                header += f" (axon {self.extra['axon_id']})"
            return f"{header}: {msg}", kwargs
        return "", kwargs


def sublogger(
    logger: logging.Logger | logging.LoggerAdapter | None, name: str
) -> logging.Logger | logging.LoggerAdapter:
    """Get a sub-logger with specific name."""
    if isinstance(logger, logging.LoggerAdapter):
        new_logger = logger.manager.getLogger(name)
        return logger.__class__(new_logger, logger.extra)
    return logging.getLogger(name)


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
        for logger_name in [
            "distributed",
            "h5py",
            "morph_tool.converter",
        ]:
            logging.getLogger(logger_name).level = max(
                logging.getLogger(logger_name).level,
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
    layer_field = atlas.brain_regions.with_data(layers)
    return layer_field.lookup(pos, outer_value=0)


def add_camera_sync(fig_path):
    """Update the HTML file to synchronize the cameras between the two plots."""
    with Path(fig_path).open(encoding="utf-8") as f:
        tmp = f.read()
        id_match = re.match('.*id="([^ ]*)" .*', tmp, flags=re.DOTALL)
        if id_match is None:
            msg = f"Could not find the figure ID in {fig_path}"
            raise ValueError(msg)
        fig_id = id_match.group(1)

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
            graph_nodes.append((node_id, *section.points[0, :4], True, -1, 0))
            last_pt = last_pts[None]
        else:
            last_pt = last_pts[section.parent.id]

        # Add segment points
        pts = section.points[1:, :4] if keep_section_segments else section.points[-1:, :4]
        len_pts = len(pts) - 1

        for num, i in enumerate(pts.tolist()):
            node_id = node_id + 1
            graph_nodes.append((node_id, *i, num == len_pts and is_terminal, section.id, num))
            graph_edges.append((last_pt, node_id))
            last_pt = node_id

        last_pts[section.id] = last_pt

    nodes = pd.DataFrame(
        graph_nodes,
        columns=["id", *COORDS_COLS, "radius", "is_terminal", "section_id", "sub_segment_num"],
    )
    nodes.set_index("id", inplace=True)

    edges = pd.DataFrame(graph_edges, columns=["source", "target"])
    edges = edges.sort_values(
        ["source", "target"],
    ).reset_index(drop=True)

    graph = nx.from_pandas_edgelist(edges, create_using=graph_cls, **graph_kwargs)
    nx.set_node_attributes(graph, nodes[[*COORDS_COLS, "radius", "is_terminal"]].to_dict("index"))

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

    nodes = pd.DataFrame(graph_nodes, columns=["id", *COORDS_COLS, "is_terminal"])
    nodes.set_index("id", inplace=True)

    edges = pd.DataFrame(graph_edges, columns=["source", "target"])
    graph = nx.from_pandas_edgelist(edges, create_using=graph_cls, **graph_kwargs)
    nx.set_node_attributes(graph, nodes[[*COORDS_COLS, "is_terminal"]].to_dict("index"))

    return nodes, edges, graph


@contextmanager
def disable_loggers(*logger_names):
    """A context manager that will disable logging messages triggered during the body.

    Args:
        *logger_names (str): The names of the loggers to be disabled.
    """
    loggers = (
        [logging.getLogger()] if not logger_names else [logging.getLogger(i) for i in logger_names]
    )

    disabled_loggers = [(i, i.disabled) for i in loggers]

    try:
        for i, _ in disabled_loggers:
            i.disabled = True
        yield
    finally:
        for i, j in disabled_loggers:
            i.disabled = j


@contextmanager
def ignore_warnings(*ignored_warnings):
    """A context manager that will ignore warnings raised during the body.

    Args:
        *ignored_warnings (Warning): The classes of the warnings to be ignored.
    """
    with warnings.catch_warnings():
        for i in ignored_warnings:
            warnings.filterwarnings("ignore", category=i)
        yield


def recursive_to_str(data):
    """Cast all Path objects into str objects in a given dict."""
    new_data = deepcopy(data)
    for k, v in new_data.items():
        if isinstance(v, dict):
            new_data[k] = recursive_to_str(v)
        elif isinstance(v, Path):
            new_data[k] = str(v)
    return new_data


def recursive_update(data, updates):
    """Update a dictionary with another with nested values."""
    for k, v in updates.items():
        if isinstance(v, collections.abc.Mapping):
            data[k] = recursive_update(data.get(k, {}), v)
        else:
            data[k] = v
    return data


def merge_json_files(*files):
    """Merge several JSON files together.

    The order is important: the files will be updated by all the next files in the list.
    """
    result: dict = {}
    for i in files:
        file = Path(i)
        if file.exists():
            with file.open(encoding="utf-8") as f:
                recursive_update(result, json.load(f))
    return result


def check_min_max(
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    strict_min: bool = False,
    strict_max: bool = False,
) -> Callable:
    """Create a validator used by attrs to check a range."""

    def range_validator(_instance, attribute, value) -> None:
        """The actual range validator used by attrs."""
        try:
            boundaries_msgs = []
            if (
                value is None
                or min_value is None
                or min_value < value
                or (not strict_min and min_value == value)
            ):
                min_ok = True
            else:
                min_ok = False
                boundaries_msgs.append(
                    f"{'strictly ' if strict_min else ''}greater than {min_value}"
                )
            if (
                value is None
                or max_value is None
                or max_value > value
                or (not strict_max and max_value == value)
            ):
                max_ok = True
            else:
                max_ok = False
                boundaries_msgs.append(f"{'strictly ' if strict_max else ''}lower than {max_value}")
            if not min_ok or not max_ok:
                msg = (
                    f"The attribute '{attribute.name}' must be {' and '.join(boundaries_msgs)} "
                    f"(got {value})"
                )
                raise ValueError(msg)
        except TypeError as exc:
            msg = f"The attribute '{attribute.name}' must have a numeric type (got {value})"
            raise ValueError(msg) from exc

    return range_validator


def compute_bbox(points, relative_buffer=None):
    """Compute the bounding box of the given point and optionally apply a buffer to it."""
    bbox = np.vstack([points.min(axis=0), points.max(axis=0)])
    if relative_buffer is not None:
        bbox_buffer = (bbox[1] - bbox[0]) * 0.1
        bbox[0] -= bbox_buffer
        bbox[1] += bbox_buffer
    return bbox


def get_code_location(back_frames=1):
    """Return the current file name and line number in the program."""
    frame = inspect.currentframe()
    if frame is None:
        msg = "Could not find the current frame"
        raise RuntimeError(msg)
    for num in range(back_frames):
        frame = frame.f_back
        if frame is None:
            msg = f"Could not find the back frame number {num}"
            raise RuntimeError(msg)
    return frame.f_code.co_filename, frame.f_lineno
