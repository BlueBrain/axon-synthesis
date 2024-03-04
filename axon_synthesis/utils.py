"""Some utils for the AxonSynthesis package."""
import collections.abc
import inspect
import json
import logging
import re
import warnings
from collections.abc import MutableMapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

try:
    from mpi4py import MPI

    mpi_enabled = True
except ImportError:
    mpi_enabled = False

import networkx as nx
import numpy as np
import pandas as pd
from morph_tool.converter import convert
from morphio.mut import Morphology as MorphIoMorphology
from neurom import NeuriteType
from neurom import load_morphology as neurom_load_morphology
from neurom.core import Morphology
from neurom.core.soma import SomaType
from neurom.geom.transform import Translation
from voxcell.cell_collection import CellCollection

from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.typing import RegionIdsType
from axon_synthesis.typing import SeedType

LOGGER = logging.getLogger(__name__)


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
    if logger is not None:
        new_logger = logger.manager.getLogger(name)
        if isinstance(logger, logging.LoggerAdapter):
            return logger.__class__(new_logger, logger.extra)
        return new_logger
    return logging.getLogger(name)


def setup_logger(level="info", prefix="", suffix=""):
    """Setup application logger."""
    if mpi_enabled:  # pragma: no cover
        comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
        if comm.Get_size() > 1:
            rank = comm.Get_rank()
            prefix = f"#{rank} - {prefix}"
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
    nx.set_node_attributes(
        graph, nodes[["section_id", *COORDS_COLS, "radius", "is_terminal"]].to_dict("index")
    )

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


def compute_bbox(points, relative_buffer=None):
    """Compute the bounding box of the given points and optionally apply a buffer to it."""
    bbox = np.vstack([points.min(axis=0), points.max(axis=0)])
    if relative_buffer is not None:
        bbox_buffer = (bbox[1] - bbox[0]) * 0.1
        bbox[0] -= bbox_buffer
        bbox[1] += bbox_buffer
    return bbox


def compute_aspect_ratios(bbox):
    """Compute the aspect ratios of a bounding box."""
    aspect_ratios = bbox[1] - bbox[0]
    aspect_ratios /= aspect_ratios[0]
    return aspect_ratios


def build_layout_properties(pts, relative_buffer: float | None = None) -> dict:
    """Build a dictionary with layout properties for Plotly figures."""
    bbox = compute_bbox(pts, relative_buffer)
    aspect_ratios = compute_aspect_ratios(bbox)

    return {
        "aspectmode": "manual",
        "aspectratio": {"x": aspect_ratios[0], "y": aspect_ratios[1], "z": aspect_ratios[2]},
        "xaxis": {
            "range": bbox[:, 0],
        },
        "yaxis": {
            "range": bbox[:, 1],
        },
        "zaxis": {
            "range": bbox[:, 2],
        },
    }


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


def load_morphology(path, *, recenter=False):
    """Load a morphology a optionally recenter it."""
    morph = neurom_load_morphology(path)
    if recenter:
        morph = morph.transform(Translation(-morph.soma.center))
    return morph


@disable_loggers("morph_tool.converter")
def save_morphology(
    morph: Morphology,
    morph_path: FileType,
    msg: str | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Export the given morphology to the given path."""
    if msg is None:
        msg = f"Export morphology to {morph_path}"
    logger = sublogger(logger, __name__)
    logger.debug(msg)
    convert(morph, morph_path)
    return morph_path


def create_random_morphologies(
    atlas,
    nb_morphologies: int,
    brain_regions: RegionIdsType | None = None,
    output_morphology_dir: FileType | None = None,
    output_cell_collection: FileType | None = None,
    morphology_prefix: str | None = None,
    rng: SeedType = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Create some random source points."""
    rng = np.random.default_rng(rng)

    if brain_regions is not None:
        coords, missing_ids = atlas.get_region_points(
            brain_regions, size=nb_morphologies, random_shifts=True, rng=rng
        )
        if missing_ids and logger is not None:
            logger.warning("Could not find the following regions in the atlas: %s", missing_ids)
    else:
        coords, _missing_ids = atlas.get_region_points(
            [-1, 0], size=nb_morphologies, random_shifts=True, inverse=True, rng=rng
        )

    if len(coords) < nb_morphologies and logger is not None:
        logger.error(
            "Not enough voxels found to place source points, found only %s voxels", len(coords)
        )

    dataset = pd.DataFrame(coords, columns=COORDS_COLS).reset_index()
    dataset.index += 1
    dataset.loc[:, "morphology"] = (morphology_prefix or "") + dataset.loc[:, "index"].astype(
        str
    ).str.zfill(len(str(dataset.index.max())))
    dataset.drop(columns=["index"], inplace=True)
    dataset.loc[:, "orientation"] = np.repeat([np.eye(3)], len(dataset), axis=0).tolist()
    dataset.loc[:, "atlas_id"] = atlas.brain_regions.lookup(dataset.loc[:, COORDS_COLS].to_numpy())
    dataset.loc[:, "region"] = dataset.loc[:, "atlas_id"].apply(
        lambda row: atlas.region_map.get(row, attr="acronym", with_ascendants=False)
    )

    if output_morphology_dir is not None:
        output_morphology_dir = Path(output_morphology_dir)
        output_morphology_dir.mkdir(parents=True, exist_ok=True)
        morph_file = output_morphology_dir / dataset.loc[:, "morphology"]
        dataset.loc[:, "morph_file"] = morph_file.astype(str) + ".swc"  # type: ignore[attr-defined]
        for _idx, i in dataset.iterrows():
            morph = MorphIoMorphology()
            morph.soma.points = [i[COORDS_COLS].to_numpy()]
            morph.soma.diameters = [1]
            morph.soma.type = SomaType.SOMA_SINGLE_POINT
            morph.write(i["morph_file"])

    cells = CellCollection.from_dataframe(dataset)

    if output_cell_collection is not None:
        output_cell_collection = Path(output_cell_collection)
        output_cell_collection.parent.mkdir(parents=True, exist_ok=True)
        cells.save(output_cell_collection)

    if logger is not None:
        logger.info("Generated %s random morphologies", len(dataset))

    return cells
