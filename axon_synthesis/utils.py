"""Some utils for the AxonSynthesis package."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import collections.abc
import hashlib
import inspect
import json
import logging
import re
import shutil
import tempfile
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

try:
    logging.getLogger("tecio").disabled = True
    from construct import Float32l as Float
    from construct import GreedyRange
    from tecio import TecDatasetAux
    from tecio import TecHeader
    from tecio import TecplotMarker
    from tecio import ZoneType
    from tecio import gen_data_struct
    from tecio import gen_zone_struct

    has_tecio = True
except ImportError:
    has_tecio = False


import dask.dataframe as dd
import networkx as nx
import numpy as np
import pandas as pd
from attrs import define
from attrs import field
from attrs import validators
from bluepyparallel import evaluate
from bluepyparallel import init_parallel_factory
from dask.distributed import LocalCluster
from morph_tool.converter import convert
from morph_tool.utils import is_morphology
from morphio.mut import Morphology as MorphIoMorphology
from neurom import NeuriteType
from neurom import load_morphology as neurom_load_morphology
from neurom.core import Morphology
from neurom.core.morphology import iter_neurites
from neurom.core.soma import SomaType
from neurom.geom.transform import Translation
from voxcell.cell_collection import CellCollection

from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import FROM_COORDS_COLS
from axon_synthesis.constants import TO_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.typing import RegionIdsType
from axon_synthesis.typing import SeedType

_DISTRIBUTED_LOGGERS = [
    "asyncio",
    "distributed",
    "distributed.batched",
    "distributed.core",
    "distributed.http",
    "distributed.http.proxy",
    "distributed.nanny",
    "distributed.scheduler",
    "distributed.worker",
]
LOGGER = logging.getLogger(__name__)


@define
class ParallelConfig:
    """Class to store the parallel configuration.

    Attributes:
        nb_processes: The number of processes.
        dask_config: The dask configuration to use.
        progress_bar: If set to True, a progress bar is displayed during computation.
        use_mpi: Trigger the use of MPI.
    """

    nb_processes: int = field(default=0, validator=validators.ge(0))
    dask_config: dict | None = field(default=None)
    progress_bar: bool = field(default=False)
    use_mpi: bool = field(default=False)


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


def setup_logger(level: str = "info", prefix: str = "", suffix: str = ""):
    """Setup application logger."""
    if mpi_enabled:  # pragma: no cover
        comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
        if comm.Get_size() > 1:
            rank = comm.Get_rank()
            prefix = f"#{rank} - {prefix}"
    level = level.lower()
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


class TemporaryDirectory(tempfile.TemporaryDirectory):
    """Backport of delete parameter for TemporaryDirectory."""

    def __init__(self, *args, delete=True, **kwargs):
        """Constructor of TemporaryDirectory."""
        super().__init__(*args, **kwargs)
        if not delete:
            self._finalizer.detach()  # type: ignore[attr-defined]


class CleanableDirectory:
    """A class to create a directory that can be cleaned up."""

    def __init__(self, dir_name, *, parents=True, exist_ok=False):
        """Constructor of the CleanableDirectory."""
        dir_name = Path(dir_name)
        dir_name.mkdir(parents=parents, exist_ok=exist_ok)
        self.name = str(dir_name)

    def cleanup(self):
        """Clean up the directory."""
        shutil.rmtree(self.name)


def temp_dir(*args, **kwargs):
    """Create a temporary directory."""
    return TemporaryDirectory(*args, **kwargs)


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


def get_nested_morphology_paths(morph_dir, *, max_level=None):
    """Get all morphology paths from a given directory and sub directories."""
    morph_dir = Path(morph_dir)
    morphology_paths = []
    for morph_path in morph_dir.iterdir():
        if morph_path.is_dir() and (max_level is None or max_level > 0):
            morphology_paths.extend(get_nested_morphology_paths(morph_path))
        elif is_morphology(morph_path):
            morphology_paths.append(morph_path)
    return morphology_paths


def get_morphology_paths(morph_dir, *, max_level=None):
    """Get all morphology paths from a given directory."""
    morphology_paths = get_nested_morphology_paths(morph_dir, max_level=max_level)
    paths = pd.DataFrame(morphology_paths, columns=["morph_path"])
    paths.loc[:, "morph_name"] = paths["morph_path"].apply(
        lambda x: str(x.relative_to(morph_dir).with_suffix(""))
    )
    paths.loc[:, "morph_path"] = paths["morph_path"].apply(str)
    return paths


def get_axons(morph, axon_ids=None):
    """Get axons of the given morphology."""
    axons = [i for i in morph.neurites if i.type == NeuriteType.axon]
    if axon_ids is None:
        return axons
    if isinstance(axon_ids, int):
        return axons[axon_ids]
    return [axons[i] for i in axon_ids]


def keep_only_neurites(morph, neurite_type=None, neurite_idx=None, *, copy=False):
    """Delete neurites except the ones with a given type or index.

    .. note::
        If both the type and index are given, the index is only applied for the neurites of the
        given type.
    """
    if copy:
        morph = Morphology(morph)

    if neurite_type is not None:
        for i in morph.root_sections:
            if i.type != neurite_type:
                morph.delete_section(i)

    if neurite_idx is not None:
        to_delete = []
        for num, i in enumerate(morph.root_sections):
            if num != neurite_idx:
                to_delete.append(i)
        for i in to_delete:
            morph.delete_section(i)

    return morph


def neurite_to_pts(neurite, *, keep_section_segments=False, edges_with_coords=False):
    """Extract points and segments from a neurite."""
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

    if edges_with_coords:
        edges = edges.merge(nodes, left_on="source", right_index=True).rename(
            columns={
                COORDS_COLS.X: FROM_COORDS_COLS.X,
                COORDS_COLS.Y: FROM_COORDS_COLS.Y,
                COORDS_COLS.Z: FROM_COORDS_COLS.Z,
            }
        )
        edges = edges.merge(
            nodes, left_on="target", right_index=True, suffixes=("_from", "_to")
        ).rename(
            columns={
                COORDS_COLS.X: TO_COORDS_COLS.X,
                COORDS_COLS.Y: TO_COORDS_COLS.Y,
                COORDS_COLS.Z: TO_COORDS_COLS.Z,
            }
        )

    return nodes, edges


def neurite_to_graph(neurite, graph_cls=nx.DiGraph, *, keep_section_segments=False, **graph_kwargs):
    """Transform a neurite into a graph."""
    nodes, edges = neurite_to_pts(neurite, keep_section_segments=keep_section_segments)
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


def export_morph_edges(morph, output_path, neurite_filter=None, logger=None):
    """Export the morphology as DataFrame for later analysis."""
    if logger is None:
        logger = LOGGER
    all_edges = []
    for i in iter_neurites(morph, filt=neurite_filter):
        _nodes, edges = neurite_to_pts(i, keep_section_segments=True, edges_with_coords=True)
        edges.loc[:, "neurite_type"] = i.type
        all_edges.append(edges)
    if all_edges:
        edges_df = pd.concat(all_edges, ignore_index=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        edges_df.to_csv(output_path, index=False)
        logger.debug("Exported morphology edges to '%s'", str(output_path))
    else:
        logger.debug("No edge to export to '%s'", str(output_path))


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
def disable_distributed_loggers():
    """A context manager that will disable logging messages from the 'distributed' library."""
    with disable_loggers(*_DISTRIBUTED_LOGGERS):
        yield


def permanently_disable_distributed_loggers():
    """Permanently disable logging messages from the 'distributed' library."""
    loggers = [logging.getLogger(i) for i in _DISTRIBUTED_LOGGERS]

    for i in loggers:
        i.disabled = True


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


def compute_bbox(points, relative_buffer=None, absolute_buffer=None):
    """Compute the bounding box of the given points and optionally apply a buffer to it."""
    bbox = np.vstack([points.min(axis=0), points.max(axis=0)])
    bbox_buffer = None
    if relative_buffer is not None:
        bbox_buffer = (bbox[1] - bbox[0]) * relative_buffer
    if absolute_buffer is not None and (bbox_buffer is None or absolute_buffer > bbox_buffer):
        bbox_buffer = absolute_buffer
    if bbox_buffer is not None:
        bbox[0] -= bbox_buffer
        bbox[1] += bbox_buffer
    return bbox


def compute_aspect_ratios(bbox):
    """Compute the aspect ratios of a bounding box."""
    aspect_ratios = bbox[1] - bbox[0]
    aspect_ratios /= aspect_ratios[np.argmax(aspect_ratios)]
    return aspect_ratios


def build_layout_properties(
    pts, relative_buffer: float | None = None, absolute_buffer: float | None = None
) -> dict:
    """Build a dictionary with layout properties for Plotly figures."""
    bbox = compute_bbox(pts, relative_buffer, absolute_buffer)
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
    Path(morph_path).parent.mkdir(parents=True, exist_ok=True)

    convert(morph, morph_path, nrn_order=True)
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


def create_dask_dataframe(data: pd.DataFrame, npartitions: int, group_col="morphology"):
    """Ensure all rows of the same group belong to the same partition."""
    ddf = dd.from_pandas(data, npartitions)
    if len(ddf.divisions) > 2:
        groups = np.array_split(data[group_col].unique(), npartitions)
        new_divisions = [
            data.loc[data[group_col].isin(i)].index.min() for i in groups if len(i) > 0
        ] + [data.index.max()]
        ddf = ddf.repartition(divisions=new_divisions)
    return ddf


def parallel_evaluator(
    data,
    func,
    parallel_config: ParallelConfig | None,
    new_columns: list,
    *,
    func_args=None,
    func_kwargs=None,
    startup_func=None,
    shuffle_rows: bool = True,
    progress_bar: bool = False,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Create a local cluster and process the given function on the given DataFrame."""
    if logger is None:
        logger = LOGGER
    if parallel_config is None:
        parallel_config = ParallelConfig()
    with disable_distributed_loggers():
        if parallel_config.nb_processes >= 1:
            LOGGER.info("Start parallel computation using %s workers", parallel_config.nb_processes)
            cluster = LocalCluster(n_workers=parallel_config.nb_processes, timeout="60s")
            if startup_func is not None:
                cluster.get_client().run(startup_func)
            parallel_factory = init_parallel_factory(
                "dask_dataframe",
                address=cluster,
            )
        else:
            LOGGER.info("Start computation")
            cluster = None
            shuffle_rows = False
            parallel_factory = init_parallel_factory(None)

        # Extract terminals of each morphology
        results = evaluate(
            data,
            func,
            new_columns,
            parallel_factory=parallel_factory,
            func_args=func_args,
            func_kwargs=func_kwargs,
            shuffle_rows=shuffle_rows,
            progress_bar=progress_bar,
        )

        # Close the Dask cluster if opened
        if cluster is not None:
            parallel_factory.shutdown()
            cluster.close()

    return results


def seed_from_name(name):
    """Build a seed from the name hash."""
    return int(hashlib.sha256(name.encode("ascii")).hexdigest(), 16) % (2**32 - 1)


def get_morph_pts(morph):
    """Return the points of a morphology and handle empty morphologies."""
    try:
        return morph.points
    except ValueError:
        return np.empty((0, 4), dtype=float)


def export_to_tecplot(voxcell_data, filename, title="Default Title"):
    """Export a 3D vector field to a Tecplot binary file."""
    if not has_tecio:
        msg = "The 'tecio' package must be installed to export a vector field into Tecplot format"
        raise RuntimeError(msg)
    shape = voxcell_data.shape
    indices = np.moveaxis(np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]], 0, -1)
    coords = indices * voxcell_data.voxel_dimensions + voxcell_data.offset
    x = coords[:, :, :, 0].ravel().astype(np.float32)
    y = coords[:, :, :, 1].ravel().astype(np.float32)
    z = coords[:, :, :, 2].ravel().astype(np.float32)
    ux = voxcell_data.raw[:, :, :, 0].ravel().astype(np.float32)
    uy = voxcell_data.raw[:, :, :, 1].ravel().astype(np.float32)
    uz = voxcell_data.raw[:, :, :, 2].ravel().astype(np.float32)
    length = np.linalg.norm(voxcell_data.raw, axis=-1).ravel().astype(np.float32)
    tec = {
        "title": title,
        "variables": ["X", "Y", "Z", "Ux", "Uy", "Uz", "length"],
        "zones": [
            {
                "zone_type": ZoneType.ORDERED,
                "ijk": [shape[2], shape[1], shape[0]],
            },
        ],
        "dataset_aux": [],
        "data": [
            {
                "min_max": [
                    [x.min(), x.max()],
                    [y.min(), y.max()],
                    [z.min(), z.max()],
                    [ux.min(), ux.max()],
                    [uy.min(), uy.max()],
                    [uz.min(), uz.max()],
                    [length.min(), length.max()],
                ],
                "data": [
                    x,
                    y,
                    z,
                    ux,
                    uy,
                    uz,
                    length,
                ],
            },
        ],
    }

    with Path(filename).open(mode="wb") as f:
        TecHeader.build_stream(tec, f)
        GreedyRange(gen_zone_struct(7)).build_stream(tec["zones"], f)
        GreedyRange(TecDatasetAux).build_stream(tec.get("dataset_aux", []), f)

        f.write(Float.build(TecplotMarker.EOH))
        for z, d in zip(tec["zones"], tec["data"]):
            gen_data_struct(tec["variables"], z).build_stream(d, f)
