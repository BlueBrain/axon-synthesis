"""Extract the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import logging
from pathlib import Path

import pandas as pd
from bluepyparallel import evaluate
from bluepyparallel import init_parallel_factory
from dask.distributed import LocalCluster
from morph_tool.utils import is_morphology

from axon_synthesis.typing import FileType
from axon_synthesis.utils import COORDS_COLS
from axon_synthesis.utils import ParallelConfig
from axon_synthesis.utils import disable_distributed_loggers
from axon_synthesis.utils import get_axons
from axon_synthesis.utils import load_morphology

LOGGER = logging.getLogger(__name__)


def process_morph(morph_path: FileType) -> list[tuple[str, int, int, int, float, float, float]]:
    """Extract the terminal points from a morphology."""
    morph_name = Path(morph_path).name
    morph_path_str = str(morph_path)
    morph = load_morphology(morph_path)
    pts = []
    axons = get_axons(morph)

    nb_axons = len(axons)
    LOGGER.info("%s: %s axon%s found", morph_name, nb_axons, "s" if nb_axons > 1 else "")

    for axon_id, axon in enumerate(axons):
        # Add root point
        pts.append(
            (morph_path_str, axon_id, 0, axon.root_node.id, *axon.root_node.points[0][:3].tolist()),
        )

        # Add terminal points
        terminal_id = 1
        for section in axon.iter_sections():
            if not section.children:
                pts.append(
                    (
                        morph_path_str,
                        axon_id,
                        terminal_id,
                        section.id,
                        *section.points[-1][:3].tolist(),
                    ),
                )
                terminal_id += 1

    return pts


def _wrapper(data: dict) -> dict:
    """Wrap process_morph() for parallel computation."""
    return {"res": process_morph(data["morph_path"])}


def process_morphologies(
    morph_dir: FileType, parallel_config: ParallelConfig | None = None
) -> pd.DataFrame:
    """Extract terminals from all the morphologies in the given directory."""
    if parallel_config is None:
        parallel_config = ParallelConfig()
    morph_dir = Path(morph_dir)
    morphology_paths = []
    for morph_path in morph_dir.iterdir():
        if not is_morphology(morph_path):
            continue
        morphology_paths.append(morph_path)

    morphologies = pd.DataFrame(morphology_paths, columns=["morph_path"])

    with disable_distributed_loggers():
        if parallel_config.nb_processes > 1:
            LOGGER.info("Start parallel computation using %s workers", parallel_config.nb_processes)
            cluster = LocalCluster(n_workers=parallel_config.nb_processes, timeout="60s")
            parallel_factory = init_parallel_factory("dask_dataframe", address=cluster)
        else:
            LOGGER.info("Start computation")
            parallel_factory = init_parallel_factory(None)

        # Extract terminals of each morphology
        results = evaluate(
            morphologies,
            _wrapper,
            [
                ["res", None],
            ],
            parallel_factory=parallel_factory,
        )

        # Close the Dask cluster if opened
        if parallel_config.nb_processes > 1:
            parallel_factory.shutdown()
            cluster.close()

    final_results = results["res"].explode().apply(pd.Series)
    final_results.columns = ["morph_file", "axon_id", "terminal_id", "section_id", *COORDS_COLS]
    return final_results
