"""Cluster the terminal points of a morphology so that a Steiner Tree can be computed on them."""
import json
import logging
from pathlib import Path

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from morph_tool import resampling
from neurom import load_morphology
from neurom.core import Morphology
from neurom.geom import translate
from plotly_helper.neuron_viewer import NeuronBuilder
from plotly.subplots import make_subplots
from tns import NeuronGrower
from tns.validator import validate_neuron_distribs
from tns.validator import validate_neuron_params

from create_dataset import RepairDataset
from PCSF.extract_terminals import ExtractTerminals
from PCSF.steiner_morphologies import SteinerMorphologies
from utils import add_camera_sync
from utils import append_section_recursive
from utils import get_axons

logger = logging.getLogger(__name__)


class TuftsOutputLocalTarget(luigi_tools.target.OutputLocalTarget):
    __prefix = Path("tufts")


class AddTufts(luigi_tools.task.WorkflowTask):
    input_dir = luigi_tools.parameter.OptionalPathParameter(
        description="Path to the input morphologies.",
        default=None,
        exists=True,
    )
    input_parameters = luigi_tools.parameter.PathParameter(
        description="Path to the input parameters.",
        default="tuft_parameters.json",
        exists=True,
    )
    input_distributions = luigi_tools.parameter.PathParameter(
        description="Path to the input distributions.",
        default="tuft_distributions.json",
        exists=True,
    )
    seed = luigi.NumericalParameter(
        description="The seed used by the random number generator.",
        var_type=int,
        default=0,
        min_value=0,
        max_value=float("inf"),
    )
    plot_debug = luigi.BoolParameter(
        description=(
            "If set to True, each group will create an interactive figure so it is possible to "
            "check the clustering parameters."
        ),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
    )

    def requires(self):
        return {
            "steiner_solutions": SteinerMorphologies(),
            # "raw_morphologies": RepairDataset(),
            "terminals": ExtractTerminals(),
        }

    def run(self):
        input_dir = self.input_dir or self.input()["steiner_solutions"].pathlib_path

        self.output()["figures"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies"].mkdir(parents=True, exist_ok=True, is_dir=True)

        rng = np.random.default_rng(self.seed)

        with self.input_parameters.open() as f:
            input_parameters = json.load(f)
        with self.input_distributions.open() as f:
            input_distributions = json.load(f)

        validate_neuron_distribs(input_distributions)
        validate_neuron_params(input_parameters)

        terminals = pd.read_csv(self.input()["terminals"].path)

        for raw_morph_file in terminals["morph_file"].apply(Path).unique():
            morph_name = raw_morph_file.name
            morph_file = (input_dir / morph_name).with_suffix(".asc")
            morph = load_morphology(morph_file)
            axons = get_axons(morph)

            for axon_id, axon in enumerate(axons):
                # Find the terminals
                terminals = []
                for section in axon.iter_sections():
                    if not section.children:
                        terminals.append(
                            [str(morph_file), axon_id, section.id, section] + section.points[-1][:3].tolist()
                        )

                # Create the tufts
                for current_terminal in terminals:
                    # Grow a tuft
                    grower = NeuronGrower(
                        input_parameters=input_parameters,
                        input_distributions=input_distributions,
                        skip_validation=True,
                        rng_or_seed=rng,
                    )
                    grower.grow()
                    new_morph = Morphology(grower.neuron)

                    # Translate the tuft to the terminal point
                    new_morph = translate(
                        new_morph,
                        np.array(current_terminal[4: 7]) - new_morph.root_sections[0].points[0],
                    )

                    # Graft the tuft to the current terminal
                    append_section_recursive(current_terminal[3], new_morph.root_sections[0])

            # Export the new morphology
            morph_path = (self.output()["morphologies"].pathlib_path / morph_name).with_suffix(".asc").as_posix()
            logger.info(f"Exported morphology to {morph_path}")
            morph.write(morph_path)

            if self.plot_debug:
                raw_morph = load_morphology(raw_morph_file)
                raw_morph = Morphology(resampling.resample_linear_density(raw_morph, 0.005))

                raw_builder = NeuronBuilder(
                    raw_morph, "3d", line_width=4, title=f"{morph_name}"
                )
                fig_builder = NeuronBuilder(
                    morph, "3d", line_width=4, title=f"{morph_name}"
                )

                fig = make_subplots(cols=2, specs=[[{"is_3d": True}, {"is_3d": True}]])
                for col_num, data in enumerate([fig_builder.get_figure()["data"], raw_builder.get_figure()["data"]]):
                    fig.add_traces(data, rows=[1] * len(data), cols=[col_num + 1] * len(data))

                # Export figure
                filepath = str(
                    self.output()["figures"].pathlib_path
                    / f"{Path(morph_name).with_suffix('').name}.html"
                )
                fig.write_html(filepath)

                add_camera_sync(filepath)
                logger.info(f"Exported figure to {filepath}")

    def output(self):
        return {
            "figures": TuftsOutputLocalTarget("figures", create_parent=True),
            "morphologies": TuftsOutputLocalTarget("morphologies", create_parent=True),
        }
