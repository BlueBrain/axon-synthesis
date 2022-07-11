"""Plot the Steiner Tree solutions."""
import logging
import re

import luigi_tools
from luigi_tools.parameter import OptionalPathParameter
from luigi_tools.parameter import PathParameter
from neurom import load_morphology
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from create_dataset import RepairDataset
from PCSF.clustering import ClusterTerminals
from PCSF.steiner_morphologies import SteinerMorphologies
from utils import add_camera_sync

logger = logging.getLogger(__name__)


class PlotSolutions(luigi_tools.task.WorkflowTask):
    input_dir = OptionalPathParameter(
        description="Path to the generated morphologies.", default=None
    )
    output_dir = PathParameter(
        description="Output folder for figures.", default="steiner_solutions"
    )

    def requires(self):
        return {
            # "biological": RepairDataset(),
            "biological": ClusterTerminals(),
            "generated": SteinerMorphologies(),
        }

    def run(self):
        input_dir = self.input_dir or self.input()["generated"].pathlib_path

        self.output().mkdir(is_dir=True)

        for morph_file in input_dir.iterdir():
            if morph_file.suffix.lower() not in [".asc", ".h5", ".swc"]:
                continue

            morph_name = morph_file.name
            bio_file = self.input()["biological"]["morphologies"].pathlib_path / morph_name
            if not bio_file.exists():
                logger.error(f"{morph_name} was not found in {bio_file.parent}")
                continue

            bio_morph = load_morphology(bio_file)
            gen_morph = load_morphology(morph_file)

            logger.debug(f"{morph_name}: {len(gen_morph.sections)} sections")

            # Build the biological figure
            bio_builder = NeuronBuilder(bio_morph, "3d", line_width=4, title=f"{morph_name}")
            bio_fig = bio_builder.get_figure()

            # Build the generated figure
            gen_builder = NeuronBuilder(gen_morph, "3d", line_width=4, title=f"{morph_name}")
            gen_fig = gen_builder.get_figure()

            # Export the solution
            fig = make_subplots(cols=2, specs=[[{"is_3d": True}, {"is_3d": True}]])
            fig.add_traces(
                gen_fig["data"],
                rows=[1] * len(gen_fig["data"]),
                cols=[1] * len(gen_fig["data"]),
            )
            fig.add_traces(
                bio_fig["data"],
                rows=[1] * len(bio_fig["data"]),
                cols=[2] * len(bio_fig["data"]),
            )
            fig_path = str((self.output().pathlib_path / morph_name).with_suffix(".html"))
            fig.write_html(fig_path)

            # Update the HTML file to synchronize the cameras between the two plots
            add_camera_sync(fig_path)

            logger.info(f"{morph_name}: exported to {fig_path}")

    def output(self):
        return luigi_tools.target.OutputLocalTarget(self.output_dir, create_parent=True)
