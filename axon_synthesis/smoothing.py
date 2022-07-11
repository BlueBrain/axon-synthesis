"""Smooth the Steiner solutions."""
import json
import logging
import sys
from pathlib import Path

import luigi
import luigi_tools
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from morph_tool import resampling
from neurom import load_morphology
from neurom.core import Morphology
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder
from scipy.spatial import KDTree

from axon_synthesis.create_tuft_props import CreateTuftTerminalProperties
from axon_synthesis.PCSF.steiner_morphologies import SteinerMorphologies
from axon_synthesis.utils import add_camera_sync

logger = logging.getLogger(__name__)


class SmoothingOutputLocalTarget(TaggedOutputLocalTarget):
    """Target for smoothing outputs."""

    __prefix = Path("smoothing")  # pylint: disable=unused-private-member


class SmoothSteinerMorphologies(luigi_tools.task.WorkflowTask):
    """Task for morphology smoothing."""

    input_dir = luigi.parameter.OptionalPathParameter(
        description="Path to the input morphologies.",
        default=None,
        exists=True,
    )
    seed = luigi.NumericalParameter(
        description="The seed used by the random number generator for jittering.",
        var_type=int,
        default=0,
        min_value=0,
        max_value=sys.float_info.max,
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
            "terminal_properties": CreateTuftTerminalProperties(),
            "steiner_solutions": SteinerMorphologies(),
        }

    def run(self):
        input_dir = self.input_dir or self.input()["steiner_solutions"]["morphologies"].pathlib_path

        self.output()["figures"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies"].mkdir(parents=True, exist_ok=True, is_dir=True)

        # rng = np.random.default_rng(self.seed)

        with self.input()["terminal_properties"].open() as f:
            cluster_props_df = pd.DataFrame.from_records(json.load(f))

        for group_name, _ in cluster_props_df.groupby("morph_file"):
            raw_morph_file = Path(group_name)
            morph_name = raw_morph_file.name
            morph_file = (input_dir / morph_name).with_suffix(".asc")
            morph = load_morphology(morph_file)

            # Get reference terminals
            ref_terminal_props = cluster_props_df.loc[
                (cluster_props_df["morph_file"] == str(raw_morph_file))
            ]

            # Find the root sections of the future tufts
            tuft_roots = []
            tree = KDTree(ref_terminal_props["common_ancestor_coords"].to_list())
            for section in morph.iter():
                for i in tree.query_ball_point(section.points[-1], 1e-6):
                    tuft_roots.append((section, ref_terminal_props.iloc[i]))

            if len(tuft_roots) != len(ref_terminal_props):
                all_props = ref_terminal_props.to_dict("records")
                for props in all_props:
                    counter = 0
                    for tuft_root_props in tuft_roots:
                        if (
                            props["common_ancestor_coords"]
                            != tuft_root_props[1]["common_ancestor_coords"]
                        ):
                            counter += 1
                    if counter == len(tuft_roots):
                        logger.warning(
                            "No section could be found for the following tuft: %s", props
                        )

            # Smooth the sections but do not move the tuft roots
            raise NotImplementedError("The smoothing method is not ready to be used")
            # pylint: disable=unreachable

            # Export the new morphology
            morph_path = (
                (self.output()["morphologies"].pathlib_path / morph_name)
                .with_suffix(".asc")
                .as_posix()
            )
            logger.info("Exported morphology to %s", morph_path)
            morph.write(morph_path)

            if self.plot_debug:
                raw_morph = load_morphology(raw_morph_file)
                raw_morph = Morphology(resampling.resample_linear_density(raw_morph, 0.005))

                raw_builder = NeuronBuilder(raw_morph, "3d", line_width=4, title=f"{morph_name}")
                fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=f"{morph_name}")

                fig = make_subplots(cols=2, specs=[[{"is_3d": True}, {"is_3d": True}]])
                for col_num, data in enumerate(
                    [fig_builder.get_figure()["data"], raw_builder.get_figure()["data"]]
                ):
                    fig.add_traces(data, rows=[1] * len(data), cols=[col_num + 1] * len(data))

                # Export figure
                filepath = str(
                    self.output()["figures"].pathlib_path
                    / f"{Path(morph_name).with_suffix('').name}.html"
                )
                fig.write_html(filepath)

                add_camera_sync(filepath)
                logger.info("Exported figure to %s", filepath)

    def output(self):
        return {
            "figures": SmoothingOutputLocalTarget("figures", create_parent=True),
            "morphologies": SmoothingOutputLocalTarget("morphologies", create_parent=True),
        }
