"""Add tufts to Steiner solutions."""
import json
import logging
from copy import deepcopy
from pathlib import Path

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from morph_tool import resampling
from morphio.mut import Morphology as MorphIoMorphology
from neurom import load_morphology
from neurom.core import Morphology
from neurots.generate.tree import TreeGrower
from neurots.validator import validate_neuron_distribs
from neurots.validator import validate_neuron_params
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder
from scipy.spatial import KDTree

from axon_synthesis import seed_param
from axon_synthesis.config import Config
from axon_synthesis.create_tuft_props import CreateTuftTerminalProperties
from axon_synthesis.PCSF.post_process import PostProcessSteinerMorphologies
from axon_synthesis.PCSF.steiner_morphologies import SteinerMorphologies
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import append_section_recursive

logger = logging.getLogger(__name__)


def plot_tuft(morph, morph_name, output_path, morph_file=None):
    """Plot the given morphology.

    If `morph_file` is not None then the given morphology is also plotted for comparison.
    """
    fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=f"{morph_name}")
    fig_data = [fig_builder.get_figure()["data"]]

    if morph_file is not None:
        raw_morph = load_morphology(morph_file)
        raw_morph = Morphology(resampling.resample_linear_density(raw_morph, 0.005))

        raw_builder = NeuronBuilder(raw_morph, "3d", line_width=4, title=f"{morph_name}")

        fig = make_subplots(cols=2, specs=[[{"is_3d": True}, {"is_3d": True}]])
        fig_data.append(raw_builder.get_figure()["data"])
    else:
        fig = make_subplots(cols=1, specs=[[{"is_3d": True}]])

    for col_num, data in enumerate(fig_data):
        fig.add_traces(data, rows=[1] * len(data), cols=[col_num + 1] * len(data))

    # Export figure
    fig.write_html(output_path)

    add_camera_sync(output_path)
    logger.info("Exported figure to %s", output_path)


class TuftsTarget(TaggedOutputLocalTarget):
    """Target for tuft outputs."""

    __prefix = Path("tufts")  # pylint: disable=unused-private-member


class TuftMorphologiesTarget(TuftsTarget):
    """Target for tuft outputs."""

    __prefix = Path("morphologies")  # pylint: disable=unused-private-member


class AddTufts(luigi_tools.task.WorkflowTask):
    """Task to add a tuft at each target point."""

    input_dir = luigi.parameter.OptionalPathParameter(
        description="Path to the input morphologies.",
        default=None,
        exists=True,
    )
    input_parameters = luigi.parameter.PathParameter(
        description="Path to the input parameters.",
        default="tuft_parameters.json",
        exists=True,
    )
    input_distributions = luigi.parameter.PathParameter(
        description="Path to the input distributions.",
        default="tuft_distributions.json",
        exists=True,
    )
    seed = seed_param()
    use_smooth_trunks = luigi.BoolParameter(
        description=("If set to True, the Steiner solutions are smoothed before adding the tufts."),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
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
        tasks = {
            "terminal_properties": CreateTuftTerminalProperties(),
        }
        if self.use_smooth_trunks:
            tasks["steiner_solutions"] = PostProcessSteinerMorphologies()
        else:
            tasks["steiner_solutions"] = SteinerMorphologies()
        return tasks

    @staticmethod
    def check_props(tuft_roots, ref_terminal_props):
        """Check tuft properties."""
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
                    logger.warning("No section could be found for the following tuft: %s", props)

    def run(self):
        config = Config()
        # input_dir = (
        #     self.input_dir
        #     or self.input()["steiner_solutions"]["morphologies"].pathlib_path
        # )

        self.output()["figures"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies_asc"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies_h5"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies_swc"].mkdir(parents=True, exist_ok=True, is_dir=True)

        rng = np.random.default_rng(self.seed)

        with self.input_parameters.open() as f:  # pylint: disable=no-member
            input_parameters = json.load(f)
            if not input_parameters.get("basal_dendrite", None):
                input_parameters["basal_dendrite"] = input_parameters["axon"]
        with self.input_distributions.open() as f:  # pylint: disable=no-member
            input_distributions = json.load(f)
            if not input_distributions.get("basal_dendrite", None):
                input_distributions["basal_dendrite"] = input_distributions["axon"]

        validate_neuron_distribs(input_distributions)
        validate_neuron_params(input_parameters)

        with self.input()["terminal_properties"].open() as f:
            cluster_props_df = pd.DataFrame.from_records(json.load(f))

        if self.use_smooth_trunks:
            post_processed_paths = pd.read_csv(
                self.input()["steiner_solutions"]["morphology_paths"].path,
                dtype={"morph_file": str},
            )
            cluster_props_df["steiner_morph_file"] = pd.merge(
                cluster_props_df, post_processed_paths, on="morph_file"
            )["post_processed_morph_file"]

        for group_name, group in cluster_props_df.groupby("steiner_morph_file"):
            steiner_morph_file = Path(group_name)
            morph_file = Path(group["morph_file"].iloc[0])
            morph_name = steiner_morph_file.name
            morph = load_morphology(steiner_morph_file)

            # Get reference terminals
            ref_terminal_props = cluster_props_df.loc[
                (cluster_props_df["morph_file"] == str(morph_file))
            ]

            # Find the root sections of the future tufts
            # (all unifurcations may not be actual tuft roots, so we have to retrieve them using
            # the coordinates)
            tuft_roots = []
            tree = KDTree(
                np.array(ref_terminal_props["common_ancestor_coords"].to_list(), dtype=np.float32)
            )
            for section in morph.iter():
                for i in tree.query_ball_point(section.points[-1], 1e-6):
                    tuft_roots.append((section, ref_terminal_props.iloc[i]))

            self.check_props(tuft_roots, ref_terminal_props)

            # Create the tufts
            for tuft_section, tuft_props in tuft_roots:
                # Create specific parameters
                params = deepcopy(input_parameters)
                params["basal_dendrite"]["orientation"]["values"]["orientations"] = [
                    tuft_props["cluster_orientation"]
                ]
                logger.debug("Cluster_orientation: %s", tuft_props["cluster_orientation"])

                # Create specific distributions
                distrib = deepcopy(input_distributions)
                distrib["basal_dendrite"]["persistence_diagram"] = [
                    tuft_props["new_cluster_barcode"]
                ]
                logger.debug("Cluster_barcode: %s", tuft_props["new_cluster_barcode"])

                # Grow a tuft
                new_morph = MorphIoMorphology()
                grower = TreeGrower(
                    new_morph,
                    initial_direction=tuft_props["cluster_orientation"],
                    initial_point=tuft_section.points[-1],
                    parameters=params["basal_dendrite"],
                    distributions=distrib["basal_dendrite"],
                    context=None,
                    random_generator=rng,
                )
                while not grower.end():
                    grower.next_point()

                # Graft the tuft to the current terminal
                append_section_recursive(tuft_section, new_morph.root_sections[0])

            # Merge consecutive sections that are not separated by a bifurcation
            morph.remove_unifurcations()

            # Export the new morphology
            for ext in [".asc", ".h5", ".swc"]:
                morph_path = (
                    (self.output()["morphologies_" + ext[1:]].pathlib_path / morph_name)
                    .with_suffix(ext)
                    .as_posix()
                )
                logger.info("Exported morphology to %s", morph_path)
                morph.write(morph_path)

            if self.plot_debug:
                plot_tuft(
                    morph,
                    morph_name,
                    self.output()["figures"].pathlib_path
                    / f"{Path(morph_name).with_suffix('').name}.html",
                    morph_file=morph_file
                    if config.input_data_type == "biological_morphologies"
                    else None,
                )

    def output(self):
        return {
            "figures": TuftsTarget("figures", create_parent=True),
            "morphologies_swc": TuftMorphologiesTarget("swc", create_parent=True),
            "morphologies_h5": TuftMorphologiesTarget("h5", create_parent=True),
            "morphologies_asc": TuftMorphologiesTarget("asc", create_parent=True),
        }
