"""Add tufts to Steiner solutions."""
import json
import logging
from copy import deepcopy
from pathlib import Path

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from morph_tool import resampling
from neurom import load_morphology
from neurom.core import Morphology
from morphio.mut import Morphology as MorphIoMorphology
from neurots.generate.tree import TreeGrower
from neurots.validator import validate_neuron_distribs
from neurots.validator import validate_neuron_params
from plotly_helper.neuron_viewer import NeuronBuilder
from plotly.subplots import make_subplots
from scipy.spatial import KDTree

from smoothing import SmoothSteinerMorphologies
from create_tuft_props import CreateTuftTerminalProperties
from PCSF.steiner_morphologies import SteinerMorphologies
from utils import add_camera_sync
from utils import append_section_recursive

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
    use_smooth_trunks = luigi.BoolParameter(
        description=(
            "If set to True, the Steiner solutions are smoothed before adding the tufts."
        ),
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
            tasks["steiner_solutions"] = SmoothSteinerMorphologies()
        else:
            tasks["steiner_solutions"] = SteinerMorphologies()
        return tasks

    def run(self):
        input_dir = self.input_dir or self.input()["steiner_solutions"].pathlib_path

        self.output()["figures"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies"].mkdir(parents=True, exist_ok=True, is_dir=True)

        rng = np.random.default_rng(self.seed)

        with self.input_parameters.open() as f:
            input_parameters = json.load(f)
            if not input_parameters.get("basal", None):
                input_parameters["basal"] = input_parameters["apical"]
        with self.input_distributions.open() as f:
            input_distributions = json.load(f)
            if not input_distributions.get("basal", None):
                input_distributions["basal"] = input_distributions["apical"]

        validate_neuron_distribs(input_distributions)
        validate_neuron_params(input_parameters)

        with self.input()["terminal_properties"].open() as f:
            cluster_props_df = pd.DataFrame.from_records(json.load(f))

        for group_name, group in cluster_props_df.groupby("morph_file"):
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
                    tuft_roots.append(
                        (
                            section,
                            ref_terminal_props.iloc[i]
                        )
                    )

            if len(tuft_roots) != len(ref_terminal_props):
                all_props = ref_terminal_props.to_dict("records")
                for props in all_props:
                    counter = 0
                    for tuft_root_props in tuft_roots:
                        if props["common_ancestor_coords"] != tuft_root_props[1]["common_ancestor_coords"]:
                            counter += 1
                    if counter == len(tuft_roots):
                        logger.warning(f"No section could be found for the following tuft: {props}")

            # Create the tufts
            for tuft_section, tuft_props in tuft_roots:
                # Create specific parameters
                params = deepcopy(input_parameters)
                params["apical"]["orientation"]["values"]["orientations"] = [tuft_props["cluster_orientation"]]
                logger.debug("Cluster_orientation: %s", tuft_props["cluster_orientation"])

                # Create specific distributions
                distrib = deepcopy(input_distributions)
                distrib["apical"]["persistence_diagram"] = [tuft_props["new_cluster_barcode"]]
                logger.debug("Cluster_barcode: %s", tuft_props["new_cluster_barcode"])

                # Grow a tuft
                new_morph = MorphIoMorphology()
                grower = TreeGrower(
                    new_morph,
                    initial_direction=tuft_props["cluster_orientation"],
                    initial_point=tuft_section.points[-1],
                    parameters=params["apical"],
                    distributions=distrib["apical"],
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
