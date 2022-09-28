"""Plot the Steiner Tree solutions."""
import logging
from pathlib import Path

import luigi
import luigi_tools
import numpy as np
import plotly.graph_objects as go
from data_validation_framework.target import TaggedOutputLocalTarget
from morph_tool import resampling
from neurom import load_morphology
from neurom.core import Morphology
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder
from voxcell.nexus.voxelbrain import Atlas

from axon_synthesis.config import Config
from axon_synthesis.PCSF.clustering import ClusterTerminals
from axon_synthesis.PCSF.steiner_morphologies import SteinerMorphologies
from axon_synthesis.target_points import FindTargetPoints
from axon_synthesis.utils import add_camera_sync

logger = logging.getLogger(__name__)


class PlotSolutions(luigi_tools.task.WorkflowTask):
    """Task to plot the Steiner solutions."""

    input_morph_paths = luigi.OptionalPathParameter(
        description="Path to the CSV file containing the paths to the generated morphologies.",
        default=None,
    )
    output_dir = luigi.PathParameter(
        description="Output folder for figures.", default="steiner_solutions"
    )
    plot_fiber_tracts = luigi.BoolParameter(
        description=("If set to True, the fiber tracts will also be plotted."),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
    )

    def requires(self):
        reqs = {
            "generated": SteinerMorphologies(),
        }

        input_data_type = Config().input_data_type
        if input_data_type == "biological_morphologies":
            reqs["biological"] = ClusterTerminals()
        elif input_data_type == "white_matter":
            reqs["biological"] = FindTargetPoints()
        else:
            raise ValueError(f"The value of 'input_data_type' is unknown ({input_data_type}).")

        return reqs

    def run(self):
        # input_dir = self.input_morph_paths or self.input()["generated"]["morphologies"].pathlib_path
        input_morph_paths = pd.read_csv(
            self.input_morph_paths or self.input()["generated"]["morphology_paths"].path,
            dtype={"morph_file": str},
        )

        self.output().mkdir(is_dir=True)

        # ################################################################################### #
        if self.plot_fiber_tracts:
            atlas_path = "/gpfs/bbp.cscs.ch/project/proj82/entities/atlas/ThalNCX/20201019/"
            atlas_hierarchy_filename = "hierarchy.json"
            atlas_region_filename = "brain_regions"
            atlas = Atlas.open(atlas_path)
            region_map = atlas.load_region_map(atlas_hierarchy_filename)
            brain_regions = atlas.load_data(atlas_region_filename)
            fiber_tracts_ids = region_map.find("fiber tracts", attr="name", with_descendants=True)
            fiber_tracts_mask = np.isin(brain_regions.raw, list(fiber_tracts_ids))
            brain_regions.raw[~fiber_tracts_mask] = 0  # Zeroes the complement region
            fiber_tract_points = brain_regions.indices_to_positions(np.argwhere(brain_regions.raw))
            # fiber_tract_tree = KDTree(fiber_tract_points)
            fiber_tract_trace = go.Scatter3d(
                x=fiber_tract_points[::10, 0],
                y=fiber_tract_points[::10, 1],
                z=fiber_tract_points[::10, 2],
                mode="markers",
                marker={"size": 0.5, "color": "red"},
                name="Fiber tracts",
                opacity=0.1,
            )
        # ################################################################################### #

        for row in input_dir.iterrows():
            morph_file = Path(row["steiner_morph_file"])
            morph_name = morph_file.name

            gen_morph = load_morphology(morph_file)
            gen_morph = Morphology(resampling.resample_linear_density(gen_morph, 0.001))

            logger.debug("%s: %s sections", morph_name, len(gen_morph.sections))

            # Build the generated figure
            gen_builder = NeuronBuilder(gen_morph, "3d", line_width=4, title=f"{morph_name}")
            gen_fig = gen_builder.get_figure()["data"]

            # Export the solution
            input_data_type = Config().input_data_type
            if input_data_type == "biological_morphologies":
                fig = make_subplots(cols=2, specs=[[{"is_3d": True}, {"is_3d": True}]])
            else:
                fig = make_subplots(cols=1, specs=[[{"is_3d": True}]])
            if self.plot_fiber_tracts:
                fig.add_trace(
                    fiber_tract_trace,
                    row=1,
                    col=1,
                )
            fig.add_traces(
                gen_fig,
                rows=[1] * len(gen_fig),
                cols=[1] * len(gen_fig),
            )

            # Add biological figure for comparison
            if input_data_type == "biological_morphologies":
                bio_file = self.input()["biological"]["morphologies"].pathlib_path / morph_name

                logger.debug("Adding biological morphology from %s", bio_file)

                if not bio_file.exists():
                    logger.error("%s was not found in %s", morph_name, bio_file.parent)
                    continue
                bio_morph = load_morphology(bio_file)

                # Build the biological figure
                bio_builder = NeuronBuilder(bio_morph, "3d", line_width=4, title=f"{morph_name}")
                bio_fig = bio_builder.get_figure()["data"]

                fig.add_traces(
                    bio_fig,
                    rows=[1] * len(bio_fig),
                    cols=[2] * len(bio_fig),
                )

            fig_path = str((self.output().pathlib_path / morph_name).with_suffix(".html"))
            fig.write_html(fig_path)

            # Update the HTML file to synchronize the cameras between the two plots
            add_camera_sync(fig_path)

            logger.info("%s: exported to %s", morph_name, fig_path)

    def output(self):
        return TaggedOutputLocalTarget(self.output_dir, create_parent=True)
