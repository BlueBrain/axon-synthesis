"""Post-process the Steiner solutions."""
import json
import logging
from pathlib import Path

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from morph_tool import resampling
from neurom import load_morphology
from neurom.apps import morph_stats
from neurom.core import Morphology
from neurom.core.dataformat import COLS
from neurom.core.morphology import Section
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder
from scipy import interpolate
from scipy.spatial import KDTree

from axon_synthesis import seed_param
from axon_synthesis.create_tuft_props import CreateTuftTerminalProperties
from axon_synthesis.PCSF.steiner_morphologies import SteinerMorphologies
from axon_synthesis.trunk_properties import LongRangeTrunkProperties
from axon_synthesis.utils import add_camera_sync

logger = logging.getLogger(__name__)


class PostProcessingOutputLocalTarget(TaggedOutputLocalTarget):
    """Target for post-processing outputs."""

    __prefix = Path("steiner_post_processing")  # pylint: disable=unused-private-member


class PostProcessSteinerMorphologies(luigi_tools.task.WorkflowTask):
    """Task for morphology post-processing."""

    input_dir = luigi.parameter.OptionalPathParameter(
        description="Path to the input morphologies.",
        default=None,
        exists=True,
    )
    seed = seed_param("The seed used by the random number generator for jittering.")
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
            "trunk_properties": LongRangeTrunkProperties(),
            "steiner_solutions": SteinerMorphologies(),
            "terminal_properties": CreateTuftTerminalProperties(),
        }

    def run(self):
        self.output()["figures"].mkdir(parents=True, exist_ok=True, is_dir=True)
        self.output()["morphologies"].mkdir(parents=True, exist_ok=True, is_dir=True)

        rng = np.random.default_rng(self.seed)

        with self.input()["terminal_properties"].open() as f:
            cluster_props_df = pd.DataFrame.from_records(json.load(f))
        trunk_props_df = pd.read_csv(
            self.input()["trunk_properties"].path, dtype={"morph_file": str}
        )
        steiner_morphs = pd.read_csv(self.input()["steiner_solutions"]["morphology_paths"].path)
        steiner_morphs["post_processed_morph_file"] = None

        for index, row in steiner_morphs.iterrows():
            morph_file = row.loc["steiner_morph_file"]
            morph_name = Path(morph_file).name
            morph = load_morphology(morph_file)

            for axon_id, neurite in enumerate(morph.neurites):
                # Get reference terminals
                ref_terminal_props = cluster_props_df.loc[
                    (cluster_props_df["morph_file"] == str(row.loc["morph_file"]))
                    & (cluster_props_df["axon_id"] == axon_id)
                ]

                # Get reference trunk
                # TODO: Pick a trunk depending on the brain region or other variables?
                ref_trunk_props = trunk_props_df.sample().iloc[0]
                logger.debug(
                    "Ref statistics of the trunk: %s",
                    ref_trunk_props.drop(["morph_file", "axon_id"]).to_dict(),
                )

                trunk_stats = morph_stats.extract_stats(
                    neurite,
                    {
                        "neurite": {
                            "segment_lengths": {"modes": ["mean", "std"]},
                            "segment_meander_angles": {"modes": ["mean", "std"]},
                        }
                    },
                )["axon"]
                logger.debug("Current statistics of the trunk: %s", trunk_stats)

                # Find the root sections of the future tufts
                # Gather sections with unifurcations into future sections
                tree = KDTree(ref_terminal_props["common_ancestor_coords"].to_list())
                sections_to_smooth = [[]]
                for section in neurite.iter_sections(order=Section.ipreorder):
                    close_pts = tree.query_ball_point(section.points[-1, COLS.XYZ], 1e-6)
                    if len(close_pts) > 1:
                        raise ValueError(f"Several points are close to {section.points[-1]}")
                    sections_to_smooth[-1].append(section)
                    if close_pts or len(section.children) != 1:
                        sections_to_smooth.append([])

                num_true_pts = 200

                # Smooth the sections but do not move the tuft roots
                for i in sections_to_smooth:
                    if not i:
                        continue
                    coords = np.concatenate([i[0].points] + [sec.points[1:] for sec in i[1:]])
                    pts = coords[:, COLS.XYZ]
                    radii = coords[:, COLS.R]

                    # Ensure the curve is close to the first and last points
                    w = np.ones(len(pts))
                    w[0] = 1000
                    w[-1] = 1000

                    # Interpolate coordinates and radii
                    k = min(len(pts) - 1, 3)
                    tck, u = interpolate.splprep(pts.T.tolist(), w=w, k=k, s=2)
                    u_fine = np.linspace(0, 1, num_true_pts)
                    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
                    new_pts = np.array([x_fine, y_fine, z_fine], dtype=pts.dtype).T
                    new_radii = np.interp(u_fine, np.linspace(0, 1, len(radii)), radii)

                    # pylint: disable=protected-access
                    ids, fractions = resampling._resample_from_linear_density(
                        new_pts, 1.0 / ref_trunk_props["mean_segment_lengths"]
                    )
                    resampled_pts = resampling._parametric_values(new_pts, ids, fractions)
                    resampled_radii = resampling._parametric_values(new_radii, ids, fractions)

                    # TODO: Fix formula to consider both length and angles
                    jitter = rng.normal(
                        0,
                        ref_trunk_props["mean_segment_lengths"]
                        * np.tan(ref_trunk_props["std_segment_meander_angles"]),
                        size=(len(resampled_pts) - 2, 3),
                    )
                    resampled_pts[1:-1] += jitter

                    # Update section points and diameters
                    sec_pts = np.array_split(resampled_pts, len(i))
                    sec_radii = np.array_split(resampled_radii, len(i))
                    for num, sec in enumerate(i):
                        if num == 0:
                            s_pts = sec_pts[num]
                            s_diams = sec_radii[num]
                        else:
                            s_pts = np.concatenate([[sec_pts[num - 1][-1]], sec_pts[num]])
                            s_diams = np.concatenate([[sec_radii[num - 1][-1]], sec_radii[num]])

                        sec.points = np.hstack([s_pts, s_diams.reshape((len(s_pts), 1))])

                trunk_stats = morph_stats.extract_stats(
                    neurite,
                    {
                        "neurite": {
                            "segment_lengths": {"modes": ["mean", "std"]},
                            "segment_meander_angles": {"modes": ["mean", "std"]},
                        }
                    },
                )["axon"]
                logger.debug("New statistics of the trunk: %s", trunk_stats)

            # Export the new morphology
            morph_path = (
                (self.output()["morphologies"].pathlib_path / morph_name)
                .with_suffix(".asc")
                .as_posix()
            )
            morph.write(morph_path)
            steiner_morphs.loc[index, "post_processed_morph_file"] = morph_path
            logger.info("Exported morphology to %s", morph_path)

            if self.plot_debug:
                steiner_morph = load_morphology(morph_file)
                steiner_morph = Morphology(resampling.resample_linear_density(steiner_morph, 0.005))

                steiner_builder = NeuronBuilder(
                    steiner_morph, "3d", line_width=4, title=f"{morph_name}"
                )
                fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=f"{morph_name}")

                fig = make_subplots(cols=2, specs=[[{"is_3d": True}, {"is_3d": True}]])
                for col_num, data in enumerate(
                    [fig_builder.get_figure()["data"], steiner_builder.get_figure()["data"]]
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

        steiner_morphs.to_csv(self.output()["morphology_paths"].path, index=False)

    def output(self):
        return {
            "figures": PostProcessingOutputLocalTarget("figures", create_parent=True),
            "morphologies": PostProcessingOutputLocalTarget("morphologies", create_parent=True),
            "morphology_paths": PostProcessingOutputLocalTarget(
                "steiner_morph_paths.csv", create_parent=True
            ),
        }
