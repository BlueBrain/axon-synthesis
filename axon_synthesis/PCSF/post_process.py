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
from neurom import morphmath
from neurom.apps import morph_stats
from neurom.core import Morphology
from neurom.core.dataformat import COLS
from neurom.core.morphology import Section
from neurots.morphmath import rotation
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder
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


def get_random_vector(
    D=1.0, norm=None, std=None, initial_theta=None, initial_phi=None, rng=np.random
):
    """Return 3-d coordinates of a new random point.

    The distance between the produced point and (0,0,0) is given by the value D.
    """
    # pylint: disable=assignment-from-no-return
    phi = rng.uniform(0.0, 2.0 * np.pi)
    if norm is not None and std is not None:
        theta = rng.normal(norm, std)
    else:
        theta = np.arccos(rng.uniform(-1.0, 1.0))

    if initial_theta:
        theta += initial_theta
    if initial_phi:
        phi += initial_phi

    sn_theta = np.sin(theta)

    x = D * np.cos(phi) * sn_theta
    y = D * np.sin(phi) * sn_theta
    z = D * np.cos(theta)

    return np.array((x, y, z))


def weights(lengths, history_path_length):
    """Compute the weights depending on the lengths."""
    return np.exp(np.array(lengths) - history_path_length)


def history(latest_lengths, latest_directions, history_path_length):
    """Returns a combination of the segments history."""
    if not latest_directions:
        return np.zeros(3)

    weighted_history = np.dot(weights(latest_lengths, history_path_length), latest_directions)

    distance = np.linalg.norm(weighted_history)
    if distance > 1e-8:
        weighted_history /= distance

    return weighted_history


def closest_seg_pt(pt, seg):
    """Compute the closest point on a line from a given point."""
    closest_pt = None
    u = seg[0] - pt
    v = seg[1] - seg[0]
    t = -np.dot(v, u) / np.dot(v, v)
    closest_pt = (1 - t) * seg[0] + t * seg[1]
    return closest_pt, t


def random_walk(
    starting_pt,
    intermediate_pts,
    length_stats,
    angle_stats,
    history_path_length,
    previous_history=None,
    global_target_coeff=0.5,
    target_coeff=2,
    random_coeff=3.5,
    history_coeff=2,
    rng=np.random,
    debug=False,
):
    """Perform a random walk guided by intermediate points."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    length_norm = length_stats["norm"]
    length_std = length_stats["std"]
    angle_norm = angle_stats["norm"]
    angle_std = angle_stats["std"]

    # Select dtype of computation
    try:
        dtype = starting_pt.dtype
    except AttributeError:
        try:
            dtype = intermediate_pts.dtype
        except AttributeError:
            dtype = float

    current_pt = np.array(starting_pt, dtype=dtype)
    intermediate_pts = np.array(intermediate_pts, dtype=dtype)
    new_pts = [current_pt]

    tree = KDTree(intermediate_pts)

    total_length = 0
    last_index = 0

    # Compute the length of each segment
    segment_lengths = morphmath.interval_lengths(np.vstack([starting_pt, intermediate_pts]))
    segment_path_lengths = np.cumsum(segment_lengths)

    # Compute the direction to the last target
    global_target_direction = intermediate_pts[-1] - current_pt
    global_target_dist = np.linalg.norm(global_target_direction)
    global_target_direction /= global_target_dist

    # Setup initial history
    target_direction = intermediate_pts[0] - current_pt
    target_direction /= np.linalg.norm(target_direction)
    if previous_history:
        latest_lengths, latest_directions = previous_history
    else:
        nb_hist = int(history_path_length // length_norm)
        latest_lengths = [length_norm] * nb_hist
        latest_directions = [target_direction] * nb_hist

    nb_intermediate_pts = len(intermediate_pts)

    min_target_dist = global_target_dist * 2

    logger.debug(
        (
            "In random walk:\n\t"
            "global_target_dist=%s\n\t"
            "global_target_direction=%s\n\t"
            "target_direction=%s\n\t"
            "current_pt=%s\n\t"
            "intermediate_pts=%s\n\t"
        ),
        global_target_dist,
        global_target_direction,
        target_direction,
        current_pt,
        intermediate_pts,
    )

    while global_target_dist >= length_norm:
        step_length = rng.normal(length_norm, length_std)

        # Compute the direction to the last target
        global_target_direction = intermediate_pts[-1] - current_pt
        global_target_dist = np.linalg.norm(global_target_direction)
        global_target_direction /= global_target_dist

        target_dist, target_index = tree.query(current_pt)
        # TODO: Should check that there is no other possible target between the last target and
        # this target that is further to this target.

        # Find next targets
        if target_index < last_index or np.isinf(target_dist):
            target_index = last_index

        next_closest_pt = intermediate_pts[-1]
        next_side = 0
        if target_index < nb_intermediate_pts - 1:
            next_closest_pt, next_side = closest_seg_pt(
                current_pt,
                (intermediate_pts[target_index], intermediate_pts[target_index + 1]),
            )
            if target_index > 0:
                _, previous_side = closest_seg_pt(
                    current_pt,
                    (intermediate_pts[target_index - 1], intermediate_pts[target_index]),
                )
            else:
                previous_side = 0
            if previous_side >= 1 or (
                next_side < 0
                and np.linalg.norm(next_closest_pt - intermediate_pts[target_index])
                <= 2 * step_length
            ):
                target_index += 1
                target_dist = np.linalg.norm(intermediate_pts[target_index] - current_pt)
                min_target_dist = target_dist * 2
                if target_index < nb_intermediate_pts - 1:
                    next_closest_pt, next_side = closest_seg_pt(
                        current_pt,
                        (intermediate_pts[target_index], intermediate_pts[target_index + 1]),
                    )

        last_index = target_index

        next_target_index = min(nb_intermediate_pts - 1, target_index + 1)

        current_target_coeff = np.exp(
            -np.linalg.norm(next_closest_pt - intermediate_pts[target_index]) / (2 * step_length)
        )

        target_direction = (1 - current_target_coeff) * (
            intermediate_pts[target_index] - current_pt
        ) + current_target_coeff * (intermediate_pts[next_target_index] - current_pt)

        # target_direction = np.dot(
        #     np.exp(-np.arange(nb_intermediate_pts - target_index)),
        #     (intermediate_pts[target_index:] - current_pt),
        # )
        target_direction = np.dot(
            np.exp(
                -np.maximum(segment_path_lengths[target_index:] - total_length, 0)
                / segment_lengths[target_index]
            ),
            (intermediate_pts[target_index:] - current_pt),
        )

        target_direction /= np.linalg.norm(target_direction)

        step_global_target_coeff = global_target_coeff * max(
            0,
            1 + 2 * np.exp(-global_target_dist / (10 * step_length)),
        )
        step_target_coeff = target_coeff * max(
            0,
            1
            + np.exp(-target_dist / (2 * step_length))
            - np.exp(-total_length / (2 * step_length)),
        )
        step_random_coeff = random_coeff
        step_history_coeff = history_coeff * max(
            0,
            1
            - np.exp(-target_dist / (2 * step_length))
            + np.exp(-total_length / (2 * step_length)),
        )

        history_direction = history(latest_lengths, latest_directions, history_path_length)
        initial_phi, initial_theta = rotation.spherical_from_vector(latest_directions[-1])
        # initial_phi, initial_theta = rotation.spherical_from_vector(target_direction)
        random_direction = get_random_vector(
            norm=angle_norm,
            std=angle_std,
            initial_theta=initial_theta,
            initial_phi=initial_phi,
            rng=rng,
        )

        direction = (
            step_global_target_coeff * global_target_direction
            + step_target_coeff * target_direction
            + step_random_coeff * random_direction
            + step_history_coeff * history_direction
        ).astype(dtype)
        direction /= np.linalg.norm(direction)

        current_pt = current_pt + direction * step_length
        total_length += step_length

        if target_dist >= min_target_dist + 10 * length_norm:
            logger.warning("The random walk is going away from the target")
            debug = True
        else:
            min_target_dist = min(min_target_dist, target_dist)

        if debug:
            actual_target_direction = intermediate_pts[target_index] - current_pt
            actual_target_direction /= np.linalg.norm(actual_target_direction)
            logger.debug(
                (
                    "In random walk:\n\t"
                    "global_target_dist=%s\n\t"
                    "global_target_direction=%s\n\t"
                    "total_length=%s\n\t"
                    "step_length=%s\n\t"
                    "target_index=%s\n\t"
                    "target_dist=%s\n\t"
                    "target_direction=%s\n\t"
                    "actual_target_direction=%s\n\t"
                    "next_target_index=%s\n\t"
                    "random_direction=%s\n\t"
                    "history_direction=%s\n\t"
                    "step_global_target_coeff=%s\n\t"
                    "step_target_coeff=%s\n\t"
                    "step_random_coeff=%s\n\t"
                    "step_history_coeff=%s\n\t"
                    "direction=%s\n\t"
                    "diff_direction=%s\n\t"
                    "current_pt=%s\n\t"
                ),
                global_target_dist,
                global_target_direction,
                total_length,
                step_length,
                target_index,
                target_dist,
                target_direction,
                actual_target_direction,
                next_target_index,
                random_direction,
                history_direction,
                step_global_target_coeff,
                step_target_coeff,
                step_random_coeff,
                step_history_coeff,
                direction,
                morphmath.angle_between_vectors(direction, target_direction),
                current_pt,
            )
        new_pts.append(current_pt)
        latest_lengths.append(step_length)
        latest_directions.append(direction)
        while sum(latest_lengths) >= history_path_length:
            latest_lengths.pop(0)
            latest_directions.pop(0)

    new_pts.append(intermediate_pts[-1])

    return np.array(new_pts, dtype=dtype), (latest_lengths, latest_directions)


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
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
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
                    ref_trunk_props.drop(
                        ["morph_file", "axon_id"]
                        + [row for row in ref_trunk_props.index if row.startswith("raw_")]
                    ).to_dict(),
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

                # Smooth the sections but do not move the tuft roots
                parent_histories = {}
                for i in sections_to_smooth:
                    if not i:
                        continue
                    coords = np.concatenate([i[0].points] + [sec.points[1:] for sec in i[1:]])
                    pts = coords[:, COLS.XYZ]
                    radii = coords[:, COLS.R]

                    length_stats = {
                        "norm": ref_trunk_props["mean_segment_lengths"],
                        "std": ref_trunk_props["std_segment_lengths"],
                    }
                    angle_stats = {
                        "norm": ref_trunk_props["mean_segment_meander_angles"],
                        "std": ref_trunk_props["std_segment_meander_angles"],
                    }
                    history_length = 10 * length_stats["norm"]
                    if i[0].parent:
                        parent_history = parent_histories[i[0].parent]
                    else:
                        parent_history = None

                    resampled_pts, last_history = random_walk(
                        pts[0],
                        pts[1:],
                        length_stats,
                        angle_stats,
                        history_length,
                        parent_history,
                        rng=rng,
                    )
                    parent_histories[i[-1]] = last_history

                    path_lengths = np.insert(
                        np.cumsum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)), 0, 0
                    )
                    new_path_lengths = np.insert(
                        np.cumsum(np.linalg.norm(resampled_pts[1:] - resampled_pts[:-1], axis=1)),
                        0,
                        0,
                    )
                    resampled_radii = np.interp(
                        new_path_lengths / new_path_lengths[-1],
                        path_lengths / path_lengths[-1],
                        radii,
                    )

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
