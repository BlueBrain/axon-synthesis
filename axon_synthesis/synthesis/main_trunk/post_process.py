"""Post-process the Steiner solutions."""
import logging
from itertools import chain

import numpy as np
import pandas as pd
from neurom import morphmath
from neurom.apps import morph_stats
from neurom.core import Morphology
from neurom.core import Neurite
from neurots.morphmath import rotation
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import add_camera_sync
from axon_synthesis.utils import sublogger

WEIGHT_DISTANCE_TOLERANCE = 1e-8


def get_random_vector(
    distance=1.0,
    norm=None,
    std=None,
    initial_theta=None,
    initial_phi=None,
    rng=np.random,
):
    """Return 3-d coordinates of a new random point.

    The distance between the produced point and (0,0,0) is given by the 'distance' argument.
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

    x = distance * np.cos(phi) * sn_theta
    y = distance * np.sin(phi) * sn_theta
    z = distance * np.cos(theta)

    return np.array((x, y, z))


def weights(lengths, history_path_length):
    """Compute the weights depending on the lengths."""
    return np.exp(np.append(np.cumsum(lengths[:-1]) - history_path_length + lengths[0], 0))


def history(latest_lengths, latest_directions, history_path_length):
    """Returns a combination of the segments history."""
    if not latest_directions:
        return np.zeros(3)
    weighted_history = np.dot(weights(latest_lengths, history_path_length), latest_directions)

    distance = np.linalg.norm(weighted_history)
    if distance > WEIGHT_DISTANCE_TOLERANCE:
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
    # angle_stats,
    previous_history=None,
    history_path_length=None,
    global_target_coeff=0,
    target_coeff=2,
    random_coeff=2,
    history_coeff=2,
    *,
    rng=np.random,
    debug=False,
    logger=None,
):
    """Perform a random walk guided by intermediate points."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    logger = sublogger(logger, __name__)

    length_norm = length_stats["norm"]
    length_std = length_stats["std"]
    # angle_norm = angle_stats["norm"]
    # angle_std = angle_stats["std"]

    if history_path_length is None:
        history_path_length = 5.0 * length_norm

    # Select dtype of computation
    dtype = float

    current_pt = np.array(starting_pt, dtype=dtype)
    intermediate_pts = np.array(intermediate_pts, dtype=dtype)
    new_pts = [current_pt]

    total_length = 0

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
        intermediate_pts.tolist(),
    )

    target_index = 0
    target = intermediate_pts[target_index]
    next_target_index = min(nb_intermediate_pts - 1, 1)

    while global_target_dist >= length_norm:
        step_length = rng.normal(length_norm, length_std)

        # Compute the direction to the last target
        global_target_direction = intermediate_pts[-1] - current_pt
        global_target_dist = np.linalg.norm(global_target_direction)
        global_target_direction /= global_target_dist

        # TODO: Should check that there is no other possible target between the last target and
        # this target that is further to this target.
        target_vec = target - current_pt
        target_dist = np.linalg.norm(target_vec)

        next_target_vec = intermediate_pts[next_target_index] - current_pt

        current_target_coeff = np.exp(-target_dist / (1 * step_length))

        target_direction = (
            1 - current_target_coeff
        ) * target_vec + current_target_coeff * next_target_vec

        target_direction /= np.linalg.norm(target_direction)

        step_global_target_coeff = global_target_coeff * max(
            0,
            1 + 2 * np.exp(-global_target_dist / (10 * step_length)),  # More when closer
        )
        step_target_coeff = target_coeff * max(
            0,
            1
            + np.exp(-target_dist / (2 * step_length))  # More near targets to pass closer
            - np.exp(-total_length / (2 * step_length)),  # Less at the beginning
        )
        step_history_coeff = history_coeff * max(
            0,
            1
            - np.exp(-target_dist / (2 * step_length))  # Less near targets to pass closer
            + np.exp(-total_length / (2 * step_length)),  # More at the beginning
        )
        step_random_coeff = random_coeff

        history_direction = history(latest_lengths, latest_directions, history_path_length)
        initial_phi, initial_theta = rotation.spherical_from_vector(latest_directions[-1])

        direction = -target_direction
        nb_rand = 0
        max_rand = 10
        non_random_direction = (
            step_global_target_coeff * global_target_direction
            + step_target_coeff * target_direction
            + step_history_coeff * history_direction
        ).astype(dtype)

        # If the non random part of the direction does not head to the target direction
        # (e.g. because of the history), then we don't care if the resulting direction
        # does not head to the target direction
        heading_target = not np.dot(target_direction, non_random_direction) < 0

        while np.dot(direction, target_direction) < 0 and nb_rand < max_rand:
            if nb_rand > 0:
                step_random_coeff = step_random_coeff / 2.0
            random_direction = get_random_vector(rng=rng)

            direction = (non_random_direction + step_random_coeff * random_direction).astype(dtype)

            if nb_rand == 0 and not heading_target:
                break

            nb_rand += 1

        direction /= np.linalg.norm(direction)

        if debug:
            actual_target_direction = target_vec / np.linalg.norm(target_vec)
            composite_target_dist = (
                1 - current_target_coeff
            ) * target_dist + current_target_coeff * np.linalg.norm(next_target_vec)
            logger.debug(
                (
                    "In random walk:\n\t"
                    "global_target_dist=%s\n\t"
                    "global_target_direction=%s\n\t"
                    "total_length=%s\n\t"
                    "step_length=%s\n\t"
                    "target_index=%s\n\t"
                    "target=%s\n\t"
                    "target_vec=%s\n\t"
                    "target_dist=%s\n\t"
                    "target_direction=%s\n\t"
                    "current_target_coeff=%s\n\t"
                    "actual_target_direction=%s\n\t"
                    "composite_target_dist=%s\n\t"
                    "next_target_index=%s\n\t"
                    "next_target_vec=%s\n\t"
                    "random_direction=%s\n\t"
                    "(initial_phi, initial_theta)=%s\n\t"
                    "history_direction=%s\n\t"
                    "step_global_target_coeff=%s\n\t"
                    "step_target_coeff=%s\n\t"
                    "step_random_coeff=%s\n\t"
                    "step_history_coeff=%s\n\t"
                    "direction=%s\n\t"
                    "(phi, theta)=%s\n\t"
                    "diff_direction=%s\n\t"
                    "diff_actual_target_direction=%s\n\t"
                    "diff_last_direction=%s\n\t"
                    "current_pt=%s\n\t"
                ),
                global_target_dist,
                rotation.spherical_from_vector(global_target_direction),
                total_length,
                step_length,
                target_index,
                target,
                target_vec,
                target_dist,
                rotation.spherical_from_vector(target_direction),
                current_target_coeff,
                rotation.spherical_from_vector(actual_target_direction),
                composite_target_dist,
                next_target_index,
                next_target_vec,
                rotation.spherical_from_vector(random_direction),
                (initial_phi, initial_theta),
                rotation.spherical_from_vector(history_direction),
                step_global_target_coeff,
                step_target_coeff,
                step_random_coeff,
                step_history_coeff,
                direction,
                rotation.spherical_from_vector(direction),
                morphmath.angle_between_vectors(direction, target_direction),
                morphmath.angle_between_vectors(direction, actual_target_direction),
                morphmath.angle_between_vectors(direction, latest_directions[-1]),
                current_pt,
            )

        if target_dist >= min_target_dist or target_dist <= length_norm:
            if target_dist >= min_target_dist:
                logger.debug("The random walk is going away from the target")
            if target_dist <= length_norm:
                logger.debug("The random walk reached the target %s", target_index)
            logger.debug(
                "Changing target from %s to %s",
                target_index,
                min(nb_intermediate_pts - 1, target_index + 1),
            )
            target_index = min(nb_intermediate_pts - 1, target_index + 1)
            target = intermediate_pts[target_index]

            next_target_index = min(nb_intermediate_pts - 1, target_index + 1)
            min_target_dist = 2.0 * global_target_dist
            continue

        min_target_dist = min(min_target_dist, target_dist)

        current_pt = current_pt + direction * step_length
        total_length += step_length
        new_pts.append(current_pt)
        latest_lengths.append(step_length)
        latest_directions.append(direction)
        while sum(latest_lengths) >= history_path_length:
            latest_lengths.pop(0)
            latest_directions.pop(0)

    new_pts.append(intermediate_pts[-1])

    return np.array(new_pts, dtype=dtype), (latest_lengths, latest_directions)


def plot(morph, initial_morph, figure_path):
    """Plot the morphology after post-processing."""
    morph_name = figure_path.stem

    steiner_builder = NeuronBuilder(
        initial_morph,
        "3d",
        line_width=4,
        title=f"{morph_name}",
    )
    fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=f"{morph_name}")

    fig = make_subplots(
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Post-processed morphology", "Raw Steiner morphology"],
    )
    current_data = fig_builder.get_figure()["data"]
    steiner_data = steiner_builder.get_figure()["data"]
    all_data = list(chain(current_data, steiner_data))
    fig.add_traces(
        all_data,
        rows=[1] * (len(current_data) + len(steiner_data)),
        cols=[1] * len(current_data) + [2] * len(steiner_data),
    )

    fig.update_scenes({"aspectmode": "data"})

    fig.layout.update(title=morph_name)

    # Export figure
    fig.write_html(figure_path)

    # Update the HTML file to synchronize the cameras between the two plots
    add_camera_sync(figure_path)


def gather_sections(root_section, tuft_barcodes):
    """Gather the sections with unifurcations."""
    sections_to_smooth = [[]]
    sec_use_parent = {
        tuple(i) for i in tuft_barcodes[["section_id", "use_parent"]].to_numpy().tolist()
    }
    for section in root_section.iter():
        sections_to_smooth[-1].append(section)
        if (
            len(section.children) != 1
            or (section.id, False) in sec_use_parent
            or any((child.id, True) in sec_use_parent for child in section.children)
        ):
            sections_to_smooth.append([])

    return sections_to_smooth


def resample_diameters(pts, resampled_pts, diams):
    """Resample the diameters on the new points."""
    path_lengths = np.insert(
        np.cumsum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)),
        0,
        0,
    )
    new_path_lengths = np.insert(
        np.cumsum(np.linalg.norm(resampled_pts[1:] - resampled_pts[:-1], axis=1)),
        0,
        0,
    )
    return np.interp(
        new_path_lengths / new_path_lengths[-1],
        path_lengths / path_lengths[-1],
        diams,
    )


def post_process_trunk(
    morph: Morphology,
    trunk_section_id: int,
    trunk_properties: pd.DataFrame,
    tuft_barcodes: pd.DataFrame,
    *,
    rng: SeedType = None,
    output_path: FileType | None = None,
    figure_path: FileType | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Post-process a trunk of the given morphology."""
    logger = sublogger(logger, __name__)

    rng = np.random.default_rng(rng)

    initial_morph = Morphology(morph) if figure_path is not None else None

    root_section = morph.section(trunk_section_id)

    # Get some statistics
    # TODO: Pick properties in a less random way? Maybe we could use the source region ID?
    ref_trunk_props = trunk_properties.sample(random_state=rng).iloc[0]

    if logger.getEffectiveLevel() <= logging.DEBUG:
        logger.debug(
            "Ref statistics of the trunk: %s",
            ref_trunk_props.drop(
                ["morph_file", "axon_id"]
                + [row for row in ref_trunk_props.index if row.startswith("raw_")],
            ).to_dict(),
        )
        trunk_stats = morph_stats.extract_stats(
            Neurite(root_section),
            {
                "neurite": {
                    "segment_lengths": {"modes": ["mean", "std"]},
                    "segment_meander_angles": {"modes": ["mean", "std"]},
                },
            },
        )["axon"]
        logger.debug("Current trunk statistics: %s", trunk_stats)

    # Gather sections with unifurcations into future sections
    sections_to_smooth = gather_sections(root_section, tuft_barcodes)

    # Smooth the sections but do not move the tuft roots
    parent_histories = {}
    for i in sections_to_smooth:
        if not i:
            continue
        pts = np.concatenate([i[0].points] + [sec.points[1:] for sec in i[1:]])
        diams = np.concatenate([i[0].diameters] + [sec.diameters[1:] for sec in i[1:]])

        length_stats = {
            "norm": ref_trunk_props["mean_segment_lengths"],
            "std": ref_trunk_props["std_segment_lengths"],
        }
        # angle_stats = {
        #     "norm": ref_trunk_props["mean_segment_meander_angles"],
        #     "std": ref_trunk_props["std_segment_meander_angles"],
        # }
        if not i[0].is_root:
            try:
                parent_history = parent_histories[i[0].parent]
            except KeyError:
                parent_history = None
        else:
            parent_history = None

        resampled_pts, last_history = random_walk(
            pts[0],
            pts[1:],
            length_stats,
            # angle_stats,
            parent_history,
            rng=rng,
            logger=logger,
        )
        parent_histories[i[-1]] = last_history

        resampled_diams = resample_diameters(pts, resampled_pts, diams)

        # Update section points and diameters
        sec_pts = np.array_split(resampled_pts, len(i))
        sec_diams = np.array_split(resampled_diams, len(i))
        for num, sec in enumerate(i):
            if num == 0:
                s_pts = sec_pts[num]
                s_diams = sec_diams[num]
            else:
                s_pts = np.concatenate([[sec_pts[num - 1][-1]], sec_pts[num]])
                s_diams = np.concatenate([[sec_diams[num - 1][-1]], sec_diams[num]])
            sec.points = s_pts
            sec.diameters = s_diams

    if logger.getEffectiveLevel() <= logging.DEBUG:
        trunk_stats = morph_stats.extract_stats(
            Neurite(root_section),
            {
                "neurite": {
                    "segment_lengths": {"modes": ["mean", "std"]},
                    "segment_meander_angles": {"modes": ["mean", "std"]},
                },
            },
        )["axon"]
        logger.debug("New trunk statistics: %s", trunk_stats)

    # Export the new morphology
    if output_path is not None:
        morph.write(output_path)
        logger.info("Exported morphology to %s", output_path)

    # Create a figure of the new morphology
    if figure_path is not None:
        plot(morph, initial_morph, figure_path)
        logger.info("Exported figure to %s", figure_path)
