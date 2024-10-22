"""Post-process the Steiner solutions."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import logging
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import define
from attrs import field
from attrs import validators
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
from axon_synthesis.utils import build_layout_properties
from axon_synthesis.utils import save_morphology
from axon_synthesis.utils import sublogger

if TYPE_CHECKING:
    from morphio.mut import Section

WEIGHT_DISTANCE_TOLERANCE = 1e-8

HistoryType = tuple[list[float], (list[list[float] | npt.NDArray[np.floating]])]


@define
class PostProcessConfig:
    """Class to store the parameters needed for long-range trunk post-processing.

    Attributes:
        skip: Skip the post-processing step if set to True.
        history_path_length: The length used to compute the random walk history.
        default_history_path_length_coeff: The coefficient used to compute the history path length
            when it is not provided.
        global_target_coeff: The coefficient applied to the global target term.
        target_coeff: The coefficient applied to the next target term.
        random_coeff: The coefficient applied to the random term.
        history_coeff: The coefficient applied to the history term.
        length_coeff: The coefficient applied to step length.
        max_random_direction_picks: The maximum number of random direction picks (random directions
            are picked again when they are not facing the target)
    """

    skip: bool = False
    history_path_length: float | None = field(
        default=None, validator=validators.optional(validators.ge(0))
    )
    default_history_path_length_coeff: float = field(default=5, validator=validators.gt(0))
    global_target_coeff: float = field(default=0, validator=validators.ge(0))
    global_target_sigma_coeff: float = field(default=10, validator=validators.gt(0))
    target_coeff: float = field(default=2, validator=validators.ge(0))
    target_sigma_coeff: float = field(default=2, validator=validators.gt(0))
    random_coeff: float = field(default=2, validator=validators.ge(0))
    history_coeff: float = field(default=2, validator=validators.ge(0))
    history_sigma_coeff: float = field(default=2, validator=validators.gt(0))
    length_coeff: float = field(default=1, validator=validators.gt(0))
    max_random_direction_picks: int = field(default=10, validator=validators.ge(1))


def get_random_vector(
    distance: float = 1.0,
    norm: float | None = None,
    std: float | None = None,
    initial_theta: float | None = None,
    initial_phi: float | None = None,
    rng=np.random,
) -> npt.NDArray[np.floating]:
    """Return 3-d coordinates of a new random point.

    The distance between the produced point and (0,0,0) is given by the 'distance' argument.
    """
    # pylint: disable=assignment-from-no-return
    phi = rng.uniform(0.0, 2.0 * np.pi)
    theta = (
        rng.normal(norm, std)
        if norm is not None and std is not None
        else np.arccos(rng.uniform(-1.0, 1.0))
    )

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
    if len(latest_directions) == 0:
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


def compute_direction(start_pt, end_pt):
    """Compute the direction and distance between 2 points."""
    direction = end_pt - start_pt
    distance = np.linalg.norm(direction)
    direction /= distance
    return direction, distance


def compute_step_direction(  # noqa: PLR0913
    intermediate_pts,
    current_pt,
    target,
    target_index,
    next_target,
    next_target_index,
    step_length,
    total_length,
    config,
    history_direction,
    latest_directions,
    rng=np.random,
    logger=None,
):
    """Compute the direction for one step."""
    # Compute the direction to the last target
    global_target_direction, global_target_dist = compute_direction(
        current_pt, intermediate_pts[-1]
    )

    # TODO: Should check that there is no other possible target between the last target and
    # this target that is further to this target.
    target_vec = target - current_pt
    target_dist = np.linalg.norm(target_vec)

    next_target_vec = next_target - current_pt

    current_target_coeff = np.exp(-target_dist / step_length)

    target_direction = (
        1 - current_target_coeff
    ) * target_vec + current_target_coeff * next_target_vec

    target_direction /= np.linalg.norm(target_direction)

    step_global_target_coeff = config.global_target_coeff * max(
        0,
        1
        + np.exp(
            -global_target_dist / (config.global_target_sigma_coeff * step_length)
        ),  # More when closer
    )
    step_target_coeff = config.target_coeff * max(
        0,
        1
        + np.exp(
            -target_dist / (config.target_sigma_coeff * step_length)
        )  # More near targets to pass closer
        - np.exp(
            -total_length / (config.target_sigma_coeff * step_length)
        ),  # Less at the beginning
    )
    step_history_coeff = config.history_coeff * max(
        0,
        1
        - np.exp(
            -target_dist / (config.history_sigma_coeff * step_length)
        )  # Less near targets to pass closer
        + np.exp(
            -total_length / (config.history_sigma_coeff * step_length)
        ),  # More at the beginning
    )
    step_random_coeff = config.random_coeff

    direction = -target_direction
    nb_rand = 0
    non_random_direction = (
        step_global_target_coeff * global_target_direction
        + step_target_coeff * target_direction
        + step_history_coeff * history_direction
    ).astype(float)

    # If the non random part of the direction does not head to the target direction
    # (e.g. because of the history), then we don't care if the resulting direction
    # does not head to the target direction
    heading_target = np.dot(target_direction, non_random_direction) >= 0

    while np.dot(direction, target_direction) < 0 and nb_rand < config.max_random_direction_picks:
        if nb_rand > 0:
            step_random_coeff = step_random_coeff / 2.0
        random_direction = get_random_vector(rng=rng)

        direction = (non_random_direction + step_random_coeff * random_direction).astype(float)

        if nb_rand == 0 and not heading_target:
            break

        nb_rand += 1

    direction /= np.linalg.norm(direction)

    if logger is not None:
        initial_phi, initial_theta = rotation.spherical_from_vector(latest_directions[-1])
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
                "next_target=%s\n\t"
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
            next_target,
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

    return direction, target_dist, global_target_dist


def check_next_target(  # noqa: PLR0913
    intermediate_pts,
    nb_intermediate_pts,
    target_index,
    target,
    target_dist,
    next_target_index,
    next_target,
    min_target_dist,
    length_norm,
    default_min_target_dist,
    logger,
):
    """Check if a next target can be selected and return it."""
    if target_dist >= min_target_dist:
        logger.debug("The random walk is going away from the target")
        new_target = True
    elif target_dist <= length_norm:
        logger.debug("The random walk reached the target %s", target_index)
        new_target = True
    else:
        new_target = False

    if new_target:
        logger.debug(
            "Changing target from %s to %s",
            target_index,
            next_target_index,
        )
        target_index = min(nb_intermediate_pts - 1, target_index + 1)
        target = intermediate_pts[target_index]
        next_target_index = min(nb_intermediate_pts - 1, target_index + 1)
        next_target = intermediate_pts[next_target_index]
        min_target_dist = default_min_target_dist

    return new_target, target_index, target, next_target_index, next_target, min_target_dist


def random_walk(
    starting_pt: Sequence[float] | npt.NDArray[np.floating],
    intermediate_pts: Sequence[Sequence[float]] | npt.NDArray[np.floating],
    length_stats: dict[str, float],
    # angle_stats,
    config: PostProcessConfig,
    previous_history: HistoryType | None = None,
    *,
    rng=np.random,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> tuple[npt.NDArray[np.floating], HistoryType]:
    """Perform a random walk guided by intermediate points."""
    logger = sublogger(logger, __name__)

    debug = logger.getEffectiveLevel() <= logging.DEBUG

    length_norm = length_stats["norm"] * config.length_coeff
    length_std = length_stats["std"] * config.length_coeff
    # angle_norm = angle_stats["norm"]
    # angle_std = angle_stats["std"]

    history_path_length = config.history_path_length
    if history_path_length is None:
        history_path_length = config.default_history_path_length_coeff * length_norm

    current_pt = np.array(starting_pt, dtype=float)
    new_intermediate_pts = np.array(intermediate_pts, dtype=float)
    new_pts: list[npt.NDArray[np.floating]] = [current_pt]

    total_length = 0

    # Compute the direction to the last target
    global_target_direction, global_target_dist = compute_direction(
        current_pt, new_intermediate_pts[-1]
    )

    # Compute the direction to the first target
    target_direction, target_dist = compute_direction(current_pt, new_intermediate_pts[0])

    # Setup initial history
    if previous_history:
        latest_lengths, latest_directions = previous_history
    else:
        nb_hist = int(history_path_length // length_norm)
        latest_lengths = [length_norm] * nb_hist
        latest_directions = [target_direction] * nb_hist

    nb_intermediate_pts = len(new_intermediate_pts)

    min_target_dist = global_target_dist * 2

    if debug:
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
            new_intermediate_pts,
        )

    target_index = 0
    target = new_intermediate_pts[target_index]
    next_target_index = min(nb_intermediate_pts - 1, 1)
    next_target = new_intermediate_pts[next_target_index]

    while global_target_dist >= length_norm:
        step_length = rng.normal(length_norm, length_std)
        while step_length <= 0:
            step_length = rng.normal(length_norm, length_std)

        history_direction = history(latest_lengths, latest_directions, history_path_length)

        direction, target_dist, global_target_dist = compute_step_direction(
            new_intermediate_pts,
            current_pt,
            target,
            target_index,
            next_target,
            next_target_index,
            step_length,
            total_length,
            config,
            history_direction,
            latest_directions,
            rng=rng,
            # logger=logger if debug else None,
        )

        (
            new_target,
            target_index,
            target,
            next_target_index,
            next_target,
            min_target_dist,
        ) = check_next_target(
            new_intermediate_pts,
            nb_intermediate_pts,
            target_index,
            target,
            target_dist,
            next_target_index,
            next_target,
            min_target_dist,
            length_norm,
            global_target_dist * 2,
            logger=logger,
        )

        if new_target:
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

    new_pts.append(new_intermediate_pts[-1])

    return np.array(new_pts, dtype=float), (latest_lengths, latest_directions)


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

    layout_props = build_layout_properties(morph.points, 0.1)

    fig.update_scenes(layout_props)
    fig.update_layout(title=morph_name)

    # Export figure
    Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(figure_path)

    # Update the HTML file to synchronize the cameras between the two plots
    add_camera_sync(figure_path)


def gather_sections(root_section, tuft_barcodes):
    """Gather the sections with unifurcations."""
    sections_to_smooth: list[list[Section]] = [[]]
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

    return [i for i in sections_to_smooth if i]


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


def export(morph, initial_morph, output_path, figure_path, logger):
    """Export morphology and figure."""
    # Export the new morphology
    if output_path is not None:
        save_morphology(morph, output_path, msg=f"Export morphology to {output_path}")

    # Create a figure of the new morphology
    if figure_path is not None:
        logger.info("Export figure to %s", figure_path)
        plot(morph, initial_morph, figure_path)


def post_process_trunk(
    morph: Morphology,
    trunk_section_id: int,
    trunk_properties: pd.DataFrame,
    tuft_barcodes: pd.DataFrame,
    config: PostProcessConfig,
    *,
    rng: SeedType = None,
    output_path: FileType | None = None,
    figure_path: FileType | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Post-process a trunk of the given morphology."""
    logger = sublogger(logger, __name__)
    debug = logger.getEffectiveLevel() <= logging.DEBUG

    if config.skip:
        logger.info("Skip post-processing")

    rng = np.random.default_rng(rng)

    logger.debug("Use the following post-processing config: %s", config)

    initial_morph = Morphology(morph) if figure_path is not None else None

    root_section = morph.section(trunk_section_id)

    # Get some statistics
    # TODO: Pick properties in a less random way? Maybe we could use the source region ID?
    ref_trunk_props = trunk_properties.sample(random_state=rng).iloc[0]

    if debug:
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

    length_stats = {
        "norm": ref_trunk_props["mean_segment_lengths"],
        "std": ref_trunk_props["std_segment_lengths"],
    }
    # angle_stats = {
    #     "norm": ref_trunk_props["mean_segment_meander_angles"],
    #     "std": ref_trunk_props["std_segment_meander_angles"],
    # }

    # Smooth the sections but do not move the tuft roots
    parent_histories: dict[int, HistoryType] = {}
    for i in sections_to_smooth:
        pts = np.concatenate([i[0].points] + [sec.points[1:] for sec in i[1:]])
        diams = np.concatenate([i[0].diameters] + [sec.diameters[1:] for sec in i[1:]])
        min_diam_not_zero = np.min(diams, initial=1, where=diams != 0)
        diams = np.clip(diams, a_min=min_diam_not_zero, a_max=np.inf)

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
            config,
            parent_history,
            rng=rng,
            logger=logger,
        )
        parent_histories[i[-1]] = last_history

        if len(resampled_pts) <= len(pts):
            resampled_pts = pts

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

    if debug:
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

    export(morph, initial_morph, output_path, figure_path, logger)
