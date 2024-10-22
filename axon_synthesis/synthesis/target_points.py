"""Find the target points of the input morphologies."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from h5py import File
from numpy.random import Generator
from scipy.spatial import KDTree

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import DEFAULT_POPULATION
from axon_synthesis.constants import SOURCE_COORDS_COLS
from axon_synthesis.constants import TARGET_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import ignore_warnings

LOGGER = logging.getLogger(__name__)


def compute_coords(
    target_points: pd.DataFrame,
    brain_regions_masks: File | None,
    rng: Generator,
    *,
    atlas: AtlasHelper | None = None,
) -> None:
    """Compute the target coordinates if they are missing."""
    if set(TARGET_COORDS_COLS).difference(target_points.columns):
        if brain_regions_masks is not None:
            mask_tmp = (
                target_points.loc[~target_points["target_brain_region_id"].isna()]
                .sort_values("target_brain_region_id")
                .index
            )
            target_points.loc[:, TARGET_COORDS_COLS] = np.nan

            def get_coords(group: pd.DataFrame) -> np.ndarray:
                try:
                    return rng.choice(  # type: ignore[arg-type, return-value]
                        brain_regions_masks[str(int(group.name))][:], size=len(group)
                    )
                except Exception as e:
                    logging.warning("Error %s for target_brain_region_id %s", repr(e), group.name)
                    return np.repeat(np.nan, len(group))

            target_points.loc[mask_tmp, TARGET_COORDS_COLS] = (
                target_points.groupby("target_brain_region_id")
                .apply(get_coords)  # type: ignore[arg-type]
                .explode()
                .sort_index()
                .apply(pd.Series)
                .to_numpy()
            )
        else:
            msg = (
                f"The target points should contain the {TARGET_COORDS_COLS} columns when no brain "
                "region mask is given"
            )
            raise RuntimeError(msg)
        if atlas is not None:
            # Convert indices into coordinates
            target_points.loc[:, TARGET_COORDS_COLS] = atlas.brain_regions.indices_to_positions(
                target_points[TARGET_COORDS_COLS].to_numpy()  # noqa: RUF005
                + [0.5, 0.5, 0.5]
            ) + atlas.get_random_voxel_shifts(len(target_points), rng=rng)
            # place the targets in the correct hemisphere, if hemisphere is given
            if "target_hemisphere" in target_points.columns:
                target_points[TARGET_COORDS_COLS] = target_points.apply(  # type: ignore[call-overload]
                    lambda row: atlas.place_point_in_hemisphere(
                        row[TARGET_COORDS_COLS], row["target_hemisphere"]
                    ),
                    axis=1,
                )[TARGET_COORDS_COLS]


def select_close_points_in_group(
    all_points_df: pd.DataFrame, duplicate_precision: float | None
) -> dict:
    """Select points that are closer to a given distance."""
    if len(all_points_df) <= 1:
        return {}

    tree = KDTree(all_points_df[TARGET_COORDS_COLS])
    close_pts = tree.query_pairs(duplicate_precision)

    if not close_pts:
        return {}

    # Find labels of duplicated points
    to_update: dict[Any, Any] = {}
    for a, b in close_pts:
        label_a = all_points_df.index[a]
        label_b = all_points_df.index[b]
        if label_a in to_update:
            to_update[label_a].add(label_b)
        elif label_b in to_update:
            to_update[label_b].add(label_a)
        else:
            to_update[label_a] = {label_b}

    return to_update


def drop_close_points(
    all_points_df: pd.DataFrame, duplicate_precision: float | None
) -> pd.DataFrame:
    """Drop points that are closer to a given distance."""
    if duplicate_precision is None or len(all_points_df) <= 1:
        return all_points_df

    all_groups = all_points_df.groupby(["morphology", "axon_id"], group_keys=True)
    deduplicated = all_groups[TARGET_COORDS_COLS].apply(
        lambda group: select_close_points_in_group(group, duplicate_precision)
    )

    # Format the labels
    def format_labels(to_update) -> list:
        skip = set()
        items = list(to_update.items())
        for num, (i, j) in enumerate(items):
            if i in skip:
                continue
            for ii, jj in items[num + 1 :]:
                if i in jj or ii in j:
                    j.update(jj)
                    skip.add(ii)
                    skip.update(jj)
        return [i for i in items if i[0] not in skip]

    to_update = list(itertools.chain.from_iterable(deduplicated.apply(format_labels)))

    # Update the terminal IDs
    for ref, changed in to_update:
        all_points_df.loc[list(changed), "terminal_id"] = all_points_df.loc[ref, "terminal_id"]

    if "level_2" in all_points_df.columns:
        all_points_df = all_points_df.drop(columns=["level_2"])

    return all_points_df


def pick_target_populations(probs, rng, logger=None, max_tries=10):
    """Pick target populations for each source point."""
    probs["random_number"] = pd.Series(-1, index=probs.index.to_numpy(), dtype=float)
    no_target_mask = probs["random_number"] < 0
    n_tries = 0
    mask_size = no_target_mask.sum()
    selected_mask = pd.Series(data=True, index=probs.index, dtype=bool)
    while n_tries < max_tries and mask_size > 0:
        # Select the populations according to the associated probabilities
        probs.loc[no_target_mask, "random_number"] = rng.uniform(size=mask_size)
        selected_mask = probs["random_number"] <= probs["target_probability"]

        # TODO: Here we implicitly suppose that we want to select at least 1 target per axon, but
        # maybe we want a customizable minimum number of targets?

        # Check which axons don't have any selected target
        no_target_mask = probs.merge(
            probs.loc[selected_mask, ["morphology", "axon_id", "random_number"]].drop_duplicates(
                subset=["morphology", "axon_id"]
            ),
            on=["morphology", "axon_id"],
            how="left",
            suffixes=("", "_tmp"),
        )["random_number_tmp"].isna()

        mask_size = no_target_mask.sum()
        n_tries = n_tries + 1

    if mask_size > 0 and logger is not None:
        logger.warning(
            "Could not find any target for the following morphologies: %s",
            ", ".join(
                [
                    f"{i[0]} (axon ID={i[1]})"
                    for i in probs.loc[no_target_mask, ["morphology", "axon_id"]]
                    .drop_duplicates()
                    .to_numpy()
                    .tolist()
                ]
            ),
        )

    return selected_mask


def get_target_points(
    source_points,
    target_probabilities,
    tufts_dist_df: pd.DataFrame | None = None,
    duplicate_precision: float | None = None,
    *,
    atlas: AtlasHelper | None = None,
    brain_regions_masks: File | None = None,
    rng: SeedType | None = None,
    max_tries: int = 10,
    output_path: FileType | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Find the target points for all given source points."""
    rng = np.random.default_rng(rng)
    if logger is None:
        logger = LOGGER

    # Create default populations if missing
    if "population_id" not in source_points.columns:
        source_points["population_id"] = DEFAULT_POPULATION
    if "source_population_id" not in target_probabilities.columns:
        target_probabilities["source_population_id"] = DEFAULT_POPULATION

    # Duplicated entries stand for different axons so we create axon IDs
    source_points["axon_id"] = source_points.groupby("morphology").cumcount()

    # Get ascendants in the hierarchy
    if atlas is not None and "st_level" not in source_points.columns:
        cells_region_parents = source_points.merge(
            atlas.brain_regions_and_ascendants,
            left_on="source_brain_region_id",
            right_on="id",
            how="left",
        ).drop(columns=["id"])
    else:
        cells_region_parents = source_points.copy(deep=False)
        cells_region_parents["st_level"] = 0

    # Remove useless columns before merge to reduce RAM usage
    cells_region_parents = cells_region_parents[
        ["morphology", "axon_id", "source_brain_region_id", "population_id", "st_level"]
    ]
    target_probabilities = target_probabilities[
        ["source_population_id", "target_brain_region_id", "target_population_id", "probability"]
        + (
            TARGET_COORDS_COLS
            if not set(TARGET_COORDS_COLS).difference(target_probabilities.columns)
            else []
        )
        + (["target_hemisphere"] if "target_hemisphere" in target_probabilities.columns else [])
    ]
    # Get the probabilities
    probs = cells_region_parents.merge(
        target_probabilities.rename(columns={"probability": "target_probability"}),
        left_on=["population_id"],
        right_on=["source_population_id"],
        how="left",
    )
    # Report missing probabilities
    missing_probs = probs.loc[probs["target_probability"].isna()]
    if len(missing_probs) > 0:
        logger.warning(
            "The following morphologies have no associated target probabilities: %s",
            missing_probs["morphology"].drop_duplicates().to_list(),
        )

    # Keep only the probabilities from the deepest level in the hierarchy
    probs = probs.dropna(axis=0, subset=["target_probability"])
    probs = probs.loc[
        probs["st_level"]
        == probs.groupby(["morphology", "source_brain_region_id"])["st_level"].transform("max")
    ].reset_index(drop=True)

    # Ensure that at least one region is selected for each morphology
    selected_mask = pick_target_populations(probs, rng, logger, max_tries)

    def draw_tuft_number(row) -> int:
        try:
            return max(
                1,
                int(
                    rng.normal(
                        tufts_dist_df.loc[row["target_population_id"], "mean_tuft_number"],  # type: ignore[union-attr]
                        tufts_dist_df.loc[row["target_population_id"], "std_tuft_number"],  # type: ignore[union-attr]
                    )
                ),
            )
        except (KeyError, AttributeError):
            return 1

    # Set the number of tufts to grow for each target population
    probs["num_tufts_to_grow"] = 0
    probs.loc[selected_mask, "num_tufts_to_grow"] = probs.apply(draw_tuft_number, axis=1)

    probs_cols = [
        "morphology",
        "axon_id",
        "source_brain_region_id",
        "target_population_id",
        "target_brain_region_id",
        "num_tufts_to_grow",
    ]
    if "target_hemisphere" in probs.columns:
        probs_cols.append("target_hemisphere")
    if not set(TARGET_COORDS_COLS).difference(probs.columns):
        probs_cols.extend(TARGET_COORDS_COLS)
    target_points = source_points.merge(
        probs.loc[
            selected_mask,
            probs_cols,
        ],
        on=["morphology", "axon_id", "source_brain_region_id"],
        how="left",
    ).dropna(subset=["target_population_id"])

    # Duplicate row according to the number of tuft in each population
    repeated_index = np.repeat(target_points.index, target_points["num_tufts_to_grow"].astype(int))

    # Reindex the DataFrame with the new repeated index
    target_points = target_points.loc[repeated_index].reset_index(drop=True)

    compute_coords(target_points, brain_regions_masks, atlas=atlas, rng=rng)
    # Build terminal IDs inside groups
    counter = target_points[["morphology", "axon_id"]].copy(deep=False)
    counter["counter"] = 1
    target_points["terminal_id"] = counter.groupby(["morphology", "axon_id"])["counter"].cumsum()

    other_columns = []
    if "seed" in target_points.columns:
        other_columns.append("seed")

    # Remove useless columns
    target_points = target_points[
        [
            "morphology",
            "morph_file",
            "axon_id",
            "terminal_id",
            *COORDS_COLS,
            "orientation",
            "grafting_section_id",
            "population_id",
            "source_brain_region_id",
            *SOURCE_COORDS_COLS,
            "target_population_id",
            "target_brain_region_id",
            *TARGET_COORDS_COLS,
            *other_columns,
        ]
    ].rename(
        columns={
            "population_id": "source_population_id",
        },
    )

    target_points = drop_close_points(target_points, duplicate_precision)

    # Export the target points
    if output_path is not None:
        with ignore_warnings(pd.errors.PerformanceWarning):
            target_points.to_hdf(output_path, key="target_points")
        logger.debug("Target points exported to %s", output_path)

    logger.info("Found %s target point(s)", len(target_points))

    return target_points.sort_values(["morphology", "axon_id", "terminal_id"]).reset_index(
        drop=True
    )
