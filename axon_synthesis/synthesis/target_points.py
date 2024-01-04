"""Find the target points of the input morphologies."""
import logging
import warnings

import numpy as np
import pandas as pd

from axon_synthesis.synthesis.source_points import SOURCE_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import COORDS_COLS
from axon_synthesis.utils import DEFAULT_POPULATION
from axon_synthesis.utils import CoordsCols

LOGGER = logging.getLogger(__name__)

TARGET_COORDS_COLS = CoordsCols("target_x", "target_y", "target_z")


def get_target_points(
    atlas,
    brain_regions_masks,
    source_points,
    target_probabilities,
    *,
    rng: SeedType | None = None,
    max_tries: int = 10,
    output_path: FileType | None = None,
):
    """Find the target points for all given source points."""
    rng = np.random.default_rng(rng)

    # Create default populations if missing
    if "population_id" not in source_points.columns:
        source_points["population_id"] = DEFAULT_POPULATION
    if "source_population_id" not in target_probabilities.columns:
        target_probabilities["source_population_id"] = DEFAULT_POPULATION

    # Duplicated entries stand for different axons so we create axon IDs
    source_points["axon_id"] = source_points.groupby("morphology").cumcount()

    # Get ascendants in the hierarchy
    cells_region_parents = source_points.merge(
        atlas.brain_regions_and_ascendants,
        left_on="source_brain_region_id",
        right_on="id",
        how="left",
    )

    # Get the probabilities
    probs = cells_region_parents.merge(
        target_probabilities.rename(columns={"probability": "target_probability"}),
        left_on=["population_id"],
        right_on=["source_population_id"],
        how="left",
        suffixes=("", "_probs"),
    )

    # Report missing probabilities
    missing_probs = probs.loc[probs["target_probability"].isna()]
    if len(missing_probs) > 0:
        LOGGER.warning(
            "The following morphologies have no associated target probabilities: %s",
            missing_probs["morphology"].drop_duplicates().to_list(),
        )

    # Keep only the probabilities from the deepest level in the hierarchy
    probs = probs.dropna(axis=0, subset=["target_probability"])
    probs = probs.loc[
        probs["st_level"]
        == probs.groupby(["morphology", "source_brain_region_id"])["st_level"].transform(max)
    ].reset_index(drop=True)

    # Ensure that at least one region is selected for each morphology
    probs["random_number"] = -1
    no_target_mask = probs["random_number"] < 0
    n_tries = 0
    mask_size = no_target_mask.sum()
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

    if mask_size > 0:
        LOGGER.warning(
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

    target_points = source_points.merge(
        probs.loc[
            selected_mask,
            [
                "morphology",
                "source_brain_region_id",
                "target_population_id",
                "target_brain_region_id",
            ],
        ],
        on=["morphology", "source_brain_region_id"],
        how="left",
    )

    mask_tmp = target_points.sort_values("target_brain_region_id").index
    target_points.loc[mask_tmp, TARGET_COORDS_COLS] = (
        target_points.groupby("target_brain_region_id")
        .apply(lambda group: rng.choice(brain_regions_masks[str(group.name)][:], size=len(group)))
        .explode()
        .sort_index()
        .apply(pd.Series)
        .to_numpy()
    )
    target_points[TARGET_COORDS_COLS] += atlas.brain_regions.indices_to_positions(
        target_points[TARGET_COORDS_COLS].to_numpy()  # noqa: RUF005
        + [0.5, 0.5, 0.5]
    ) + atlas.get_random_voxel_shifts(len(target_points), rng=rng)

    # Build terminal IDs inside groups
    counter = target_points[["morphology", "axon_id"]].copy(deep=False)
    counter["counter"] = 1
    target_points["terminal_id"] = counter.groupby(["morphology", "axon_id"])["counter"].cumsum()

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
        ]
    ].rename(
        columns={
            "population_id": "source_population_id",
        },
    )

    # Export the target points
    if output_path is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)
            target_points.to_hdf(output_path, "target_points")

    return target_points
