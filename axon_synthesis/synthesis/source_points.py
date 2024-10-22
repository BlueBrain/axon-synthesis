"""Create the source points from the atlas."""

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
from pathlib import Path

import numpy as np
import pandas as pd
from morphio import Morphology
from morphio import RawDataError
from morphio import SectionType
from neurom import COLS

from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.constants import COORDS_COLS
from axon_synthesis.constants import DEFAULT_POPULATION
from axon_synthesis.constants import SOURCE_COORDS_COLS
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import load_morphology

_MAX_DISPLAYED = 100


def section_id_to_position(morph, sec_id):
    """Find the position of the last point of the section from its ID."""
    morph = Morphology(morph)
    try:
        return morph.section(sec_id).points[-1, COLS.XYZ] - morph.soma.center
    except RawDataError:
        return None


def find_existing_axons(morph_file):
    """Find the positions of the existing axons in a morphology."""
    morph = load_morphology(morph_file, recenter=True)
    return [sec.points[0] for sec in morph.root_sections if sec.type == SectionType.axon]


def map_population(
    cells_df: pd.DataFrame,
    atlas: AtlasHelper | None,
    populations: pd.DataFrame | None = None,
    *,
    rng: SeedType = None,
):
    """Find the population given the position of the morphology and the populations."""
    if "population_id" in cells_df.columns:
        return cells_df

    rng = np.random.default_rng(rng)
    if populations is None:
        cells_df["population_id"] = DEFAULT_POPULATION
    elif atlas is None or populations is None:
        msg = (
            "The 'populations' and 'atlas' arguments should not be None when the 'population_id' "
            "column is not given"
        )
        raise RuntimeError(msg)
    else:
        # Get all the parent IDs in the brain region hierarchy
        cells_region_parents = cells_df.merge(
            atlas.brain_regions_and_ascendants,
            left_on="source_brain_region_id",
            right_on="id",
            how="left",
        )

        # Get the probabilities
        probs = cells_region_parents.merge(
            populations.rename(columns={"probability": "population_probability"}),
            left_on="elder_id",
            right_on="brain_region_id",
            how="left",
        )
        probs = probs.dropna(axis=0, subset=["population_id"])

        # Keep only the probabilities from the deepest level in the hierarchy
        probs = probs.loc[
            probs["st_level"]
            == probs.groupby(["morphology", "source_brain_region_id"])["st_level"].transform("max")
        ]

        # if probs has column 'hemisphere', select probabilities also on hemispheres
        if "hemisphere" in probs.columns:
            # rename 'hemisphere' to 'population_hemisphere' to not mistake with the cell
            probs = probs.rename(columns={"hemisphere": "population_hemisphere"})
            # add a column 'cell_hemisphere' for each cell
            probs["cell_hemisphere"] = probs[COORDS_COLS].apply(atlas.get_hemisphere, axis=1)
            # and drop the rows where the cell and the population are not in the same hemisphere
            probs = probs.loc[probs["cell_hemisphere"] == probs["population_hemisphere"]]

        # Select the populations according to the associated probabilities
        selected = probs.groupby(["morphology", "source_brain_region_id"]).sample(
            weights=probs["population_probability"],
            random_state=rng,
        )

        cells_df = cells_df.merge(
            selected[["morphology", "source_brain_region_id", "population_id"]],
            on=["morphology", "source_brain_region_id"],
            how="left",
        ).fillna({"population_id": DEFAULT_POPULATION})

    return cells_df


def fill_morph_file_col(cells_df, morph_dir):
    """Find the files corresponding to the given morphology names.

    .. warning::

        The validity of the morphology files is not tested but only their existence.
    """
    cells_df.loc[:, "morph_file"] = None
    for ext in [".asc", ".h5", ".swc"]:
        new_paths = (Path(morph_dir) / cells_df["morphology"]).apply(
            lambda x, ext=ext: x.with_suffix(ext).resolve()
        )
        existing_paths_mask = new_paths.apply(lambda x: x.exists())
        cells_df.loc[existing_paths_mask, "morph_file"] = new_paths.loc[existing_paths_mask]
    if cells_df["morph_file"].isna().any():
        msg = (
            f"Could not find morphology files in the '{morph_dir}' directory for the following "
            "morphologies: ["
        )
        missing_files = cells_df.loc[cells_df["morph_file"].isna(), "morphology"]
        nb_tot = len(missing_files)
        if nb_tot > _MAX_DISPLAYED:
            missing_files = missing_files.head(_MAX_DISPLAYED)
            suffix = f", ... ({nb_tot} morphologies in total but only {_MAX_DISPLAYED} displayed)]"
        else:
            suffix = "]"
        msg += ", ".join(("'" + missing_files + "'").to_list()) + suffix
        raise RuntimeError(msg)


def combine_existing_axons(cells_df):
    """Combine rebuilt existing axons with explicitly given axons."""
    existing_axons = (
        cells_df.groupby("morph_file")["morph_file"]
        .apply(lambda group: find_existing_axons(group.name))
        .apply(pd.Series)
        .stack()
        .rename("XYZ")  # type: ignore[call-overload]
    )
    existing_axons.index.rename("axon_id", level=1, inplace=True)  # type: ignore[call-arg]
    existing_axons = existing_axons.reset_index()
    existing_axons["grafting_section_id"] = -1
    if existing_axons.empty:
        existing_axons[SOURCE_COORDS_COLS] = pd.DataFrame(
            {col: pd.Series(dtype=float) for col in SOURCE_COORDS_COLS}
        )
    else:
        existing_axons[SOURCE_COORDS_COLS] = np.stack(
            existing_axons["XYZ"].to_numpy()  # type: ignore[arg-type]
        )
    new_axons = (
        cells_df[
            [
                col
                for col in cells_df.columns
                if col not in ["grafting_section_id", *SOURCE_COORDS_COLS]
            ]
        ]
        .drop_duplicates("morphology")
        .merge(
            existing_axons[["morph_file", "grafting_section_id", *SOURCE_COORDS_COLS]],
            on="morph_file",
            how="right",
        )
    )

    if "rebuilt_existing_axon_id" in cells_df.columns:
        if cells_df[["morphology", "rebuilt_existing_axon_id"]].duplicated().any():
            msg = "The 'rebuilt_existing_axon_id' should be unique for each morphology"
            raise ValueError(msg)
        new_axons["rebuilt_existing_axon_id"] = new_axons.groupby("morphology").cumcount()
        all_axons = (
            new_axons.reset_index()
            .merge(
                cells_df[["morphology", "rebuilt_existing_axon_id"]]
                .drop_duplicates()
                .reset_index(),
                on=["morphology", "rebuilt_existing_axon_id"],
                how="outer",
                indicator=True,
                suffixes=("_new_axons", "_cells_df"),
            )
            .dropna(subset=["morph_file"])
        )
        new_axons = new_axons.loc[
            all_axons.loc[all_axons["_merge"] == "left_only", "index_new_axons"]
            .astype(int)
            .to_numpy()
        ]
        cells_df = cells_df.loc[
            all_axons.loc[all_axons["_merge"] == "both", "index_cells_df"].astype(int).to_numpy()
        ]

    # We don't add axons starting from the soma when an existing axon is rebuilt
    cells_df = pd.concat([cells_df.loc[cells_df.index.difference(new_axons.index)], new_axons])
    cells_df.fillna(
        {source_col: 0 for source_col in SOURCE_COORDS_COLS},
        inplace=True,
    )
    return cells_df


def set_source_points(
    cells_df: pd.DataFrame,
    atlas: AtlasHelper | None,
    morph_dir: FileType,
    population_probabilities: pd.DataFrame | None = None,
    axon_grafting_points: pd.DataFrame | None = None,
    *,
    rng: SeedType = None,
    rebuild_existing_axons: bool = False,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Extract source points from a cell collection."""
    if "morph_file" not in cells_df.columns:
        fill_morph_file_col(cells_df, morph_dir)

    cells_df = cells_df.reset_index(drop=True)

    # Get source points from the axon_grafting_points file
    if axon_grafting_points is not None:
        axon_grafting_points = axon_grafting_points[
            [
                col
                for col in axon_grafting_points.columns
                if col
                in [
                    "morphology",
                    "grafting_section_id",
                    *SOURCE_COORDS_COLS,
                    "population_id",
                    "rebuilt_existing_axon_id",
                ]
            ]
        ]
        cells_df = cells_df.merge(axon_grafting_points, on="morphology", how="left")

    # Find existing axons to rebuild them if required
    if rebuild_existing_axons:
        cells_df = combine_existing_axons(cells_df)

    # Format the grafting_section_id column
    if "grafting_section_id" not in cells_df.columns:
        cells_df["grafting_section_id"] = -1
    else:
        cells_df["grafting_section_id"] = cells_df["grafting_section_id"].fillna(-1).astype(int)

    # If some coordinate columns are missing we reset them
    if len(set(SOURCE_COORDS_COLS).difference(cells_df.columns)) > 0:
        cells_df[SOURCE_COORDS_COLS] = np.nan

    # Find where the coordinates should be updated
    missing_coords_mask = cells_df[SOURCE_COORDS_COLS].isna().any(axis=1)
    section_id_mask = (cells_df["grafting_section_id"] != -1) & missing_coords_mask

    # If no section ID is provided we start the axon from the center of the morphology
    if missing_coords_mask.any():
        cells_df.loc[missing_coords_mask, SOURCE_COORDS_COLS] = 0

    # We shift all the coordinates to the positions in the atlas
    cells_df[SOURCE_COORDS_COLS] += cells_df[COORDS_COLS].to_numpy()

    # If a section ID is provided we start the axon from the last point of this section
    # Note: The coordinates of the points of each morphology are relative to the center of this
    # morphology
    if section_id_mask.any():
        cells_df.loc[section_id_mask, SOURCE_COORDS_COLS] += (
            cells_df.loc[section_id_mask]
            .apply(
                lambda row: section_id_to_position(row["morph_file"], row["grafting_section_id"]),
                axis=1,
            )
            .apply(pd.Series)
            .to_numpy()
        )

    # Set atlas regions
    if "source_brain_region_id" not in cells_df.columns:
        if atlas is None:
            msg = "The 'source_brain_region_id' column is missing and no atlas is provided"
            raise ValueError(msg)
        cells_df["source_brain_region_id"] = atlas.brain_regions.lookup(
            cells_df[COORDS_COLS].to_numpy()
        )

    # Choose population
    source_points = map_population(cells_df, atlas, population_probabilities, rng=rng)

    if logger is not None:
        logger.debug("Found %s source point(s)", len(source_points))
    return source_points
