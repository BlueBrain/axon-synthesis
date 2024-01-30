"""Create the source points from the atlas."""
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

logger = logging.getLogger(__name__)


def section_id_to_position(morph, sec_id):
    """Find the position of the last point of the section from its ID."""
    morph = Morphology(morph)
    try:
        return morph.section(sec_id).points[-1, COLS.XYZ] - morph.soma.center
    except RawDataError:
        return None


def find_existing_axons(morph):
    """Find the positions of the existing axons in a morphology."""
    morph = Morphology(morph)
    return [sec.points[0] for sec in morph.root_sections if sec.type == SectionType.axon]


def map_population(
    cells_df: pd.DataFrame,
    atlas: AtlasHelper,
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
            == probs.groupby(["morphology", "source_brain_region_id"])["st_level"].transform(max)
        ]

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


def set_source_points(
    cells_df: pd.DataFrame,
    atlas: AtlasHelper,
    morph_dir: FileType,
    ext: str = ".h5",
    population_probabilities: pd.DataFrame | None = None,
    axon_grafting_points: pd.DataFrame | None = None,
    *,
    rng: SeedType = None,
    rebuild_existing_axons: bool = False,
):
    """Extract source points from a cell collection."""
    if not ext.startswith("."):
        ext = "." + ext

    if "morph_file" not in cells_df.columns:
        cells_df["morph_file"] = (Path(morph_dir) / cells_df["morphology"]).apply(
            lambda x: x.with_suffix(ext).resolve()
        )

    # Get source points from the axon_grafting_points file
    if axon_grafting_points is not None:
        axon_grafting_points = axon_grafting_points[
            [
                col
                for col in axon_grafting_points.columns
                if col in ["morphology", "grafting_section_id", *SOURCE_COORDS_COLS]
            ]
        ]
        cells_df = cells_df.merge(axon_grafting_points, on="morphology", how="left")

    # Find existing axons to rebuild them if required
    if rebuild_existing_axons:
        existing_axons = (
            cells_df.groupby("morph_file")["morph_file"]
            .apply(lambda group: find_existing_axons(group.name))
            .apply(pd.Series)
            .stack()
            .rename("XYZ")
        )
        existing_axons.index.rename("axon_id", level=1, inplace=True)
        existing_axons = existing_axons.reset_index()
        existing_axons["grafting_section_id"] = -1
        existing_axons[SOURCE_COORDS_COLS] = np.stack(existing_axons["XYZ"].to_numpy())
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

        # We don't add axons starting from the soma when an existing axon is rebuilt
        cells_df = (
            pd.concat([cells_df.dropna(subset="grafting_section_id"), new_axons])
            .sort_values(["morphology", "grafting_section_id"])
            .reset_index(drop=True)
        )

    # Format the grafting_section_id column
    if "grafting_section_id" not in cells_df.columns:
        cells_df["grafting_section_id"] = -1
    else:
        cells_df["grafting_section_id"] = cells_df["grafting_section_id"].fillna(-1)
    cells_df["grafting_section_id"] = cells_df["grafting_section_id"].astype(int)

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
        cells_df["source_brain_region_id"] = atlas.brain_regions.lookup(
            cells_df[COORDS_COLS].to_numpy()
        )

    # Choose population
    return map_population(cells_df, atlas, population_probabilities, rng=rng)


def create_random_sources(
    atlas,
    source_regions: list[int | str],
    nb_points: int,
    output_path: FileType = None,
    seed: int | None = None,
):
    """Create some random source points."""
    rng = np.random.default_rng(seed)

    if source_regions:
        coords, missing_ids = atlas.get_region_points(
            source_regions, size=nb_points, return_missing=True, rng=rng
        )
        if missing_ids:
            logger.warning("Could not find the following regions in the atlas: %s", missing_ids)
    else:
        coords = atlas.get_region_points(
            [0], size=nb_points, inverse=True, return_missing=True, rng=rng
        )

    if len(coords) < nb_points:
        logger.error(
            "Not enough voxels found to place source points, foune only %s voxels", len(coords)
        )

    dataset = pd.DataFrame(coords, columns=COORDS_COLS).reset_index()
    dataset.rename(columns={"index": "morph_file"}, inplace=True)
    dataset["axon_id"] = 0
    dataset["terminal_id"] = -1
    dataset["section_id"] = -1

    if output_path is not None:
        # TODO: Should export a CellCollection to a MVD3 file?
        dataset[["morph_file", "axon_id", "terminal_id", "section_id", *COORDS_COLS]].to_hdf(
            output_path,
            index=False,
        )

    return dataset
