"""Helpers for white matter recipe."""
import logging
import shutil
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import voxcell
import yaml
from git import Repo
from scipy.spatial.distance import squareform

from axon_synthesis.utils import fill_diag

logger = logging.getLogger(__name__)


@lru_cache
def load(white_matter_file: Path):
    """Load the white matter recipe."""
    logger.debug("Loading white matter recipe file from: %s", white_matter_file)
    with white_matter_file.open("r", encoding="utf-8") as f:
        wm_recipe = yaml.load(f, Loader=yaml.SafeLoader)

    return wm_recipe


def get_atlas_region_id(region_map, pop_row, col_name, second_col_name=None):
    """Get the ID of an atlas region."""

    def get_ids(region_map, pop_row, col_name):
        if not pop_row.isnull()[col_name]:
            acronym = pop_row[col_name]
            ids = region_map.find(acronym, attr="acronym")
        else:
            acronym = None
            ids = []
        return ids, acronym

    ids, acronym = get_ids(region_map, pop_row, col_name)
    if len(ids) == 0 and second_col_name is not None:
        ids, new_acronym = get_ids(region_map, pop_row, second_col_name)
        if len(ids) == 1 and acronym is not None:
            logger.warning(
                "Could not find any ID for %s in the region map but found one for %s",
                acronym,
                new_acronym,
            )
    else:
        new_acronym = None

    if len(ids) > 1:
        raise ValueError(
            f"Found several IDs for the acronym '{acronym or new_acronym}' in the region "
            f"map: {sorted(ids)}"
        )
    if len(ids) == 0:
        raise ValueError(f"Could not find the acronym '{acronym or new_acronym}' in the region map")
    return ids.pop()


def process(
    wm_recipe: dict,
    region_map: voxcell.region_map.RegionMap,
    subregion_uppercase: bool,
    subregion_remove_prefix: bool,
    sub_region_separator: str,
):
    """Process the white matter recipe."""
    # Get populations
    logger.debug("Extracting populations from white matter recipe")
    wm_populations = pd.DataFrame.from_records(wm_recipe["populations"])
    wm_populations_sub = wm_populations.loc[
        wm_populations["atlas_region"].apply(lambda x: isinstance(x, list)),
        "atlas_region",
    ]
    if not wm_populations_sub.empty:
        wm_populations_sub = (
            wm_populations_sub.apply(pd.Series)
            .stack()
            .dropna()
            .rename("atlas_region_split")
            .reset_index(level=1, drop=True)
        )
        wm_populations = wm_populations.join(wm_populations_sub, how="left")
        wm_populations["atlas_region_split"].fillna(wm_populations["atlas_region"], inplace=True)
        wm_populations.drop(columns=["atlas_region"], inplace=True)
        wm_populations.rename(
            columns={
                "atlas_region_split": "atlas_region",
            },
            inplace=True,
        )
    wm_populations.rename(
        columns={
            "name": "pop_raw_name",
        },
        inplace=True,
    )
    wm_populations["region_acronym"] = wm_populations["atlas_region"].apply(lambda row: row["name"])
    wm_populations_sub = (
        wm_populations["atlas_region"]
        .apply(lambda row: pd.Series(row.get("subregions", [])))
        .stack()
        .dropna()
        .rename("sub_region")
        .reset_index(level=1, drop=True)
    )
    wm_populations = wm_populations.join(wm_populations_sub, how="left")

    # Get subregion names
    wm_populations["formatted_subregion"] = wm_populations["sub_region"]
    if subregion_uppercase:
        wm_populations["formatted_subregion"] = wm_populations["formatted_subregion"].str.upper()
    if subregion_remove_prefix:
        wm_populations["formatted_subregion"] = wm_populations["formatted_subregion"].str.extract(
            r"(\d+.*)"
        )
    wm_populations["subregion_acronym"] = (
        wm_populations["region_acronym"]
        + sub_region_separator
        + wm_populations["formatted_subregion"]
    )

    # Get atlas subregion IDs
    wm_populations["atlas_region_id"] = wm_populations.apply(
        lambda row: get_atlas_region_id(region_map, row, "subregion_acronym", "region_acronym"),
        axis=1,
    )

    # Get projections
    logger.debug("Extracting projections from white matter recipe")
    wm_projections = pd.DataFrame.from_records(wm_recipe["projections"])
    if wm_projections["source"].duplicated().any():
        raise ValueError(
            "Found several equal sources in the 'projections' entry: "
            f"{sorted(wm_projections.loc[wm_projections['a'].duplicated(), 'a'].tolist())}"
        )

    # Map projections
    wm_projections = wm_projections.merge(
        wm_populations, left_on="source", right_on="pop_raw_name", how="left"
    )

    wm_targets = (
        wm_projections["targets"]
        .apply(pd.Series)
        .stack()
        .rename("target")
        .reset_index(level=1)
        .rename(columns={"level_1": "target_num"})
    )
    projection_targets = wm_projections.join(wm_targets).set_index("target_num", append=True)
    projection_targets["strength"] = projection_targets["target"].apply(lambda row: row["density"])
    projection_targets["topographical_mapping"] = projection_targets["target"].apply(
        lambda row: row["presynaptic_mapping"]
    )

    # Get fractions
    logger.debug("Extracting fractions from white matter recipe")
    wm_fractions = {i["population"]: i["fractions"] for i in wm_recipe["p-types"]}

    # Get interaction_mat and strengths
    logger.debug("Extracting interaction matrix from white matter recipe")
    wm_interaction_mat = {
        i["population"]: i["interaction_mat"]
        for i in wm_recipe["p-types"]
        if "interaction_mat" in i
    }

    wm_interaction_strengths = {
        k: pd.DataFrame(
            fill_diag(squareform(v["strengths"]), 1),
            columns=wm_interaction_mat[k]["projections"],
            index=wm_interaction_mat[k]["projections"],
        )
        for k, v in wm_interaction_mat.items()
    }

    return (
        wm_populations,
        wm_projections,
        wm_targets,
        wm_fractions,
        wm_interaction_strengths,
        projection_targets,
    )


def fetch(url, output_path, file_path="white_matter_FULL_RECIPE_v1p20.yaml", version=None):
    """Fetch the White Natter Recipe file from our internal repository."""
    with TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "tmp_repo"
        Repo.clone_from(url, dest)
        if version is not None:
            r = Repo(dest)
            r.git.checkout(version)
        shutil.copy(dest / file_path, output_path)
    if version is None:
        version = "latest"
    logger.info(
        (
            "Fetched the White Matter Recipe using the '%s' file from the '%s' repository at "
            "version '%s' to the file '%s'"
        ),
        file_path,
        url,
        version,
        output_path,
    )
