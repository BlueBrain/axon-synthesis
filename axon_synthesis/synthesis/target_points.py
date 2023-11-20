"""Find the target points of the input morphologies."""
import logging

import h5py
import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget

from axon_synthesis.atlas import load as load_atlas
from axon_synthesis.config import Config
from axon_synthesis.create_dataset import FetchWhiteMatterRecipe
from axon_synthesis.prepare_atlas import PrepareAtlas
from axon_synthesis.source_points import CreateSourcePoints
from axon_synthesis.utils import get_layers
from axon_synthesis.white_matter_recipe import load_wmr_data

logger = logging.getLogger(__name__)


def remove_invalid_points(target_df, min_target_points):
    """Remove points with less target points than the minimum permitted."""
    nb_targets = target_df.groupby(["morph_file", "axon_id"]).size()
    invalid_pts = nb_targets < min_target_points
    if invalid_pts.any():
        # invalid_mask = target_df["morph_file"].isin(invalid_pts.loc[invalid_pts].index)
        invalid_mask = target_df.merge(
            invalid_pts.rename("__is_not_valid__"),
            left_on=["morph_file", "axon_id"],
            right_index=True,
            how="left",
        )["__is_not_valid__"]

        for k, v in nb_targets.loc[nb_targets < min_target_points].to_dict().items():
            coords = (
                target_df.loc[
                    (target_df[["morph_file", "axon_id"]] == k).any(axis=1),
                    ["x", "y", "z"],
                ]
                .values.flatten()
                .tolist()
            )
            logger.warning(
                "This point have less target points (%s) than the minimum permitted (%s): %s",
                v,
                min_target_points,
                [k] + coords,
            )
        target_df.drop(target_df.loc[invalid_mask].index, inplace=True)


class TargetPointsOutputLocalTarget(TaggedOutputLocalTarget):
    """Target for target point outputs."""

    __prefix = "target_points"  # pylint: disable=unused-private-member


class FindTargetPoints(luigi_tools.task.WorkflowTask):
    """Task to find the target points used for axon synthesis."""

    source_points = luigi.parameter.OptionalPathParameter(
        description="Path to the terminals CSV file.",
        default=None,
        exists=True,
    )
    output_terminals = luigi.Parameter(
        description="Output dataset file",
        default="target_terminals.csv",
    )
    output_source_populations = luigi.Parameter(
        description="Output source population dataset file",
        default="source_populations.csv",
    )
    seed = luigi.IntParameter(
        description="The seed used to generate random points.",
        default=0,
    )
    min_target_points = luigi.IntParameter(
        description="The minimal number of target points to consider a point as valid.",
        default=2,
    )
    debug_flatmap = luigi.BoolParameter(
        description=("If set to True, the flatmap is exported."),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
    )

    # Attributes that are populated in the run() method
    wm_populations = None
    wm_projections = None
    wm_projection_targets = None
    wm_fractions = None
    wm_layer_profiles = None
    region_data = None

    def requires(self):
        return {
            "brain_region_masks": PrepareAtlas(),
            "source_points": CreateSourcePoints(),
            "WMR": FetchWhiteMatterRecipe(),
        }

    def load_wmr(self):
        """Get the white matter recipe data."""
        data = load_wmr_data(
            wm_populations_path=self.input()["WMR"]["wm_populations"].pathlib_path,
            wm_projections_path=self.input()["WMR"]["wm_projections"].pathlib_path,
            wm_projection_targets_path=self.input()["WMR"]["wm_projection_targets"].pathlib_path,
            wm_fractions_path=self.input()["WMR"]["wm_fractions"].pathlib_path,
            wm_layer_profiles_path=self.input()["WMR"]["wm_layer_profiles"].pathlib_path,
            region_data_path=self.input()["WMR"]["region_data"].pathlib_path,
        )
        self.wm_populations = data["wm_populations"]
        self.wm_projections = data["wm_projections"]
        self.wm_projection_targets = data["wm_projection_targets"]
        self.wm_fractions = data["wm_fractions"]
        self.wm_layer_profiles = data["wm_layer_profiles"]
        self.region_data = data["region_data"]

    def find_pop_names(self, source_points):
        """Find population name of each source point."""
        source_points = source_points.merge(
            self.wm_populations[["pop_raw_name", "atlas_region_id"]],
            left_on="brain_region",
            right_on="atlas_region_id",
            how="left",
        ).drop(columns=["atlas_region_id"])

        if source_points["pop_raw_name"].isnull().any():
            for i in source_points.loc[
                source_points["pop_raw_name"].isnull(),
                ["morph_file", "x", "y", "z", "brain_region"],
            ].to_dict("records"):
                logger.warning("Could not find the population name of: %s", i)

            source_points.dropna(subset=["pop_raw_name"], inplace=True)

        return source_points

    def map_projections(self, source_points, rng=np.random):
        """Map a projection to each source point."""
        projection_targets = self.wm_projection_targets.loc[
            ~self.wm_projection_targets["target_region_atlas_id"].isnull()
        ]
        projection_targets = projection_targets.fillna(
            {"target_subregion_atlas_id": projection_targets["target_region_atlas_id"]},
        ).astype({"target_region_atlas_id": int, "target_subregion_atlas_id": int})

        # Choose one population for each source point
        unique_source_points = source_points.groupby("morph_file").sample()

        # Find all targets for each source point
        all_targets = unique_source_points.merge(
            projection_targets[
                [
                    "pop_raw_name",
                    "target_density",
                    "target_population_name",
                    "target_projection_name",
                    "target_region",
                    "target_subregion_acronym",
                    "target_subregion_atlas_id",
                    "target_layer_profile_region_prob",
                ]
            ],
            on="pop_raw_name",
            how="left",
        )

        # TODO: Should we remove duplicated in all_targets? These duplicates are for layers 2 and
        # 3, so we have to check what does mean to target layer '23': same proba for each or one
        # proba for the union of them?

        # Choose targets for each source point
        projection_matrix = (
            pd.DataFrame.from_records(self.wm_fractions)
            .stack()
            .rename("target_projection_strength")
            .reset_index()
            .rename(columns={"level_0": "target_projection_name", "level_1": "pop_raw_name"})
        )
        all_targets = all_targets.merge(
            projection_matrix,
            on=["pop_raw_name", "target_projection_name"],
            how="left",
        )
        target_probs = (
            all_targets["target_projection_strength"]
            * all_targets["target_layer_profile_region_prob"]
        )
        target_projection_chooser = rng.uniform(size=len(all_targets))

        selected_targets = all_targets.loc[target_projection_chooser <= target_probs]
        n_tries = 0
        while n_tries < 10 and not unique_source_points[["morph_file", "axon_id"]].sort_values(
            ["morph_file", "axon_id"],
        ).reset_index(drop=True).equals(
            selected_targets[["morph_file", "axon_id"]]
            .drop_duplicates()
            .sort_values(["morph_file", "axon_id"])
            .reset_index(drop=True),
        ):
            n_tries += 1
            target_projection_chooser = rng.uniform(size=len(all_targets))
            selected_targets = all_targets.loc[target_projection_chooser <= target_probs]

        current_ids = set(source_points["morph_file"].unique())
        missing_ids = current_ids.difference(set(selected_targets["morph_file"].unique()))
        if missing_ids:
            logger.warning(
                "Could not map the projection of the following point IDs: %s",
                missing_ids,
            )
        return selected_targets

    def run(self):
        rng = np.random.default_rng(self.seed)
        config = Config()

        # Get source points
        source_points = pd.read_csv(
            self.source_points or self.input()["source_points"]["terminals"].pathlib_path,
        )

        # Get atlas data
        atlas, brain_regions, region_map = load_atlas(  # pylint: disable=unused-variable
            str(config.atlas_path),
            config.atlas_region_filename,
            config.atlas_hierarchy_filename,
        )

        # Get brain region masks
        brain_regions_mask_file = h5py.File(self.input()["brain_region_masks"].path)

        # Get the white matter recipe data
        self.load_wmr()

        # Join region data to region map (keep only the ones uses in populations)
        # region_map_df = region_map.as_dataframe().merge(
        #     self.region_data, left_on="atlas_id", right_on="atlas_region_id"
        # ).drop(columns=["ontology_id", "color_hex_triplet"])

        # Get brain regions from source positions
        source_points["brain_region"] = brain_regions.lookup(source_points[["x", "y", "z"]].values)

        # Find in which layer is each source point
        source_points["layer"] = get_layers(
            atlas,
            brain_regions,
            source_points[["x", "y", "z"]].values,
        )

        # Find population name of each source point
        source_points = self.find_pop_names(source_points)

        # Map projections to the source population of each source point
        target_points = self.map_projections(source_points, rng)

        # Remove useless columns
        target_points = target_points[
            [
                "morph_file",
                "axon_id",
                "pop_raw_name",
                "target_population_name",
                "target_projection_name",
                "target_region",
                "target_subregion_acronym",
                "target_subregion_atlas_id",
                "target_density",
                "target_layer_profile_region_prob",
                "target_projection_strength",
            ]
        ].rename(
            columns={
                "pop_raw_name": "source_population_name",
                "target_population_name": "pop_raw_name",
            },
        )

        # Export the source populations
        target_points.to_csv(self.output()["source_populations"].path, index=False)

        logger.debug("Pick random points in regions")

        # Pick a random voxel in each region
        target_points["target_voxel_coords"] = target_points["target_subregion_atlas_id"].apply(
            # lambda row: rng.choice(brain_regions_mask_file[str(999)][:])
            lambda row: rng.choice(brain_regions_mask_file[str(row)][:]),
        )

        # Compute coordinates of this voxel and add a random component up to the voxel size
        # and convert to float32 to avoid rounding error later in MorphIO
        target_points["target_point_coords"] = target_points["target_voxel_coords"].apply(
            lambda row: brain_regions.indices_to_positions(row)
            + np.array(
                [
                    rng.uniform(
                        -0.5 * np.abs(brain_regions.voxel_dimensions[i]),
                        0.5 * np.abs(brain_regions.voxel_dimensions[i]),
                    )
                    for i in range(3)
                ],
            ).astype(np.float32),
        )

        target_points[["x", "y", "z"]] = pd.DataFrame(
            target_points["target_point_coords"].to_list(),
            index=target_points.index,
        )

        # Build terminal IDs inside groups
        counter = target_points[["morph_file", "axon_id"]].copy()
        counter["counter"] = 1
        target_points["terminal_id"] = counter.groupby(["morph_file", "axon_id"])[
            "counter"
        ].cumsum()

        # Format the results
        target_df = target_points[
            [
                "morph_file",
                "axon_id",
                "terminal_id",
                "x",
                "y",
                "z",
                "target_projection_name",
                "target_density",
            ]
        ].copy()

        # Discard invalid points
        remove_invalid_points(target_df, self.min_target_points)

        # Add source points as soma points
        soma_points = (
            target_df[["morph_file"]]
            .merge(source_points, on="morph_file")
            .drop_duplicates("morph_file")[["morph_file", "x", "y", "z"]]
        )
        soma_points[["axon_id", "terminal_id", "target_projection_name", "target_density"]] = (
            -1,
            -1,
            None,
            np.nan,
        )

        # Add source points as terminals
        root_pts = soma_points.copy(deep=True)
        root_pts[["axon_id", "terminal_id"]] = (0, 0)

        target_df = pd.concat([target_df, soma_points, root_pts], ignore_index=True)
        target_df.sort_values(["morph_file", "axon_id", "terminal_id"], inplace=True)

        logger.info(
            "Found enough targets for %s source points",
            len(target_df.groupby("morph_file")),
        )

        # Export the results
        target_df.to_csv(self.output()["terminals"].path, index=False)

    def output(self):
        targets = {
            "terminals": TaggedOutputLocalTarget(self.output_terminals, create_parent=True),
            "source_populations": TaggedOutputLocalTarget(
                self.output_source_populations,
                create_parent=True,
            ),
        }
        # if self.debug_flatmap:
        #     targets["flatmap"] = TargetPointsOutputLocalTarget("flatmap.nrrd", create_parent=True)
        return targets
