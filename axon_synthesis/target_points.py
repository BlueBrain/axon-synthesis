"""Find the target points of the input morphologies."""
import logging

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget

from axon_synthesis.atlas import load as load_atlas
from axon_synthesis.config import Config
from axon_synthesis.create_dataset import FetchWhiteMatterRecipe
from axon_synthesis.source_points import CreateSourcePoints
from axon_synthesis.utils import get_layers
from axon_synthesis.white_matter_recipe import load_WMR_data

logger = logging.getLogger(__name__)


def remove_invalid_points(target_df, min_target_points):
    """Remove points with less target points than the minimum permitted."""
    nb_targets = target_df.groupby("morph_file").size()
    invalid_pts = nb_targets < min_target_points
    if invalid_pts.any():
        invalid_mask = target_df["morph_file"].isin(invalid_pts.loc[invalid_pts].index)

        for k, v in nb_targets.loc[nb_targets < min_target_points].to_dict().items():
            coords = (
                target_df.loc[target_df["morph_file"] == k, ["x", "y", "z"]]
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
        description="Output dataset file", default="target_terminals.csv"
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
    wm_targets = None
    projection_targets = None
    wm_fractions = None
    wm_interaction_strengths = None

    def requires(self):
        return {
            "source_points": CreateSourcePoints(),
            "WMR": FetchWhiteMatterRecipe(),
        }

    def load_WMR(self):
        """Get the white matter recipe data."""
        (
            self.wm_populations,
            self.wm_projections,
            self.wm_targets,
            self.projection_targets,
            self.wm_fractions,
            self.wm_interaction_strengths,
        ) = load_WMR_data(
            self.input()["WMR"]["wm_populations"].pathlib_path,
            self.input()["WMR"]["wm_projections"].pathlib_path,
            None,
            None,
            self.input()["WMR"]["wm_fractions"].pathlib_path,
            None,
        )

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

    def map_projections(self, source_points):
        """Map a projection to each source point."""
        source_points = source_points.merge(self.wm_projections, on="pop_raw_name", how="left")
        current_ids = set(source_points["morph_file"].unique())
        source_points.dropna(subset=["source"], inplace=True)
        missing_ids = current_ids.difference(set(source_points["morph_file"].unique()))
        if missing_ids:
            logger.warning(
                "Could not map the projection of the following point IDs: %s",
                missing_ids,
            )
        return source_points

    def drop_null_fractions(self, source_pop):
        """Find population name of each source point."""
        if source_pop["pop_raw_name"].isnull().any():
            for i in source_pop.loc[
                source_pop["pop_raw_name"].isnull(),
                ["morph_file", "x", "y", "z", "pop_raw_name"],
            ].to_dict("records"):
                logger.warning("Could not map fractions for: %s", i)
            source_pop = source_pop.dropna(subset=["pop_raw_name"])
        return source_pop

    def run(self):
        rng = np.random.default_rng(self.seed)
        config = Config()

        # Get source points
        source_points = pd.read_csv(
            self.source_points or self.input()["source_points"]["terminals"].pathlib_path
        )

        # Get atlas data
        atlas, brain_regions, region_map = load_atlas(
            str(config.atlas_path),
            config.atlas_region_filename,
            config.atlas_hierarchy_filename,
        )

        # Get the white matter recipe data
        self.load_WMR()

        # Get brain regions from source positions
        source_points["brain_region"] = brain_regions.lookup(source_points[["x", "y", "z"]].values)

        # Find in which layer is each source point
        source_points["layer"] = get_layers(
            atlas, brain_regions, source_points[["x", "y", "z"]].values
        )

        # Find population name of each source point
        source_points = self.find_pop_names(source_points)

        # Map projections to the source population of each source point
        source_points = self.map_projections(source_points)

        # Remove duplicates to ensure that duplicated populations have the same probability to be
        # chosen than the others
        source_points.drop_duplicates(subset=["morph_file", "pop_raw_name", "layer"], inplace=True)

        # Choose which population is used for each source point
        source_pop = source_points.groupby("morph_file").sample(random_state=rng.bit_generator)

        # Remove source populations with a null fraction
        source_pop = self.drop_null_fractions(source_pop)

        # Export the source populations
        source_pop.to_csv(self.output()["source_populations"].path, index=False)

        # Compute connections (find regions and pick random coordinates in these regions)
        targets = []
        for _, row in source_pop.iterrows():
            term_id = 1
            row_targets = []
            n_tries = 0
            row_fractions = self.wm_fractions[row["pop_raw_name"]]
            if not row_fractions:
                logger.warning("No fraction found for %s", row["morph_file"])
                continue

            logger.debug("Fractions for %s: %s", row["morph_file"], row_fractions)
            logger.debug('Value of row["targets"]: %s', row["targets"])

            while not row_targets and n_tries <= 10:
                row_targets = [
                    j
                    for j in list(row["targets"])
                    if rng.random() <= row_fractions[j["projection_name"]]
                ]
                n_tries += 1
            # TODO: Use interaction_map to improve pair-projection probabilities

            logger.debug("Targets found: %s", row_targets)

            for target in row_targets:
                logger.debug(
                    "Potential populations:\n%s",
                    self.wm_populations.loc[
                        self.wm_populations["pop_raw_name"] == target["population"],
                        "atlas_region_id",
                    ],
                )
                region_id = (
                    self.wm_populations.loc[
                        self.wm_populations["pop_raw_name"] == target["population"],
                        "atlas_region_id",
                    ]
                    .sample(random_state=rng.bit_generator)
                    .iloc[0]
                )

                logger.debug(
                    "Density for %s - %s: %s", row["morph_file"], term_id, target["density"]
                )

                # TODO: Create several targets in the region where the density is high? => We could
                # find a ratio between input axons and number of tufts in the region?

                # TODO: Use the topographical mapping to refine targeted area in the target region

                # Get a random voxel where the brain region value is equal to the target id
                voxel = rng.choice(
                    np.argwhere(
                        np.isin(
                            brain_regions.raw,
                            list(region_map.find(region_id, attr="id", with_descendants=True)),
                        )
                    ),
                    1,
                )[0]

                # Compute coordinates of this voxel and add a random component up to the voxel size
                # and convert to float32 to avoid rounding error later in MorphIO
                coords = brain_regions.indices_to_positions(voxel) + np.array(
                    [
                        rng.uniform(
                            -0.5 * np.abs(brain_regions.voxel_dimensions[i]),
                            0.5 * np.abs(brain_regions.voxel_dimensions[i]),
                        )
                        for i in range(3)
                    ]
                ).astype(np.float32)

                targets.append([row["morph_file"], 0, term_id] + coords.tolist() + [target])
                term_id += 1

        # Format the results
        target_df = pd.DataFrame(
            targets,
            columns=[
                "morph_file",
                "axon_id",
                "terminal_id",
                "x",
                "y",
                "z",
                "target_properties",
            ],
        )

        # Discard invalid points
        remove_invalid_points(target_df, self.min_target_points)

        # Add source points as soma points
        soma_points = (
            target_df[["morph_file"]]
            .merge(source_points, on="morph_file")
            .drop_duplicates("morph_file")[["morph_file", "x", "y", "z"]]
        )
        soma_points[["axon_id", "terminal_id", "target_properties"]] = (-1, -1, None)

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
                self.output_source_populations, create_parent=True
            ),
        }
        # if self.debug_flatmap:
        #     targets["flatmap"] = TargetPointsOutputLocalTarget("flatmap.nrrd", create_parent=True)
        return targets
