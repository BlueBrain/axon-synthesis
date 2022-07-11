"""Find the target points of the input morphologies."""
import json
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
from axon_synthesis.white_matter_recipe import load as load_wmr
from axon_synthesis.white_matter_recipe import process as process_wmr

logger = logging.getLogger(__name__)


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
    sub_region_separator = luigi.Parameter(
        description="Separator use between region and subregion names to build the acronym.",
        default="",
    )
    seed = luigi.IntParameter(
        description="The seed used to generate random points.",
        default=0,
    )
    min_target_points = luigi.IntParameter(
        description="The minimal number of target points to consider a point as valid.",
        default=2,
    )
    subregion_uppercase = luigi.BoolParameter(
        description=("If set to True, the subregion names are uppercased."),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
    )
    subregion_remove_prefix = luigi.BoolParameter(
        description=(
            "If set to True, only the layer numbers are extracted from the subregion names."
        ),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
    )
    debug_flatmap = luigi.BoolParameter(
        description=("If set to True, the flatmap is exported."),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
    )

    def requires(self):
        return {
            "source_points": CreateSourcePoints(),
            "WMR": FetchWhiteMatterRecipe(),
        }

    def run(self):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
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

        # Get the white matter recipe
        wm_recipe = load_wmr(self.input()["WMR"].pathlib_path)

        # Process the white matter recipe
        (
            wm_populations,
            wm_projections,
            wm_targets,
            wm_fractions,
            wm_interaction_strengths,
            projection_targets,
        ) = process_wmr(
            wm_recipe,
            region_map,
            self.subregion_uppercase,
            self.subregion_remove_prefix,
            self.sub_region_separator,
        )

        # Export the population DataFrame
        wm_populations.to_csv(self.output()["wm_populations"].path, index=False)

        # Export the projection DataFrame
        wm_projections.to_csv(self.output()["wm_projections"].path, index=False)

        # Export the projection DataFrame
        projection_targets.to_csv(self.output()["wm_projection_targets"].path, index=False)

        # Export the fractions
        with self.output()["wm_fractions"].pathlib_path.open("w", encoding="utf-8") as f:
            json.dump(wm_fractions, f)

        # Export the targets DataFrame
        wm_targets.to_csv(self.output()["wm_targets"].path, index=False)

        # Export the interaction strengths
        with self.output()["wm_interaction_strengths"].pathlib_path.open(
            "w", encoding="utf-8"
        ) as f:
            json.dump({k: v.to_dict("index") for k, v in wm_interaction_strengths.items()}, f)

        # Get brain regions from source positions
        source_points["brain_region"] = brain_regions.lookup(source_points[["x", "y", "z"]].values)

        def get_layer(atlas, brain_regions, pos):
            # Get layer data
            # TODO: get layer from the region names?
            names, ids = atlas.get_layers()
            layers = np.zeros_like(brain_regions.raw, dtype="uint8")
            layer_mapping = {}
            for layer_id, (ids_set, layer) in enumerate(zip(ids, names)):
                layer_mapping[layer_id] = layer
                layers[np.isin(brain_regions.raw, list(ids_set))] = layer_id + 1
            layers = brain_regions.with_data(layers)
            return layers.lookup(pos, outer_value=0)

        # Find in which layer is each source point
        source_points["layer"] = get_layer(
            atlas, brain_regions, source_points[["x", "y", "z"]].values
        )

        # Find population name of each source point
        source_points = source_points.merge(
            wm_populations[["pop_raw_name", "atlas_region_id"]],
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

        # Map projections to the source population of each source point
        source_points = source_points.merge(wm_projections, on="pop_raw_name", how="left")
        current_ids = set(source_points["morph_file"].unique())
        source_points.dropna(subset=["source"], inplace=True)
        missing_ids = current_ids.difference(set(source_points["morph_file"].unique()))
        if missing_ids:
            logger.warning(
                "Could not map the projection of the following point IDs: %s",
                missing_ids,
            )

        # Remove duplicates to ensure that duplicated populations have the same probability to be
        # chosen than the others
        source_points.drop_duplicates(subset=["morph_file", "pop_raw_name", "layer"], inplace=True)

        # Choose which population is used for each source point
        source_pop = source_points.groupby("morph_file").sample(random_state=rng.bit_generator)

        if source_pop["pop_raw_name"].isnull().any():
            for i in source_pop.loc[
                source_pop["pop_raw_name"].isnull(),
                ["morph_file", "x", "y", "z", "pop_raw_name"],
            ].to_dict("records"):
                logger.warning("Could not map fractions for: %s", i)
            source_pop.dropna(subset=["pop_raw_name"], inplace=True)

        # Export the source populations
        source_pop.to_csv(self.output()["source_populations"].path, index=False)

        # Compute connections (find regions and pick random coordinates in these regions)
        targets = []
        for _, row in source_pop.iterrows():
            term_id = 1
            row_targets = []
            n_tries = 0
            row_fractions = wm_fractions[row["pop_raw_name"]]
            if not row_fractions:
                logger.warning("No fraction found for %s", row["morph_file"])
                continue
            logger.debug("Fractions for %s: %s", row["morph_file"], row_fractions)
            while not row_targets and n_tries <= 10:
                row_targets = [
                    j
                    for j in list(row["targets"])
                    if rng.random() <= row_fractions[j["projection_name"]]
                ]
                n_tries += 1
            # TODO: Use interaction_map to improve pair-projection probabilities

            for target in row_targets:
                region_id = (
                    wm_populations.loc[
                        wm_populations["pop_raw_name"] == target["population"],
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
                coords = brain_regions.indices_to_positions(voxel)
                coords += np.array(
                    [
                        rng.uniform(
                            -0.5 * np.abs(brain_regions.voxel_dimensions[i]),
                            0.5 * np.abs(brain_regions.voxel_dimensions[i]),
                        )
                        for i in range(3)
                    ]
                )

                # Convert to float32 to avoid rounding error later in MorphIO
                coords = coords.astype(np.float32)

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
        nb_targets = target_df.groupby("morph_file").size()
        invalid_pts = nb_targets < self.min_target_points
        if invalid_pts.any():
            invalid_mask = target_df["morph_file"].isin(invalid_pts.loc[invalid_pts].index)

            for k, v in nb_targets.loc[nb_targets < self.min_target_points].to_dict().items():
                coords = (
                    target_df.loc[target_df["morph_file"] == k, ["x", "y", "z"]]
                    .values.flatten()
                    .tolist()
                )
                logger.warning(
                    "This point have less target points (%s) than the minimum permitted (%s): %s",
                    v,
                    self.min_target_points,
                    [k] + coords,
                )
            target_df.drop(target_df.loc[invalid_mask].index, inplace=True)

        # Add source points as soma points
        soma_points = (
            target_df[["morph_file"]]
            .merge(source_points, on="morph_file")
            .drop_duplicates("morph_file")[["morph_file", "x", "y", "z"]]
        )
        soma_points["axon_id"] = -1
        soma_points["terminal_id"] = -1
        soma_points["target_properties"] = None

        # Add source points as terminals
        root_pts = soma_points.copy(deep=True)
        root_pts["axon_id"] = 0
        root_pts["terminal_id"] = 0

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
            "wm_populations": TargetPointsOutputLocalTarget(
                "white_matter_population.csv", create_parent=True
            ),
            "wm_projections": TargetPointsOutputLocalTarget(
                "white_matter_projections.csv", create_parent=True
            ),
            "wm_projection_targets": TargetPointsOutputLocalTarget(
                "white_matter_projection_targets.csv", create_parent=True
            ),
            "wm_fractions": TargetPointsOutputLocalTarget(
                "white_matter_fractions.csv", create_parent=True
            ),
            "wm_targets": TargetPointsOutputLocalTarget(
                "white_matter_targets.csv", create_parent=True
            ),
            "wm_interaction_strengths": TargetPointsOutputLocalTarget(
                "white_matter_interaction_strengths.csv", create_parent=True
            ),
        }
        if self.debug_flatmap:
            targets["flatmap"] = TargetPointsOutputLocalTarget("flatmap.nrrd", create_parent=True)
        return targets
