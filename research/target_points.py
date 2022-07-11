"""Find the target points of the input morphologies."""
import logging
import yaml

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas

from config import Config
from source_points import CreateSourcePoints

logger = logging.getLogger(__name__)


def _fill_diag(mat, val=1):
    np.fill_diagonal(mat, val)
    return mat


class TargetPointsOutputLocalTarget(luigi_tools.target.OutputLocalTarget):
    __prefix = "target_points"


class FindTargetPoints(luigi_tools.task.WorkflowTask):
    source_points = luigi_tools.parameter.OptionalPathParameter(
        description="Path to the terminals CSV file.",
        default=None,
        exists=True,
    )
    output_dataset = luigi.Parameter(
        description="Output dataset file", default="terminals.csv"
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
        return CreateSourcePoints()

    def run(self):
        rng = np.random.default_rng(self.seed)
        config = Config()

        # Get source points
        source_points = pd.read_csv(
            self.source_points or self.input()["terminals"].pathlib_path
        )

        # Get atlas data
        logger.info(f"Loading atlas from: {str(config.atlas_path)}")
        atlas = Atlas.open(str(config.atlas_path))

        logger.debug(
            f"Loading brain regions from the atlas using: {config.atlas_region_filename}"
        )
        brain_regions = atlas.load_data(config.atlas_region_filename)

        logger.debug(
            f"Loading region map from the atlas using: {config.atlas_hierarchy_filename}"
        )
        region_map = atlas.load_region_map(config.atlas_hierarchy_filename)

        # if config.atlas_flatmap_filename is None:
        #     # Create the flatmap of the atlas
        #     logger.debug("Building flatmap")
        #     one_layer_flatmap = np.mgrid[
        #         : brain_regions.raw.shape[2],
        #         : brain_regions.raw.shape[0],
        #     ].T[:, :, ::-1]
        #     flatmap = VoxelData(
        #         np.stack([one_layer_flatmap] * brain_regions.raw.shape[1], axis=1),
        #         voxel_dimensions=brain_regions.voxel_dimensions,
        #     )
        # else:
        #     # Load the flatmap of the atlas
        #     flatmap = atlas.load_data(config.atlas_flatmap_filename)

        # if self.debug_flatmap:
        #     logger.debug(f"Saving flatmap to: {self.output()['flatmap'].path}")
        #     flatmap.save_nrrd(self.output()["flatmap"].path, encoding="raw")

        # Get the white matter recipe
        logger.debug(
            f"Loading white matter recipe file from: {config.white_matter_file}"
        )
        with config.white_matter_file.open("r", encoding="utf-8") as f:
            wm_recipe = yaml.load(f, Loader=yaml.SafeLoader)

        # Get populations
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
            wm_populations["atlas_region_split"].fillna(
                wm_populations["atlas_region"], inplace=True
            )
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
        wm_populations["region_acronym"] = wm_populations["atlas_region"].apply(
            lambda row: row["name"]
        )
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
        if self.subregion_uppercase:
            wm_populations["formatted_subregion"] = wm_populations[
                "formatted_subregion"
            ].str.upper()
        if self.subregion_remove_prefix:
            wm_populations["formatted_subregion"] = wm_populations[
                "formatted_subregion"
            ].str.extract(r"(\d+.*)")
        wm_populations["subregion_acronym"] = (
            wm_populations["region_acronym"]
            + self.sub_region_separator
            + wm_populations["formatted_subregion"]
        )

        def get_atlas_region_id(region_map, pop_row, col_name, second_col_name=None):
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
                        f"Could not find any ID for {acronym} in the region map but found one for "
                        f"{new_acronym}"
                    )
            else:
                new_acronym = None
            if len(ids) > 1:
                raise ValueError(
                    f"Found several IDs for the acronym '{acronym or new_acronym}' in the region "
                    f"map: {sorted(ids)}"
                )
            elif len(ids) == 0:
                raise ValueError(
                    f"Could not find the acronym '{acronym or new_acronym}' in the region map"
                )
            return ids.pop()

        # Get atlas subregion IDs
        wm_populations["atlas_region_id"] = wm_populations.apply(
            lambda row: get_atlas_region_id(
                region_map, row, "subregion_acronym", "region_acronym"
            ),
            axis=1,
        )

        # Get projections
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
        projection_targets = wm_projections.join(wm_targets).set_index(
            "target_num", append=True
        )
        projection_targets["strength"] = projection_targets["target"].apply(
            lambda row: row["density"]
        )
        projection_targets["topographical_mapping"] = projection_targets[
            "target"
        ].apply(lambda row: row["presynaptic_mapping"])

        # Get brain regions from source positions
        source_points["brain_region"] = brain_regions.lookup(
            source_points[["x", "y", "z"]].values
        )

        def get_layer(atlas, brain_regions, pos):
            # Get layer data
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
                logger.warning(f"Could not find the population name of: {i}")

            source_points.dropna(subset=["pop_raw_name"], inplace=True)

        # Map projections to the source population of each source point
        source_points = source_points.merge(
            wm_projections, on="pop_raw_name", how="left"
        )
        current_ids = set(source_points["morph_file"].unique())
        source_points.dropna(subset=["source"], inplace=True)
        missing_ids = current_ids.difference(set(source_points["morph_file"].unique()))
        if missing_ids:
            logger.warning(
                f"Could not map the projection of the following point IDs: {missing_ids}"
            )

        # Remove duplicates to ensure that duplicated populations have the same probability to be
        # chosen than the others
        source_points.drop_duplicates(
            subset=["morph_file", "pop_raw_name", "layer"], inplace=True
        )

        # Choose which population is used for each source point
        source_pop = source_points.groupby("morph_file").sample(
            random_state=rng.bit_generator
        )

        if source_pop["pop_raw_name"].isnull().any():
            for i in source_pop.loc[
                source_pop["pop_raw_name"].isnull(),
                ["morph_file", "x", "y", "z", "pop_raw_name"],
            ].to_dict("records"):
                logger.warning(f"Could not map fractions for: {i}")
            source_pop.dropna(subset=["pop_raw_name"], inplace=True)

        # Get fractions
        wm_fractions = {i["population"]: i["fractions"] for i in wm_recipe["p-types"]}

        # Get interaction_mat and strengths
        wm_interaction_mat = {
            i["population"]: i["interaction_mat"]
            for i in wm_recipe["p-types"]
            if "interaction_mat" in i
        }

        wm_interaction_strengths = {
            k: pd.DataFrame(
                _fill_diag(squareform(v["strengths"]), 1),
                columns=wm_interaction_mat[k]["projections"],
                index=wm_interaction_mat[k]["projections"],
            )
            for k, v in wm_interaction_mat.items()
        }

        # Compute connections (find regions and pick random coordinates in these regions)
        targets = []
        for source_index, row in source_pop.iterrows():
            term_id = 1
            row_targets = []
            n_tries = 0
            row_fractions = wm_fractions[row["pop_raw_name"]]
            if not row_fractions:
                logger.warning(f"No fraction found for {row['morph_file']}")
                continue
            logger.debug(f"Fractions for {row['morph_file']}: {row_fractions}")
            while not row_targets and n_tries <= 10:
                row_targets = [
                    j
                    for j in [i for i in row["targets"]]
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
                    f"Density for {row['morph_file']} - {term_id}: {target['density']}"
                )

                # Get a random voxel where the brain region value is equal to the target id
                voxel = rng.choice(
                    np.argwhere(
                        np.isin(
                            brain_regions.raw,
                            list(
                                region_map.find(
                                    region_id, attr="id", with_descendants=True
                                )
                            ),
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

                targets.append([row["morph_file"], 0, term_id] + coords.tolist())
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
            ],
        )

        # Discard invalid points
        nb_targets = target_df.groupby("morph_file").size()
        invalid_pts = nb_targets < self.min_target_points
        if invalid_pts.any():
            invalid_mask = target_df["morph_file"].isin(
                invalid_pts.loc[invalid_pts].index
            )

            for k, v in (
                nb_targets.loc[nb_targets < self.min_target_points].to_dict().items()
            ):
                coords = (
                    target_df.loc[target_df["morph_file"] == k, ["x", "y", "z"]]
                    .values.flatten()
                    .tolist()
                )
                logger.warning(
                    f"This point have less target points ({v}) than the minimum permitted "
                    f"({self.min_target_points}): "
                    f"{[k] + coords}"
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

        # Add source points as terminals
        root_pts = soma_points.copy(deep=True)
        root_pts["axon_id"] = 0
        root_pts["terminal_id"] = 0

        target_df = pd.concat([target_df, soma_points, root_pts], ignore_index=True)
        target_df.sort_values(["morph_file", "axon_id", "terminal_id"], inplace=True)

        logger.info(
            "Found enough targets for {} source points".format(
                len(target_df.groupby("morph_file"))
            )
        )

        # Export the results
        target_df.to_csv(self.output()["terminals"].path, index=False)

    def output(self):
        targets = {
            "terminals": luigi_tools.target.OutputLocalTarget(
                self.output_dataset, create_parent=True
            )
        }
        if self.debug_flatmap:
            targets["flatmap"] = TargetPointsOutputLocalTarget(
                "flatmap.nrrd", create_parent=True
            )
        return targets
