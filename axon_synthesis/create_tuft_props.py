"""Update the properties of the tufts that will be generated later."""
import json
import logging
import sys
from ast import literal_eval
from collections import defaultdict
from pathlib import Path

import luigi
import luigi_tools.parameter
import luigi_tools.task
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from morph_tool import resampling
from neurom import COLS
from neurom import load_morphology
from neurom.morphmath import interval_lengths
from neurom.morphmath import section_length
from scipy.spatial import KDTree
from voxcell import OrientationField

from axon_synthesis.atlas import load as load_atlas
from axon_synthesis.config import Config
from axon_synthesis.create_dataset import FetchWhiteMatterRecipe
from axon_synthesis.PCSF.clustering import ClusterTerminals
from axon_synthesis.PCSF.create_graph import CreateGraph
from axon_synthesis.PCSF.steiner_morphologies import SteinerMorphologies
from axon_synthesis.pop_neuron_numbers import PickPopulationNeuronNumbers
from axon_synthesis.target_points import FindTargetPoints
from axon_synthesis.utils import get_axons

logger = logging.getLogger(__name__)


class TuftsOutputLocalTarget(TaggedOutputLocalTarget):
    """Target for tuft outputs."""

    __prefix = Path("tufts")  # pylint: disable=unused-private-member


def _exp(values, sigma, default_ind):
    if sigma != 0:
        return (
            1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-np.power(values, 2) / (2.0 * sigma**2))
        )
    else:
        new_values = pd.Series(0, index=values.index)
        new_values.loc[default_ind] = 1
        return new_values


def tree_region_lengths(tree, brain_regions):
    """Compute the length of the tree in each region it passes through."""
    # TODO: Use geometry.voxel_intersection for this
    lengths = defaultdict(int)
    voxel_dim = brain_regions.voxel_dimensions.min()
    for sec in tree.iter_sections():
        pts = sec.morphio_section.points

        # Resample the tree using the voxel size
        # pylint: disable=protected-access
        ids, fractions = resampling._resample_from_linear_density(
            pts,
            voxel_dim,
        )
        points = resampling._parametric_values(pts, ids, fractions)
        regions = brain_regions.lookup(points)

        # If all points are in the same region, just add the section length
        if (regions == regions[0]).all():
            lengths[regions[0]] += sec.length
            continue

        # Else we estimate the points that cross region boundaries and compute partial lengths
        change_region = regions[:-1] != regions[1:]
        change_region_ind = np.argwhere(change_region).flatten()
        change_region_ind = np.insert(change_region_ind, 0, 0)
        change_region_ind = np.append(change_region_ind, -1)

        for start, ind, end in zip(
            change_region_ind[:-2], change_region_ind[1:-1], change_region_ind[2:]
        ):
            first_region = regions[ind]
            second_region = regions[ind + 1]
            first_pt = points[ind]
            second_pt = points[ind + 1]
            center = np.mean([first_pt, second_pt], axis=0)
            intervals = np.append(points[start:ind], [center], axis=0)
            lengths[first_region] += interval_lengths(intervals).sum()
            intervals = np.append([center], points[ind + 1 : end], axis=0)
            lengths[second_region] += interval_lengths(intervals).sum()

    return dict(lengths)


class CreateTuftTerminalProperties(luigi_tools.task.WorkflowTask):
    """Task to create the tuft properties used for axon synthesis."""

    size_sigma = luigi.NumericalParameter(
        description="The sigma value used to select the barcode along the size axis.",
        var_type=float,
        default=0,
        min_value=0,
        max_value=sys.float_info.max,
    )
    distance_variable = luigi.ChoiceParameter(
        description="The variable name to use to find the distance in the JSON records.",
        choices=["path_distance", "radial_distance"],
        default="path_distance",
    )
    distance_sigma = luigi.NumericalParameter(
        description="The sigma value used to select the barcode along the distance axis.",
        var_type=float,
        default=0,
        min_value=0,
        max_value=sys.float_info.max,
    )
    length_sigma = luigi.NumericalParameter(
        description="The sigma value used to select the barcode along the length axis.",
        var_type=float,
        default=0,
        min_value=0,
        max_value=sys.float_info.max,
    )
    use_cluster_props = luigi.BoolParameter(
        description=(
            "If set to True, the properties of each morphology are taken from the clustering step "
            "instead of computed from the white matter recipe (in this case the steiner "
            "morphologies must have the same name as the clustered morphologies)."
        ),
        default=False,
        parsing=luigi.parameter.BoolParameter.EXPLICIT_PARSING,
    )
    pop_numbers_file = luigi.parameter.OptionalPathParameter(
        description=(
            "Path to a CSV file containing the number of neuron for each population. These numbers "
            "are used to compute the axon lengths."
        ),
        exists=True,
        default=None,
    )
    bouton_density = luigi.parameter.OptionalNumericalParameter(
        description=(
            "Path to a CSV file containing the number of neuron for each population. These numbers "
            "are used to compute the axon lengths."
        ),
        var_type=float,
        default=0.2,
        min_value=0,
        max_value=sys.float_info.max,
        left_op=luigi.parameter.operator.lt,
    )
    seed = luigi.NumericalParameter(
        description="The seed used by the random number generator.",
        var_type=int,
        default=0,
        min_value=0,
        max_value=sys.float_info.max,
    )

    def requires(self):
        tasks = {
            "clustered_morphologies": ClusterTerminals(),
            "steiner_morphologies": SteinerMorphologies(),
            "pop_neuron_numbers": PickPopulationNeuronNumbers(),
            "terminals": CreateGraph(),
        }
        if Config().input_data_type == "white_matter":
            tasks["target_points"] = FindTargetPoints()
            tasks["WMR"] = FetchWhiteMatterRecipe()
        return tasks

    def run(self):
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        rng = np.random.default_rng(self.seed)
        config = Config()

        # Get terminal properties
        # terminals = pd.read_csv(
        #     self.input()["terminals"]["input_terminals"].path, dtype={"morph_file": str}
        # )

        # Get tuft data from the Steiner morphologies
        all_tuft_roots = pd.read_csv(
            self.input()["steiner_morphologies"]["nodes"].path,
            dtype={"morph_file": str},
        )
        all_tuft_roots = all_tuft_roots.loc[all_tuft_roots["is_terminal"]]
        all_tuft_roots[["x", "y", "z"]] = all_tuft_roots[["x", "y", "z"]].round(6)

        if config.input_data_type == "white_matter":
            # Get neuron numbers
            pop_neuron_numbers = pd.read_csv(
                self.input()["pop_neuron_numbers"]["population_numbers"].path
            )

            # Get populations
            target_properties = pd.read_csv(
                self.input()["target_points"]["terminals"].path,
                dtype={"morph_file": str},
            )
            target_properties[["x", "y", "z"]] = target_properties[["x", "y", "z"]].round(6)
            target_properties.loc[
                ~target_properties["target_properties"].isnull(), "target_properties"
            ] = target_properties.loc[
                ~target_properties["target_properties"].isnull(), "target_properties"
            ].apply(
                literal_eval
            )

            # Add target properties to the tuft roots
            all_tuft_roots = all_tuft_roots.merge(
                target_properties.loc[target_properties["axon_id"] != -1],
                on=["morph_file", "x", "y", "z"],
                how="left",
            )

            # Get populations
            # wm_populations = pd.read_csv(self.input()["target_points"]["wm_populations"].path)

            # Get fractions
            with self.input()["WMR"]["wm_fractions"].pathlib_path.open("r", encoding="utf-8") as f:
                wm_fractions = json.load(f)

            # Get targets
            # wm_targets = pd.read_csv(self.input()["WMR"]["wm_targets"].path)

            # Get projections
            # wm_projections = pd.read_csv(self.input()["WMR"]["wm_projections"].path)
            # wm_projection_targets = pd.read_csv(
            #     self.input()["WMR"]["wm_projection_targets"].path
            # )
            # wm_projection_targets["target"] = wm_projection_targets["target"].apply(literal_eval)

            # Get interaction strengths
            # with self.input()["WMR"]["wm_interaction_strengths"].pathlib_path.open(
            #     "r", encoding="utf-8"
            # ) as f:
            #     wm_interation_strengths = {
            #         k: pd.DataFrame.from_records(v) for k, v in json.load(f).items()
            #     }

            # Get target points data
            source_populations = pd.read_csv(
                self.input()["target_points"]["source_populations"].path,
                dtype={"morph_file": str},
            )

            pop_numbers = pd.merge(
                source_populations,
                pop_neuron_numbers[["pop_raw_name", "atlas_region_volume", "pop_neuron_numbers"]],
                on="pop_raw_name",
                how="left",
            )

            # Get atlas data
            atlas, brain_regions, _ = load_atlas(
                str(config.atlas_path),
                config.atlas_region_filename,
                config.atlas_hierarchy_filename,
            )
            atlas_orientations = atlas.load_data("orientation", cls=OrientationField)

        if self.size_sigma == 0 or self.distance_sigma == 0:
            self.size_sigma = 0
            self.distance_sigma = 0

        with self.input()["clustered_morphologies"]["tuft_properties"].open() as f:
            # Get tuft data from the input biological morphologies
            cluster_props_df = pd.DataFrame.from_records(json.load(f))

        # Ensure morphology file names are strings
        cluster_props_df["morph_file"] = cluster_props_df["morph_file"].astype(str)

        cluster_props_df["old_new_cluster_barcode"] = None
        cluster_props_df["new_cluster_barcode"] = None

        # ############################################################ #
        # Old method
        # ############################################################ #
        for group_name, group in cluster_props_df.groupby("morph_file"):
            for terminal_index, terminal in group.iterrows():
                size_prob = _exp(
                    cluster_props_df["cluster_size"] - terminal["cluster_size"],
                    self.size_sigma,
                    terminal_index,
                )
                distance_prob = _exp(
                    cluster_props_df[self.distance_variable] - terminal[self.distance_variable],
                    self.distance_sigma,
                    terminal_index,
                )

                prob = size_prob * distance_prob
                if prob.sum() == 0:
                    prob.loc[terminal_index] = 1
                else:
                    prob /= prob.sum()
                print(
                    "OLD:",
                    group_name,
                    terminal_index,
                    terminal["cluster_size"],
                    terminal[self.distance_variable],
                    prob.idxmax(),
                    prob.max(),
                )

                chosen_index = rng.choice(cluster_props_df.index, p=prob)
                cluster_props_df.at[
                    terminal_index, "old_new_cluster_barcode"
                ] = cluster_props_df.at[chosen_index, "cluster_barcode"]
        # ############################################################ #

        tuft_props = []

        for group_name, group in all_tuft_roots.groupby("steiner_morph_file"):
            steiner_morph_file = Path(str(group_name))
            morph_file = group["morph_file"].iloc[0]

            # Load morph
            morph = load_morphology(group_name)

            # Get axons
            axons = get_axons(morph)

            for axon_id, axon in enumerate(axons):
                # Get terminals of the current group
                axon_tree = KDTree(group[["x", "y", "z"]].values)

                # Compute the length of the tree in each brain region
                lengths_in_regions = tree_region_lengths(axon, brain_regions)

                for sec in axon.iter_sections():
                    if sec.parent is None:
                        continue

                    last_pt = sec.points[-1, COLS.XYZ]
                    dist, index = axon_tree.query(last_pt)
                    if dist > 1e-3:
                        logger.debug(
                            (
                                "Skip section %s with point %s since no tuft root was found near "
                                "this location (the point %s is the closest with %s distance)."
                            ),
                            sec.id,
                            last_pt,
                            group.iloc[index][["x", "y", "z"]].tolist(),
                            dist,
                        )
                        continue

                    tuft_root = group.iloc[index]
                    logger.debug(
                        "Found tuft root for the section %s with point %s at a distance %s",
                        sec.id,
                        last_pt,
                        dist,
                    )

                    if self.use_cluster_props:
                        # Use properties from clustered morphologies
                        axon_terminals = cluster_props_df.loc[
                            (cluster_props_df["morph_file"].str.endswith(steiner_morph_file.name))
                            & (cluster_props_df["axon_id"] == axon_id)
                        ].copy()
                        axon_terminals["dist"] = np.linalg.norm(
                            (
                                np.array(axon_terminals["common_ancestor_coords"].tolist())
                                - tuft_root[["x", "y", "z"]].values
                            ).astype(float),
                            axis=1,
                        )
                        axon_terminal = cluster_props_df.loc[axon_terminals["dist"].idxmin()]
                        terminal_index = axon_terminal.name
                    else:
                        # Pick a random index for when the probabilities can not be computed
                        terminal_index = rng.choice(cluster_props_df.index)

                        # Compute raw terminal properties
                        terminal_data = {}
                        terminal_data["path_distance"] = sum(
                            section_length(i.points) for i in sec.iupstream()
                        )
                        terminal_data["radial_distance"] = np.linalg.norm(
                            axon.points[0, COLS.XYZ] - sec.points[-1, COLS.XYZ]
                        )

                        # Compute cluster size from white matter recipe
                        source = source_populations.loc[
                            source_populations["morph_file"] == morph_file, "source"
                        ].iloc[0]
                        target_region_volume, atlas_region_id, N_pot = pop_numbers.loc[
                            pop_numbers["morph_file"] == morph_file,
                            [
                                "atlas_region_volume",
                                "atlas_region_id",
                                "pop_neuron_numbers",
                            ],
                        ].iloc[0]
                        atlas_region_id = int(atlas_region_id)
                        fraction = wm_fractions[source][
                            tuft_root["target_properties"]["projection_name"]
                        ]
                        N_act = N_pot * fraction
                        strength = tuft_root["target_properties"]["density"]
                        N_tot = target_region_volume * strength
                        n_syn_per = N_tot / N_act
                        l_mean = n_syn_per / self.bouton_density

                        # The path length of the future tuft should be equal to the desired length
                        # minus the trunk length that lies in the target region
                        # TODO: For now we set the minimal length of a tuft to 100 but this should
                        # be improved
                        terminal_data["path_length"] = max(
                            100, l_mean - lengths_in_regions.get(atlas_region_id, 0)
                        )

                        # TODO: remove the length of the part of the trunk that is inside the target
                        # region

                        # Compute the orientation of the tuft from the atlas (the default
                        # orientation is up so we take the 2nd row of the ortation matrix)
                        # TODO: check that this formula is correct
                        terminal_data["cluster_orientation"] = atlas_orientations.lookup(
                            sec.points[-1, COLS.XYZ]
                        ).tolist()[0][1]

                        # TODO: add a shift and a random deviation around this orientation?

                        axon_terminal = pd.Series(terminal_data)

                    # Choose a tuft according to these properties
                    # size_prob = _exp(
                    #     cluster_props_df["cluster_size"]
                    #     - axon_terminal["cluster_size"],
                    #     self.size_sigma,
                    #     terminal_index,
                    # )
                    length_prob = _exp(
                        cluster_props_df["path_length"] - axon_terminal["path_length"],
                        self.length_sigma,
                        terminal_index,
                    )
                    distance_prob = _exp(
                        cluster_props_df[self.distance_variable]
                        - axon_terminal[self.distance_variable],
                        self.distance_sigma,
                        terminal_index,
                    )

                    prob = length_prob * distance_prob
                    if prob.sum() == 0:
                        prob.loc[terminal_index] = 1
                    else:
                        prob /= prob.sum()

                    print(
                        "raw_NEW:",
                        steiner_morph_file,
                        tuft_root["id"],
                        axon_terminal["path_length"],
                        axon_terminal[self.distance_variable],
                        prob.idxmax(),
                        prob.max(),
                    )

                    chosen_index = rng.choice(cluster_props_df.index, p=prob)
                    logger.info(
                        (
                            "Chosen tuft for %s, axon %s and ancestor at %s from morph '%s', "
                            "axon %s and ancestor at %s"
                        ),
                        steiner_morph_file,
                        axon_id,
                        tuft_root[["x", "y", "z"]].tolist(),
                        *cluster_props_df.loc[
                            chosen_index,
                            ["morph_file", "axon_id", "common_ancestor_coords"],
                        ].tolist(),
                    )

                    cluster_props_df.at[
                        terminal_index, "new_cluster_barcode"
                    ] = cluster_props_df.at[chosen_index, "cluster_barcode"]
                    tuft_props.append(
                        [
                            group.iloc[0]["morph_file"],
                            group_name,
                            axon_id,
                            tuft_root["id"],
                            tuft_root[["x", "y", "z"]].tolist(),
                            axon_terminal["path_distance"],
                            axon_terminal["radial_distance"],
                            axon_terminal["path_length"],
                            # axon_terminal["cluster_size"],
                            axon_terminal["cluster_orientation"],
                            cluster_props_df.at[chosen_index, "cluster_barcode"],
                        ]
                    )

        tuft_props_df = pd.DataFrame(
            tuft_props,
            columns=[
                "morph_file",
                "steiner_morph_file",
                "axon_id",
                "common_ancestor_id",
                "common_ancestor_coords",
                "path_distance",
                "radial_distance",
                "path_length",
                # "cluster_size",
                "cluster_orientation",
                "new_cluster_barcode",
            ],
        )
        tuft_props_df.sort_values(["morph_file", "axon_id", "common_ancestor_id"], inplace=True)

        with self.output().open(mode="w") as f:
            json.dump(tuft_props_df.to_dict("records"), f, indent=4)

    def output(self):
        return TuftsOutputLocalTarget("tuft_terminals.csv", create_parent=True)
