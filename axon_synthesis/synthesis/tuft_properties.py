"""Update the properties of the tufts that will be generated later."""
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import luigi
import luigi_tools.parameter
import luigi_tools.task
import numpy as np
import pandas as pd
from morph_tool import resampling
from neurom import COLS
from neurom import load_morphology
from neurom.morphmath import interval_lengths
from neurom.morphmath import section_length
from scipy.spatial import KDTree

from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import get_axons

logger = logging.getLogger(__name__)


def _exp(values, sigma, default_ind=None) -> np.ndarray:
    if sigma != 0:
        return (
            # 1.0 / (sigma * np.sqrt(2 * np.pi)) *
            np.exp(-np.power(values, 2) / (2.0 * sigma**2))
        )

    new_values = pd.Series(0, index=values.index)
    if default_ind is not None:
        new_values.loc[default_ind] = 1
    return new_values.to_numpy()


def tree_region_lengths(tree, brain_regions):
    """Compute the length of the tree in each region it passes through."""
    # TODO: Use geometry.voxel_intersection for this because this naive implementation may miss
    # some brain region boundary crossing in some cases.
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
            change_region_ind[:-2],
            change_region_ind[1:-1],
            change_region_ind[2:],
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


def load_wmr_data(
    all_tuft_roots,
    population_numbers_path,
    terminals_path,
    wm_fractions_path,
    source_populations_path,
):
    """Get the white matter recipe data."""
    # Get neuron numbers
    pop_neuron_numbers = pd.read_csv(population_numbers_path)

    # Get populations
    target_properties = pd.read_csv(
        terminals_path,
        dtype={"morph_file": str},
    )
    target_properties[["x", "y", "z"]] = target_properties[["x", "y", "z"]].round(6)

    # Add target properties to the tuft roots
    all_tuft_roots = all_tuft_roots.merge(
        target_properties.loc[target_properties["axon_id"] != -1],
        on=["morph_file", "x", "y", "z"],
        how="left",
    )

    # Get populations
    # wm_populations = pd.read_csv(self.input()["target_points"]["wm_populations"].path)

    # Get fractions
    with wm_fractions_path.open("r", encoding="utf-8") as f:
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
        source_populations_path,
        dtype={"morph_file": str},
    )

    pop_numbers = pd.merge(
        source_populations,
        pop_neuron_numbers[
            ["pop_raw_name", "atlas_region_id", "atlas_region_volume", "pop_neuron_numbers"]
        ],
        on="pop_raw_name",
        how="left",
    )

    return all_tuft_roots, wm_fractions, source_populations, pop_numbers


def compute_cluster_properties(
    cluster_props_df,
    axon,
    sec,
    source_populations,
    pop_numbers,
    morph_file,
    wm_fractions,
    tuft_root,
    bouton_density,
    atlas,
    lengths_in_regions,
    rng,
):  # pylint: disable=too-many-arguments
    """Compute properties for the cluster."""
    # Pick a random index for when the probabilities can not be computed
    terminal_index = rng.choice(cluster_props_df.index)

    # Compute raw terminal properties
    terminal_data = {}
    terminal_data["path_distance"] = sum(section_length(i.points) for i in sec.iupstream())
    terminal_data["radial_distance"] = np.linalg.norm(
        axon.points[0, COLS.XYZ] - sec.points[-1, COLS.XYZ],
    )

    # Compute cluster size from white matter recipe
    source = source_populations.loc[
        source_populations["morph_file"] == morph_file,
        "source_population_name",
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
    fraction = wm_fractions[source][tuft_root["target_projection_name"]]
    N_act = N_pot * fraction
    strength = tuft_root["target_density"]
    N_tot = target_region_volume * strength
    n_syn_per = N_tot / N_act
    l_mean = n_syn_per / bouton_density

    # The path length of the future tuft should be equal to the desired length
    # minus the trunk length that lies in the target region
    # TODO: For now we set the minimal length of a tuft to 100 but this should
    # be improved
    terminal_data["path_length"] = max(100, l_mean - lengths_in_regions.get(atlas_region_id, 0))

    # TODO: remove the length of the part of the trunk that is inside the target
    # region

    # Compute the orientation of the tuft from the atlas (the default
    # orientation is up so we take the 2nd row of the rotation matrix)
    # TODO: check that this formula is correct
    terminal_data["cluster_orientation"] = atlas.orientations.lookup(
        sec.points[-1, COLS.XYZ],
    ).tolist()[0][1]

    # TODO: add a shift and a random deviation around this orientation?

    return terminal_index, pd.Series(terminal_data)


class CreateTuftTerminalProperties(luigi_tools.task.WorkflowTask):
    """Task to create the tuft properties used for axon synthesis."""

    size_sigma = luigi.NumericalParameter(
        description="The sigma value used to select the barcode along the size axis.",
        var_type=float,
        default=0,
        min_value=0,
        max_value=sys.float_info.max,
    )
    size_target = luigi.NumericalParameter(
        description="The target value of the size used to select the barcode along the size axis.",
        var_type=int,
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

    # def requires(self):
    #     tasks = {
    #         "clustered_morphologies": ClusterTerminals(),
    #         "steiner_morphologies": SteinerMorphologies(),
    #         "pop_neuron_numbers": PickPopulationNeuronNumbers(),
    #         "terminals": CreateGraph(),
    #     }
    #     if Config().input_data_type == "white_matter":
    #         tasks["target_points"] = FindTargetPoints()
    #         tasks["WMR"] = FetchWhiteMatterRecipe()
    #     return tasks

    def pick_tuft(self, cluster_props_df, axon_terminal, terminal_index=None):
        """Choose a tuft according to the given properties."""
        size_prob = _exp(
            cluster_props_df["cluster_size"] - axon_terminal.get("cluster_size", self.size_target),
            self.size_sigma,
            terminal_index,
        )
        length_prob = _exp(
            cluster_props_df["path_length"] - axon_terminal["path_length"],
            self.length_sigma,
            terminal_index,
        )
        distance_prob = _exp(
            cluster_props_df[self.distance_variable] - axon_terminal[self.distance_variable],
            self.distance_sigma,
            terminal_index,
        )

        prob = length_prob + distance_prob + size_prob
        if prob.sum() <= 1e-8:
            if terminal_index is None:
                prob.loc[:] = 1 / len(prob)
            else:
                prob.loc[:] = 0
                prob.loc[terminal_index] = 1
        else:
            prob /= prob.sum()

        return self.rng.choice(cluster_props_df.index, p=prob)

    @staticmethod
    def get_tuft_root(sec, axon_tree, group):
        """Get the root section of the tuft."""
        last_pt = sec.points[-1, COLS.XYZ]
        dist, index = axon_tree.query(last_pt)
        if dist > 1e-3:
            logger.debug(
                (
                    "Skip section %s with point %s since no tuft root was found near "
                    "this location (the point %s is the closest with %s distance). This probably "
                    "just means that this section is an intermediate section but if all sections "
                    "are skipped then there is an issue."
                ),
                sec.id,
                last_pt,
                group.iloc[index][["x", "y", "z"]].tolist(),
                dist,
            )
            return None

        tuft_root = group.iloc[index]
        logger.debug(
            "Found tuft root for the section %s with point %s at a distance %s",
            sec.id,
            last_pt,
            dist,
        )
        return tuft_root


def create(
    self,
    morph,
    atlas,
    axon_terminals: pd.DataFrame,
    tuft_properties: pd.DataFrame,
    *,
    mode: str = "biological_morphologies",  # Can be "white_matter" or "biological_morphologies"
    output_path: FileType | None = None,
    rng: SeedType = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
):
    """Create the properties for each tuft."""
    rng = np.random.default_rng(rng)

    # Get tuft data from the Steiner morphologies
    all_tuft_roots = axon_terminals
    all_tuft_roots = all_tuft_roots.loc[all_tuft_roots["is_terminal"]]
    all_tuft_roots[["x", "y", "z"]] = all_tuft_roots[["x", "y", "z"]].round(6)

    if mode == "white_matter":
        target_properties = target_properties.copy()
        target_properties[["x", "y", "z"]] = target_properties[["x", "y", "z"]].round(6)

        # Add target properties to the tuft roots
        all_tuft_roots = all_tuft_roots.merge(
            target_properties.loc[target_properties["axon_id"] != -1],
            on=["morph_file", "x", "y", "z"],
            how="left",
        )

    # Ensure morphology file names are strings
    tuft_properties["morph_file"] = tuft_properties["morph_file"].astype(str)

    tuft_properties["old_new_cluster_barcode"] = None
    tuft_properties["new_cluster_barcode"] = None
    tuft_props = []

    for group_name, group in all_tuft_roots.groupby("steiner_morph_file"):
        steiner_morph_file = Path(str(group_name))
        morph_file = group["morph_file"].iloc[0]

        # Load morph (loaded in a persistent Python object to keep C++ objects in memory)
        morph = load_morphology(group_name)

        # Get axons
        axons = get_axons(morph)

        for axon_id, axon in enumerate(axons):
            # Get terminals of the current group
            axon_tree = KDTree(group[["x", "y", "z"]].to_numpy().astype(np.float32))

            # Compute the length of the tree in each brain region
            if mode != "biological_morphologies":
                lengths_in_regions = tree_region_lengths(axon, atlas.brain_regions)

            for sec in axon.iter_sections():
                if sec.parent is None:
                    continue

                tuft_root = self.get_tuft_root(sec, axon_tree, group)
                if tuft_root is None:
                    continue

                if mode == "biological_morphologies":
                    # Use properties from clustered morphologies
                    axon_terminals = tuft_properties.loc[
                        (
                            tuft_properties["morph_file"].str.match(
                                f".*{steiner_morph_file.stem}\\.(asc|swc|h5)",
                            )
                        )
                        & (tuft_properties["axon_id"] == axon_id)
                    ].copy()
                    axon_terminals["dist"] = np.linalg.norm(
                        (
                            np.array(axon_terminals["common_ancestor_coords"].tolist())
                            - tuft_root[["x", "y", "z"]].to_numpy()
                        ).astype(float),
                        axis=1,
                    )
                    axon_terminal = tuft_properties.loc[axon_terminals["dist"].idxmin()]
                    terminal_index = axon_terminal.name
                else:
                    # Compute cluster properties
                    terminal_index, axon_terminal = compute_cluster_properties(
                        tuft_properties,
                        axon,
                        sec,
                        source_populations,
                        pop_numbers,
                        morph_file,
                        wm_fractions,
                        tuft_root,
                        self.bouton_density,
                        atlas,
                        lengths_in_regions,
                        self.rng,
                    )

                # Choose a tuft according to these properties
                chosen_index = self.pick_tuft(tuft_properties, axon_terminal, terminal_index)
                logger.info(
                    (
                        "Chosen tuft for %s, axon %s and ancestor at %s from morph '%s', "
                        "axon %s and ancestor at %s"
                    ),
                    steiner_morph_file,
                    axon_id,
                    tuft_root[["x", "y", "z"]].tolist(),
                    *tuft_properties.loc[
                        chosen_index,
                        ["morph_file", "axon_id", "common_ancestor_coords"],
                    ].tolist(),
                )

                tuft_properties.at[
                    terminal_index,
                    "new_cluster_barcode",
                ] = tuft_properties.at[chosen_index, "cluster_barcode"]
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
                        tuft_properties.at[chosen_index, "cluster_barcode"],
                    ],
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

    if output_path is not None:
        output_path = Path(output_path)
        with output_path.open(mode="w") as f:
            json.dump(tuft_props_df.to_dict("records"), f, indent=4)
