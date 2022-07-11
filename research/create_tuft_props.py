"""Update the properties of the tufts that will be generated later."""
import sys
import json
import logging
from pathlib import Path

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from neurom import COLS
from neurom import load_morphology
from neurom.morphmath import section_length
from scipy.spatial import KDTree

from PCSF.clustering import ClusterTerminals
from PCSF.steiner_morphologies import SteinerMorphologies
from utils import get_axons
from utils import neurite_to_graph

logger = logging.getLogger(__name__)


class TuftsOutputLocalTarget(TaggedOutputLocalTarget):
    __prefix = Path("tufts")


def _exp(values, sigma, default_ind):
    if sigma != 0:
        return (
            1.0
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-np.power(values, 2) / (2.0 * sigma ** 2))
        )
    else:
        new_values = pd.Series(0, index=values.index)
        new_values.loc[default_ind] = 1
        return new_values


class CreateTuftTerminalProperties(luigi_tools.task.WorkflowTask):

    size_sigma = luigi.NumericalParameter(
        description="The sigma value used to select the barcode along the size axis.",
        var_type=float,
        default=0,
        min_value=0,
        max_value=sys.float_info.max,
    )
    distance_variable = luigi.Parameter(
        description="The variable name to use to find the distance in the JSON records.",
        default="path_distance",
    )
    distance_sigma = luigi.NumericalParameter(
        description="The sigma value used to select the barcode along the distance axis.",
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
    pop_numbers_file = luigi_tools.parameter.OptionalPathParameter(
        description=(
            "Path to a CSV file containing the number of neuron for each population. These numbers "
            "are used to compute the axon lengths."
        ),
        exists=True,
        default=None,
    )
    seed = luigi.NumericalParameter(
        description="The seed used by the random number generator.",
        var_type=int,
        default=0,
        min_value=0,
        max_value=float("inf"),
    )

    def requires(self):
        return {
            "clustered_morphologies": ClusterTerminals(),
            "steiner_morphologies": SteinerMorphologies(),
        }

    def run(self):

        rng = np.random.default_rng(self.seed)
        all_tuft_roots = pd.read_csv(self.requires()["steiner_morphologies"].input()["steiner_tree"]["nodes"].path)

        if self.size_sigma == 0 or self.distance_sigma == 0:
            self.size_sigma = 0
            self.distance_sigma = 0

        with self.input()["clustered_morphologies"]["tuft_properties"].open() as f:
            cluster_props_df = pd.DataFrame.from_records(json.load(f))

        cluster_props_df["old_new_cluster_barcode"] = None
        cluster_props_df["new_cluster_barcode"] = None

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
                print("OLD:", group_name, terminal_index, terminal["cluster_size"], terminal[self.distance_variable], prob.idxmax(), prob.max())

                chosen_index = rng.choice(cluster_props_df.index, p=prob)
                cluster_props_df.at[terminal_index, "old_new_cluster_barcode"] = cluster_props_df.at[chosen_index, "cluster_barcode"]

        tuft_props = []

        for group_name, group in all_tuft_roots.groupby("morph_file"):
            raw_morph_file = Path(group_name)
            morph_file = self.input()["steiner_morphologies"].pathlib_path / raw_morph_file.with_suffix(".asc").name

            # Load morph
            morph = load_morphology(morph_file)

            # Get axons
            axons = get_axons(morph)

            for axon_id, axon in enumerate(axons):
                # Get terminals
                tuft_roots = all_tuft_roots.loc[
                    (
                        all_tuft_roots["morph_file"].str.endswith(
                            morph_file.name
                        )
                    )
                    & (all_tuft_roots["is_terminal"])
                ].copy()
                axon_tree = KDTree(tuft_roots[["x", "y", "z"]].values)

                for sec in axon.iter_sections():
                    if sec.parent is None:
                        continue
                    last_pt = sec.points[-1, COLS.XYZ]
                    dist, index = axon_tree.query(last_pt)
                    if dist > 1e-6:
                        logger.debug("Skip section %s with point %s since no tuft root was found near this location.", sec.id, last_pt)
                        continue
                    else:
                        tuft_root = tuft_roots.iloc[index]
                        logger.debug("Found tuft root for the section %s with point %s at a distance %s", sec.id, last_pt, dist)

                    if self.use_cluster_props:
                        # Use properties from clustered morphologies
                        axon_terminals = cluster_props_df.loc[
                            (
                                cluster_props_df["morph_file"].str.endswith(
                                    morph_file.name
                                )
                            )
                            & (cluster_props_df["axon_id"] == axon_id)
                        ].copy()
                        axon_terminals["dist"] = np.linalg.norm(
                            (
                                np.array(
                                    axon_terminals["common_ancestor_coords"].tolist()
                                )
                                - tuft_root[["x", "y", "z"]].values
                            ).astype(float),
                            axis=1,
                        )
                        axon_terminal = cluster_props_df.loc[
                            axon_terminals["dist"].idxmin()
                        ]
                        terminal_index = axon_terminal.name
                    else:
                        terminal_index = rng.choice(cluster_props_df.index)
                        terminal_data = {}

                        # Compute raw terminal properties
                        terminal_data["path_distance"] = sum(
                            [
                                section_length(i.points)
                                for i in sec.iupstream()
                            ]
                        )
                        terminal_data["radial_distance"] = np.linalg.norm(
                            axon.points[0, COLS.XYZ]
                            - sec.points[-1]
                        )

                        # Compute terminal properties from white matter recipe
                        import pdb
                        pdb.set_trace()
                        terminal_data["cluster_size"] = -1
                        terminal_data["cluster_orientation"] = None

                        axon_terminal = pd.Series(terminal_data)

                    # Choose a tuft according to these properties
                    size_prob = _exp(
                        cluster_props_df["cluster_size"]
                        - axon_terminal["cluster_size"],
                        self.size_sigma,
                        terminal_index,
                    )
                    distance_prob = _exp(
                        cluster_props_df[self.distance_variable]
                        - axon_terminal[self.distance_variable],
                        self.distance_sigma,
                        terminal_index,
                    )

                    prob = size_prob * distance_prob
                    if prob.sum() == 0:
                        prob.loc[terminal_index] = 1
                    else:
                        prob /= prob.sum()

                    print("NEW:", morph_file, tuft_root["id"], axon_terminal["cluster_size"], axon_terminal[self.distance_variable], prob.idxmax(), prob.max())

                    chosen_index = rng.choice(cluster_props_df.index, p=prob)
                    logger.info(
                        (
                            "Chosen tuft for %s, axon %s and ancestor at %s from morph '%s', "
                            "axon %s and ancestor at %s"
                        ),
                        morph_file,
                        axon_id,
                        tuft_root[["x", "y", "z"]].tolist(),
                        *cluster_props_df.loc[
                            chosen_index,
                            ["morph_file", "axon_id", "common_ancestor_coords"]
                        ].tolist(),
                    )

                    cluster_props_df.at[
                        terminal_index, "new_cluster_barcode"
                    ] = cluster_props_df.at[chosen_index, "cluster_barcode"]
                    tuft_props.append(
                        [
                            str(raw_morph_file),
                            axon_id,
                            tuft_root["id"],
                            tuft_root[["x", "y", "z"]].tolist(),
                            axon_terminal["path_distance"],
                            axon_terminal["radial_distance"],
                            axon_terminal["cluster_size"],
                            axon_terminal["cluster_orientation"],
                            cluster_props_df.at[chosen_index, "cluster_barcode"],
                        ]
                    )

        import pdb
        pdb.set_trace()

        tuft_props_df = pd.DataFrame(
            tuft_props,
            columns=[
                "morph_file",
                "axon_id",
                "common_ancestor_id",
                "common_ancestor_coords",
                "path_distance",
                "radial_distance",
                "cluster_size",
                "cluster_orientation",
                "new_cluster_barcode",

            ],
        )
        tuft_props_df.sort_values(["morph_file", "axon_id", "common_ancestor_id"], inplace=True)

        with self.output().open(mode="w") as f:
            json.dump(tuft_props_df.to_dict("records"), f, indent=4)

    def output(self):
        return TuftsOutputLocalTarget("tuft_terminals.csv", create_parent=True)
