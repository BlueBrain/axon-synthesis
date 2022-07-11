"""Update the properties of the tufts that will be generated later."""
import sys
import json
import logging
from pathlib import Path

import luigi
import luigi_tools
import numpy as np
import pandas as pd

from PCSF.clustering import ClusterTerminals

logger = logging.getLogger(__name__)


class TuftsOutputLocalTarget(luigi_tools.target.OutputLocalTarget):
    __prefix = Path("tufts")


def _exp(values, sigma, default_ind):
    if sigma != 0:
        return 1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-np.power(values, 2) / (2.0 * sigma**2))
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
        default="path_distance"
    )
    distance_sigma = luigi.NumericalParameter(
        description="The sigma value used to select the barcode along the distance axis.",
        var_type=float,
        default=0,
        min_value=0,
        max_value=sys.float_info.max,
    )

    def requires(self):
        return ClusterTerminals()

    def run(self):

        if self.size_sigma == 0 or self.distance_sigma == 0:
            self.size_sigma = 0
            self.distance_sigma = 0

        with self.input()["tuft_properties"].open() as f:
            cluster_props_df = pd.DataFrame.from_records(json.load(f))

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

                chosen_index = np.random.choice(cluster_props_df.index, p=prob)
                cluster_props_df.at[terminal_index, "new_cluster_barcode"] = cluster_props_df.at[chosen_index, "cluster_barcode"]

        with self.output().open(mode="w") as f:
            json.dump(cluster_props_df.to_dict("records"), f, indent=4)

    def output(self):
        return TuftsOutputLocalTarget("tuft_terminals.csv", create_parent=True)
