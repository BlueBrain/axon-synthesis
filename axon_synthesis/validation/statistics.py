"""Compute and plot some statistics."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool.utils import iter_morphology_files
from neurom import load_morphologies
from neurom.apps.morph_stats import extract_dataframe

# import json
# from collections import defaultdict
# from itertools import chain

# import attr
# import luigi
# import luigi_tools
# import neurom as nm
# import neurots.extract_input
# from data_validation_framework.target import TaggedOutputLocalTarget
# from luigi.parameter import OptionalPathParameter
# from luigi.parameter import PathParameter
# from neurom import load_morphology
# from neurom.core.types import NeuriteType

# from axon_synthesis.add_tufts import AddTufts
# from axon_synthesis.create_dataset import RepairDataset
# from axon_synthesis.main_trunk.clustering import ClusterTerminals

# # from maint_trunk.clustering import ClusterTerminals
# # from maint_trunk.steiner_morphologies import SteinerMorphologies

logger = logging.getLogger(__name__)


# def _np_cast(array, do_sum=False):
#     if do_sum:
#         return np.array(array).sum()
#     return np.array(array).tolist()


def default_config():
    """Create the default config used to compute the scores."""
    return {
        "neurite": {
            "number_of_bifurcations": ["sum"],
            "number_of_sections_per_neurite": ["mean", "sum"],
            "number_of_leaves": ["sum"],
            "partition_asymmetry": ["mean", "sum"],
            "partition_asymmetry_length": ["mean", "sum"],
            "remote_bifurcation_angles": ["mean"],
            "section_bif_branch_orders": ["max", "mean"],
            "section_bif_lengths": ["min", "max", "mean"],
            "section_bif_radial_distances": ["max", "mean"],
            "section_branch_orders": ["max", "mean"],
            "section_lengths": ["min", "max", "mean", "sum"],
            "section_path_distances": ["min", "max", "mean"],
            "section_radial_distances": ["min", "max", "mean"],
            "section_strahler_orders": ["mean"],
            "section_term_branch_orders": ["max", "mean"],
            "section_term_lengths": ["min", "max", "mean"],
            "section_term_radial_distances": ["max", "mean"],
            "section_tortuosity": ["min", "max", "mean"],
            "total_length_per_neurite": ["mean"],
        },
        "neurite_type": [
            "AXON",
        ],
    }


def relative_score(data1, data2):
    """Get the score.

    Args:
        data1 (list): the first data set.
        data2 (list): the second data set.
    """
    return (data2 - data1) / data1


def get_scores(df1, df2):
    """Return scores between two data sets.

    Args:
        df1 (pandas.DataFrame): the first data set.
        df2 (pandas.DataFrame): the second data set.

    Returns:
        The list of feature scores.
    """
    scores = []
    score_names = []
    key_names = {
        "basal_dendrite": "Basal",
        "apical_dendrite": "Apical",
        "axon": "Axon",
    }
    for neurite_type, neurite_name in key_names.items():
        if neurite_type in df1.columns and neurite_type in df2.columns:
            _df1 = df1[neurite_type]
            _df2 = df2[neurite_type]
            for k in _df1.columns:
                data1 = _df1[k].to_numpy()[0]
                data2 = _df2[k].to_numpy()[0]
                score_name = neurite_name + " " + k.replace("_", " ")
                score_names.append(score_name)
                if not np.isnan(data1) and not np.isnan(data2):
                    sc1 = relative_score(data1, data2)
                    if not np.isnan(sc1):
                        scores.append(sc1)
                    else:
                        scores.append(0.0)
                else:
                    scores.append(np.nan)
                logger.debug(
                    "Score name: %s ; Biological value: %s ; Generated value: %s ; Score: %s",
                    score_name,
                    data1,
                    data2,
                    scores[-1],
                )

    return score_names, scores


def compute_scores(ref, test, config):
    """Compute scores of a test population against a reference population.

    Args:
        ref (tuple(str, list)): the reference data.
        test (tuple(str, list)): the test data.
        config (dict): the configuration used to compute the scores.

    Returns:
        The scores and the feature list.
    """
    ref_mtype, ref_files = ref
    test_mtype, test_files = test
    if ref_mtype != test_mtype:
        msg = "The mtypes of ref and test files must be the same."
        raise AssertionError(msg)

    ref_pop = load_morphologies(ref_files)
    test_pop = load_morphologies(test_files)

    ref_features = extract_dataframe(ref_pop, config)
    test_features = extract_dataframe(test_pop, config)

    logger.debug(ref_files)
    return get_scores(ref_features, test_features)


def _check_scores(scores, keys) -> int:
    n_scores = len(keys[0])
    for k, s in zip(keys[1:], scores):
        if keys[0] != k:
            msg = "Score names must all be the same for each feature."
            raise AssertionError(msg)
        if len(k) != n_scores:
            msg = "The number of keys must be the same for each morphology."
            raise AssertionError(msg)
        if len(s) != n_scores:
            msg = "The number of scores must be the same for each morphology."
            raise AssertionError(msg)
    return n_scores


def plot_score_matrix(
    ref_morphs_dir,
    test_morphs_dir,
    output_path,
    config,
    dpi=100,
):
    """Plot score matrix for a test population against a reference population."""
    # pylint: disable=too-many-locals
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build the file list
    if not isinstance(ref_morphs_dir, list):
        ref_file_lists = [(i.name, [i]) for i in sorted(iter_morphology_files(ref_morphs_dir))]
    else:
        ref_file_lists = ref_morphs_dir
    if not isinstance(test_morphs_dir, list):
        test_file_lists = [(i.name, [i]) for i in sorted(iter_morphology_files(test_morphs_dir))]
    else:
        test_file_lists = test_morphs_dir
    size = len(ref_file_lists)
    names = [i[0] for i in ref_file_lists]
    if size != len(test_file_lists):
        msg = (
            f"The number of ref files ({size}) must be equal to the res files "
            f"({len(test_file_lists)})"
        )
        raise AssertionError(msg)

    # Compute scores
    scores = []
    keys = []
    for ref_files, test_files in zip(ref_file_lists, test_file_lists):
        key_name, score = compute_scores(ref_files, test_files, config)
        keys.append(key_name)
        scores.append(score)

    n_scores = _check_scores(scores, keys)

    # Plot statistics
    with PdfPages(output_path) as pdf:
        # Compute subplot ratios and figure size
        height_ratios = [7, (1 + n_scores)]
        fig_width = size
        fig_height = sum(height_ratios) * 0.3

        hspace = 0.625 / fig_height
        wspace = 0.2 / fig_width

        cbar_ratio = 0.4 / fig_width

        # Create the figure and the subplots
        fig, ((a0, a2), (a1, a3)) = plt.subplots(  # type: ignore[misc]
            2,
            2,
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": [1 - cbar_ratio, cbar_ratio],
                "hspace": hspace,
                "wspace": wspace,
            },
            figsize=(fig_width, fig_height),
        )

        # Plot score errors
        clipped_scores = np.clip(np.abs(scores), 0, 1)
        a0.errorbar(
            np.arange(size),
            np.nanmean(clipped_scores, axis=1),
            yerr=np.nanstd(clipped_scores, axis=1),
            color="black",
            label="Synthesized",
        )
        a0.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
        a0.xaxis.set_tick_params(rotation=45)
        a0.set_xticks(np.arange(size))
        a0.set_xticklabels(names)

        a0.set_xlim([a0.xaxis.get_ticklocs().min() - 0.5, a0.xaxis.get_ticklocs().max() + 0.5])
        a0.set_ylim([-0.1, 1.1])

        # Plot score heatmap
        scores_transpose = np.transpose(scores)
        scores_df = pd.DataFrame(scores_transpose, index=keys[0], columns=names)

        g = sns.heatmap(
            scores_df,
            vmin=-1,
            vmax=1,
            mask=np.isnan(scores_transpose),
            ax=a1,
            cmap=cm.seismic,  # type: ignore[attr-defined]
            cbar_ax=a3,
        )

        g.xaxis.set_tick_params(rotation=45)
        g.set_facecolor("xkcd:black")

        # Remove upper right subplot
        a2.remove()

        # Export the figure
        try:
            logging.disable(logging.CRITICAL)
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
        finally:
            logging.disable(0)
        plt.close(fig)


# @attr.s(auto_attribs=True)
# class Statistics:
#     """The object to store basic statistics."""

#     min: np.number
#     max: np.number
#     mean: np.number
#     std: np.number

#     def to_list(self):
#         """Return a list with the min, max, mean and std stat values."""
#         return [
#             self.min,
#             self.max,
#             self.mean,
#             self.std,
#         ]

#     @staticmethod
#     def gather_stats(stats, prefix=""):
#         """Gather stats from a list of Statistics instances."""
#         values = np.array([i.to_list() for i in stats])
#         return {
#             f"{prefix}min": values[:, 0].tolist(),
#             f"{prefix}max": values[:, 1].tolist(),
#             f"{prefix}mean": values[:, 2].tolist(),
#             f"{prefix}std": values[:, 3].tolist(),
#         }


# def to_stats(values):
#     """Compute statistics of the given list of values."""
#     if isinstance(values[0], list):
#         values = np.array(list(chain.from_iterable(values)))
#     else:
#         values = np.array(values)
#     return Statistics(
#         values.min(),
#         values.max(),
#         values.mean(),
#         values.std(),
#     )


# def population_statistics(pop, neurite_type=NeuriteType.axon):
#     """Compute statistics for the given population."""
#     # Statistics we want to check
#     # section_tortuosity = []
#     # section_radial_distances = []
#     terminal_path_lengths = []
#     # section_term_radial_distances = []
#     # neurite_tortuosity = []
#     local_bifurcation_angles = []
#     remote_bifurcation_angles = []
#     total_axon_length = []
#     radial_moment_0 = []
#     # radial_moment_1 = []
#     radial_moment_2 = []
#     # normalized_radial_moment_0 = []
#     # normalized_radial_moment_1 = []
#     normalized_radial_moment_2 = []
#     for neuron in pop:
#         logger.info(neuron)
#         # import pdb
#         # pdb.set_trace()
#         # neurite_tortuosity.append(
#         #     _np_cast(nm.get("tortuosity_per_neurite", neuron, neurite_type=neurite_type))
#         # )
#         # neurite_tortuosity.append(
#         #     _np_cast(nm.get("tortuosity", neuron, neurite_type=neurite_type))
#         # )
#         # section_tortuosity.append(
#         #     _np_cast(nm.get("section_tortuosity", neuron, neurite_type=neurite_type))
#         # )
#         # section_radial_distances.append(
#         #     _np_cast(nm.get("section_radial_distances", neuron, neurite_type=neurite_type))
#         # )
#         # section_term_radial_distances.append(
#         #     _np_cast(nm.get("section_term_radial_distances", neuron, neurite_type=neurite_type))
#         # )
#         # terminal_path_lengths.append(
#         #     to_stats(nm.get("terminal_path_lengths", neuron, neurite_type=neurite_type))
#         # )
#         # neurite_tortuosity.append((
#         #     np.array(terminal_path_lengths[-1])
#         #     / np.array(section_term_radial_distances[-1])
#         # ).tolist())
#         local_bifurcation_angles.append(
#             to_stats(nm.get("local_bifurcation_angles", neuron, neurite_type=neurite_type)),
#         )
#         remote_bifurcation_angles.append(
#             to_stats(nm.get("remote_bifurcation_angles", neuron, neurite_type=neurite_type)),
#         )
#         total_axon_length.append(
#             sum(nm.get("total_length_per_neurite", neuron, neurite_type=neurite_type)),
#         )
#         radial_moments = {
#             i: nm.get(
#                 "radial_moment",
#                 neuron,
#                 neurite_type=neurite_type,
#                 order=i,
#                 use_radius=False,
#             )
#             for i in [0, 2]
#         }
#         normalized_moments = {order: m / radial_moments[0] for order, m in radial_moments.items()}
#         radial_moment_0.append(radial_moments[0])
#         # radial_moment_1.append(radial_moments[1])
#         radial_moment_2.append(radial_moments[2])
#         # normalized_radial_moment_1.append(normalized_moments[1])
#         normalized_radial_moment_2.append(normalized_moments[2])

#     result = {
#         # "section_tortuosity": section_tortuosity,
#         # "neurite_tortuosity": neurite_tortuosity,
#         # "local_bifurcation_angles_stats": to_stats(local_bifurcation_angles),
#         # "remote_bifurcation_angles_stats": to_stats(remote_bifurcation_angles),
#         # "section_radial_distances": section_radial_distances,
#         # "section_term_radial_distances": to_stats(section_term_radial_distances),
#         # "terminal_path_lengths": to_stats(terminal_path_lengths),
#         "total_axon_length": _np_cast(total_axon_length),
#         "radial_moment_0": radial_moment_0,
#         # "radial_moment_1": radial_moment_1,
#         "radial_moment_2": radial_moment_2,
#         # "normalized_radial_moment_1": normalized_radial_moment_1,
#         "normalized_radial_moment_2": normalized_radial_moment_2,
#     }

#     for stat_name, stats in {
#         "local_bifurcation_angles": local_bifurcation_angles,
#         "remote_bifurcation_angles_stats": remote_bifurcation_angles,
#         "terminal_path_lengths": terminal_path_lengths,
#     }.items():
#         result.update(Statistics.gather_stats(stats, prefix=f"{stat_name}_"))

#     return result


# class ComputeStatistics(luigi_tools.task.WorkflowTask):
#     """Task to compute the statistics of the given morphologies."""

#     morph_dir = OptionalPathParameter(
#         description="Folder containing the input morphologies.",
#         default=None,
#     )
#     output_dataset = luigi.Parameter(description="Output dataset file", default="statistics.json")

#     def requires(self):
#         if self.morph_dir is None:
#             return RepairDataset()
#         return None

#     def run(self):
#         morph_dir = self.morph_dir or self.input().pathlib_path

#         pop = nm.core.Population(sorted(f for f in morph_dir.iterdir()))

#         result = population_statistics(pop)

#         with open(self.output().path, "w", encoding="utf-8") as f:
#             json.dump(result, f)

#         return result

#     def output(self):
#         return StatisticsOutputLocalTarget(self.output_dataset, create_parent=True)


# class PlotStatistics(luigi_tools.task.WorkflowTask):
#     """Task to plot the statistics of the given morphologies."""

#     output_dir = PathParameter(description="Output directory", default="figures")
#     nb_bins = luigi.IntParameter(description="The number of bins used for histograms", default=20)

#     def requires(self):
#         return ComputeStatistics()

#     def run(self):
#         with open(self.input().path, encoding="utf-8") as f:
#             statistics = json.load(f)

#         with PdfPages(self.output().pathlib_path / "input_statistics.pdf") as pdf:
#             for key, values in statistics.items():
#                 fig = plt.figure()
#                 ax = fig.gca()

#                 ax.hist(values, bins=self.nb_bins, density=True)

#                 ax.set_xlabel(key)
#                 ax.set_ylabel("Density")
#                 fig.suptitle(f"Input {key}")
#                 pdf.savefig()
#                 plt.close(fig)

#     def output(self):
#         return StatisticsOutputLocalTarget(self.output_dir, create_parent=True)


# class CompareStatistics(luigi_tools.task.WorkflowTask):
#     """Task to compare 2 sets of statistics."""

#     output_dir = PathParameter(description="Output directory", default="compare_statistics")
#     nb_bins = luigi.IntParameter(description="The number of bins used for histograms", default=20)
#     morph_dir_biological = OptionalPathParameter(
#         description="Folder containing the biological morphologies.",
#         default=None,
#     )
#     morph_dir_generated = OptionalPathParameter(
#         description="Folder containing the generated morphologies.",
#         default=None,
#     )

#     def requires(self):
#         return {
#             "bio": RepairDataset(),
#             "gen": AddTufts(),
#         }
#         # bio_kwargs = {"output_dataset": self.output_dir / "bio_stats.json"}
#         # if self.morph_dir_biological is not None:
#         #     bio_kwargs["morph_dir"] = self.morph_dir_biological
#         # else:
#         #     bio_kwargs["morph_dir"] = ClusterTerminals().output()["morphologies"].path
#         #     # bio_kwargs["morph_dir"] = RepairDataset().output().path

#         # gen_kwargs = {"output_dataset": self.output_dir / "gen_stats.json"}
#         # if self.morph_dir_generated is not None:
#         #     gen_kwargs["morph_dir"] = self.morph_dir_generated
#         # else:
#         #     gen_kwargs["morph_dir"] = SteinerMorphologies().output().path

#         # return {
#         #     "bio": ComputeStatistics(**bio_kwargs),
#         #     "gen": ComputeStatistics(**gen_kwargs),
#         # }

#     def run(self):
#         self.output().pathlib_path.mkdir(parents=True, exist_ok=True)

#         # import pdb
#         # pdb.set_trace()
#         plot_score_matrix(
#             self.input()["bio"].path,
#             self.input()["gen"]["morphologies_asc"].path,
#             self.output().pathlib_path / "score_matrix.pdf",
#             default_config(),
#         )

#         # with open(self.input()["bio"].path, encoding="utf-8") as f:
#         #     bio_statistics = json.load(f)
#         # with open(self.input()["gen"].path, encoding="utf-8") as f:
#         #     gen_statistics = json.load(f)

#         # with PdfPages(self.output().pathlib_path / "compare_statistics.pdf") as pdf:

#         #     for key, bio_values in bio_statistics.items():

#         #         gen_values = gen_statistics.get(key)

#         #         if gen_values is None:
#         #             logger.error(f"'{key}' was not found in {self.input()['gen'].path}")

#         #         fig = plt.figure()
#         #         ax = fig.gca()

#         #         gen_values = np.array(gen_values)
#         #         bio_values = np.array(bio_values)

#         #         values = gen_values / bio_values

#         #         ax.hist(values, bins=self.nb_bins, density=True)

#         #         ax.set_xlabel(f"Relative deviation for {key}")
#         #         ax.set_ylabel("Density")
#         #         fig.suptitle(f"Relative deviation for {key}")
#         #         pdf.savefig()
#         #         plt.close(fig)

#     def output(self):
#         return StatisticsOutputLocalTarget(self.output_dir)


# class CheckTuftStatistics(luigi_tools.task.WorkflowTask):
#     """Task to compute the statistics of the given tuft morphologies and check them."""

#     morph_dir = OptionalPathParameter(
#         description="Folder containing the input morphologies.",
#         default=None,
#     )
#     output_dataset = luigi.Parameter(description="Output dataset file", default="statistics.json")

#     def requires(self):
#         if self.morph_dir is None:
#             return ClusterTerminals()
#         return None

#     def find_morph_from_pt(self, tufts_by_region, x, y, z):
#         """Find a morphology from a given point."""
#         for region_name, tuft_files in tufts_by_region.items():
#             for i in tuft_files:
#                 morph = load_morphology(i)
#                 if (
#                     (morph.points[:, 0] >= x)
#                     & (morph.points[:, 0] < x + 1)
#                     & (morph.points[:, 1] >= y)
#                     & (morph.points[:, 1] < y + 1)
#                     & (morph.points[:, 2] >= z)
#                     & (morph.points[:, 2] < z + 1)
#                 ).any():
#                     print("File for:", x, y, z, "=>", region_name, i)

#     def run(self):
#         self.output().pathlib_path.mkdir(parents=True, exist_ok=True)
#         morph_dir = self.morph_dir or self.input()["tuft_morphologies"].pathlib_path
#         csv_files = sorted(filter(lambda x: x.suffix == ".csv", morph_dir.iterdir()))
#         tufts_by_region = defaultdict(list)
#         all_stats = {}
#         N = 0
#         for csv_file in csv_files:
#             df = pd.read_csv(csv_file)
#             df.drop(df.loc[df["tuft_morph_path"].isnull()].index, inplace=True)
#             for (region_name, _), tuft_files in df.groupby(["region_acronym", "cluster_id"]):
#                 tufts_by_region[region_name].append(tuft_files["tuft_morph_path"].tolist()[0])
#                 N += len(tuft_files["tuft_morph_path"].tolist())
#             if N > 100:
#                 break

#         for region_name, tuft_files in tufts_by_region.items():
#             all_stats[region_name] = neurots.extract_input.distributions(tuft_files)

#         self.find_morph_from_pt(tufts_by_region, 4435, 4137, 7273)
#         self.find_morph_from_pt(tufts_by_region, 4458, 5309, 7710)

#     def output(self):
#         return StatisticsOutputLocalTarget("tufts", create_parent=True)
