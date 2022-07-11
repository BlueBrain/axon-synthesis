"""Compute and plot some statistics."""
import attr
import json
import logging
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import luigi
import luigi_tools
import numpy as np
import neurom as nm
from matplotlib.backends.backend_pdf import PdfPages
from luigi_tools.parameter import PathParameter
from luigi_tools.parameter import OptionalPathParameter
from neurom.core.types import NeuriteType

from create_dataset import RepairDataset
from PCSF.clustering import ClusterTerminals
from PCSF.steiner_morphologies import SteinerMorphologies

logger = logging.getLogger(__name__)


class StatisticsOutputLocalTarget(luigi_tools.target.OutputLocalTarget):
    __prefix = Path("statistics")


def _np_cast(array, do_sum=False):
    if do_sum:
        return np.array(array).sum()
    return np.array(array).tolist()


@attr.s(auto_attribs=True)
class Statistics:
    """The object to store basic statistics."""

    min: np.number
    max: np.number
    mean: np.number
    std: np.number

    def to_list(self):
        return [
            self.min,
            self.max,
            self.mean,
            self.std,
        ]

    @staticmethod
    def gather_stats(stats, prefix=""):
        values = np.array([i.to_list() for i in stats])
        return {
            f"{prefix}min": values[:, 0].tolist(),
            f"{prefix}max": values[:, 1].tolist(),
            f"{prefix}mean": values[:, 2].tolist(),
            f"{prefix}std": values[:, 3].tolist(),
        }


def to_stats(values):
    if isinstance(values[0], list):
        values = np.array(list(chain.from_iterable(values)))
    else:
        values = np.array(values)
    return Statistics(
        values.min(),
        values.max(),
        values.mean(),
        values.std(),
    )


def population_statistics(pop, neurite_type=NeuriteType.axon):
    # Statistics we want to check
    section_tortuosity = []
    section_radial_distances = []
    terminal_path_lengths = []
    section_term_radial_distances = []
    neurite_tortuosity = []
    local_bifurcation_angles = []
    remote_bifurcation_angles = []
    total_axon_length = []
    radial_moment_0 = []
    radial_moment_1 = []
    radial_moment_2 = []
    # normalized_radial_moment_0 = []
    normalized_radial_moment_1 = []
    normalized_radial_moment_2 = []
    for neuron in pop:
        logger.info(neuron)
        # import pdb
        # pdb.set_trace()
        # neurite_tortuosity.append(_np_cast(nm.get("tortuosity_per_neurite", neuron, neurite_type=neurite_type)))
        # neurite_tortuosity.append(_np_cast(nm.get("tortuosity", neuron, neurite_type=neurite_type)))
        # section_tortuosity.append(_np_cast(nm.get("section_tortuosity", neuron, neurite_type=neurite_type)))
        # section_radial_distances.append(_np_cast(nm.get("section_radial_distances", neuron, neurite_type=neurite_type)))
        # section_term_radial_distances.append(_np_cast(nm.get("section_term_radial_distances", neuron, neurite_type=neurite_type)))
        terminal_path_lengths.append(
            to_stats(nm.get("terminal_path_lengths", neuron, neurite_type=neurite_type))
        )
        # neurite_tortuosity.append((
        #     np.array(terminal_path_lengths[-1])
        #     / np.array(section_term_radial_distances[-1])
        # ).tolist())
        local_bifurcation_angles.append(
            to_stats(nm.get("local_bifurcation_angles", neuron, neurite_type=neurite_type))
        )
        remote_bifurcation_angles.append(
            to_stats(nm.get("remote_bifurcation_angles", neuron, neurite_type=neurite_type))
        )
        total_axon_length.append(
            sum(nm.get("total_length_per_neurite", neuron, neurite_type=neurite_type))
        )
        radial_moments = {
            i: nm.get("radial_moment", neuron, neurite_type=neurite_type, order=i, use_radius=False)
            for i in [0, 2]
        }
        normalized_moments = {order: m / radial_moments[0] for order, m in radial_moments.items()}
        radial_moment_0.append(radial_moments[0])
        # radial_moment_1.append(radial_moments[1])
        radial_moment_2.append(radial_moments[2])
        # normalized_radial_moment_1.append(normalized_moments[1])
        normalized_radial_moment_2.append(normalized_moments[2])

    result = {
        # "section_tortuosity": section_tortuosity,
        # "neurite_tortuosity": neurite_tortuosity,
        # "local_bifurcation_angles_stats": to_stats(local_bifurcation_angles),
        # "remote_bifurcation_angles_stats": to_stats(remote_bifurcation_angles),
        # "section_radial_distances": section_radial_distances,
        # "section_term_radial_distances": to_stats(section_term_radial_distances),
        # "terminal_path_lengths": to_stats(terminal_path_lengths),
        "total_axon_length": _np_cast(total_axon_length),
        "radial_moment_0": radial_moment_0,
        # "radial_moment_1": radial_moment_1,
        "radial_moment_2": radial_moment_2,
        # "normalized_radial_moment_1": normalized_radial_moment_1,
        "normalized_radial_moment_2": normalized_radial_moment_2,
    }

    for stat_name, stats in {
        "local_bifurcation_angles": local_bifurcation_angles,
        "remote_bifurcation_angles_stats": remote_bifurcation_angles,
        "terminal_path_lengths": terminal_path_lengths,
    }.items():
        result.update(Statistics.gather_stats(stats, prefix=f"{stat_name}_"))

    return result


class ComputeStatistics(luigi_tools.task.WorkflowTask):
    morph_dir = OptionalPathParameter(
        description="Folder containing the input morphologies.",
        default=None,
    )
    output_dataset = luigi.Parameter(description="Output dataset file", default="statistics.json")

    def requires(self):
        if self.morph_dir is None:
            return RepairDataset()
        return None

    def run(self):
        morph_dir = self.morph_dir or self.input().pathlib_path

        pop = nm.core.Population(sorted([f for f in morph_dir.iterdir()]))

        result = population_statistics(pop)

        with open(self.output().path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        return result

    def output(self):
        return StatisticsOutputLocalTarget(self.output_dataset, create_parent=True)


class PlotStatistics(luigi_tools.task.WorkflowTask):
    output_dir = PathParameter(description="Output directory", default="figures")
    nb_bins = luigi.IntParameter(description="The number of bins used for histograms", default=20)

    def requires(self):
        return ComputeStatistics()

    def run(self):
        with open(self.input().path, encoding="utf-8") as f:
            statistics = json.load(f)

        with PdfPages(self.output().pathlib_path / "input_statistics.pdf") as pdf:

            for key, values in statistics.items():

                fig = plt.figure()
                ax = fig.gca()

                ax.hist(values, bins=self.nb_bins, density=True)

                ax.set_xlabel(key)
                ax.set_ylabel("Density")
                fig.suptitle(f"Input {key}")
                pdf.savefig()
                plt.close(fig)

    def output(self):
        return StatisticsOutputLocalTarget(self.output_dir, create=True)


class CompareStatistics(luigi_tools.task.WorkflowTask):
    output_dir = PathParameter(description="Output directory", default="compare_statistics")
    nb_bins = luigi.IntParameter(description="The number of bins used for histograms", default=20)
    morph_dir_biological = OptionalPathParameter(
        description="Folder containing the biological morphologies.",
        default=None,
    )
    morph_dir_generated = OptionalPathParameter(
        description="Folder containing the generated morphologies.",
        default=None,
    )

    def requires(self):
        bio_kwargs = {"output_dataset": self.output_dir / "bio_stats.json"}
        if self.morph_dir_biological is not None:
            bio_kwargs["morph_dir"] = self.morph_dir_biological
        else:
            bio_kwargs["morph_dir"] = ClusterTerminals().output()["morphologies"].path
            # bio_kwargs["morph_dir"] = RepairDataset().output().path

        gen_kwargs = {"output_dataset": self.output_dir / "gen_stats.json"}
        if self.morph_dir_generated is not None:
            gen_kwargs["morph_dir"] = self.morph_dir_generated
        else:
            gen_kwargs["morph_dir"] = SteinerMorphologies().output().path

        return {
            "bio": ComputeStatistics(**bio_kwargs),
            "gen": ComputeStatistics(**gen_kwargs),
        }

    def run(self):
        self.output().pathlib_path.mkdir(parents=True, exist_ok=True)

        with open(self.input()["bio"].path, encoding="utf-8") as f:
            bio_statistics = json.load(f)
        with open(self.input()["gen"].path, encoding="utf-8") as f:
            gen_statistics = json.load(f)

        with PdfPages(self.output().pathlib_path / "compare_statistics.pdf") as pdf:

            for key, bio_values in bio_statistics.items():

                gen_values = gen_statistics.get(key)

                if gen_values is None:
                    logger.error(f"'{key}' was not found in {self.input()['gen'].path}")

                fig = plt.figure()
                ax = fig.gca()

                gen_values = np.array(gen_values)
                bio_values = np.array(bio_values)

                values = gen_values / bio_values

                ax.hist(values, bins=self.nb_bins, density=True)

                ax.set_xlabel(f"Relative deviation for {key}")
                ax.set_ylabel("Density")
                fig.suptitle(f"Relative deviation for {key}")
                pdf.savefig()
                plt.close(fig)

    def output(self):
        return StatisticsOutputLocalTarget(self.output_dir)
