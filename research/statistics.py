"""Compute and plot some statistics."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import luigi
import luigi_tools
import numpy as np
import neurom as nm
from matplotlib.backends.backend_pdf import PdfPages
from luigi_tools.parameter import PathParameter

from create_dataset import RepairDataset


class StatisticsOutputLocalTarget(luigi_tools.target.OutputLocalTarget):
    __prefix = Path("statistics")


def _np_cast(array):
    return np.array(array).tolist()


def population_statistics(pop):
    # Statistics we want to check
    section_tortuosity = _np_cast(nm.get("section_tortuosity", pop))
    section_radial_distances = _np_cast(nm.get("section_radial_distances", pop))
    terminal_path_lengths = _np_cast(nm.get("terminal_path_lengths", pop))
    section_term_radial_distances = _np_cast(nm.get("section_term_radial_distances", pop))
    neurite_tortuosity = (
        np.array(terminal_path_lengths)
        / np.array(section_term_radial_distances)
    ).tolist()
    local_bifurcation_angles = _np_cast(nm.get("local_bifurcation_angles", pop))
    remote_bifurcation_angles = _np_cast(nm.get("remote_bifurcation_angles", pop))

    result = {
        "section_tortuosity": section_tortuosity,
        "neurite_tortuosity": neurite_tortuosity,
        "local_bifurcation_angles": local_bifurcation_angles,
        "remote_bifurcation_angles": remote_bifurcation_angles,
        "section_radial_distances": section_radial_distances,
        "section_term_radial_distances": section_term_radial_distances,
        "terminal_path_lengths": terminal_path_lengths,
    }

    nm.morph_stats.extract_stats(
        pop,
        {
            "neurite": {
                "total_length_per_neurite": {
                    "modes": ["sum"],
                }
            },
            "neurite_type": ["axon"],
        }
    )

    return result


class ComputeStatistics(luigi_tools.task.WorkflowTask):
    morph_dir = PathParameter(
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
        output_file = self.output().pathlib_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        pop = nm.core.Population([f for f in morph_dir.iterdir()])

        result = population_statistics(pop)

        with open(self.output().path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        return result

    def output(self):
        return StatisticsOutputLocalTarget(self.output_dataset)


class PlotStatistics(luigi_tools.task.WorkflowTask):
    output_dir = PathParameter(description="Output directory", default="figures")
    nb_bins = luigi.IntParameter(description="The number of bins used for histograms", default=20)

    def requires(self):
        return ComputeStatistics()

    def run(self):
        self.output().pathlib_path.mkdir(parents=True, exist_ok=True)

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
        return StatisticsOutputLocalTarget(self.output_dir)


class CompareStatistics(luigi_tools.task.WorkflowTask):
    output_dir = PathParameter(description="Output directory", default="figures")
    nb_bins = luigi.IntParameter(description="The number of bins used for histograms", default=20)
    morph_dir_biological = PathParameter(
        description="Folder containing the biological morphologies.",
        default=None,
    )
    morph_dir_generated = PathParameter(
        description="Folder containing the generated morphologies.",
    )

    def requires(self):
        return {
            "bio": ComputeStatistics(
                morph_dir=self.morph_dir_biological,
                output_dataset=self.output_dir / "bio_stats",
            ),
            "gen": ComputeStatistics(
                morph_dir=self.morph_dir_generated,
                output_dataset=self.output_dir / "gen_stats",
            ),
        }

    def run(self):
        self.output().pathlib_path.mkdir(parents=True, exist_ok=True)

        with open(self.input()["bio"].path, encoding="utf-8") as f:
            bio_statistics = json.load(f)
        with open(self.input()["gen"].path, encoding="utf-8") as f:
            gen_statistics = json.load(f)

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
        return StatisticsOutputLocalTarget(self.output_dir)
