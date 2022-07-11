"""Compute and plot some statistics."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import luigi
import luigi_tools
import numpy as np
import neurom as nm
from matplotlib.backends.backend_pdf import PdfPages

from create_dataset import RepairDataset


class StatisticsOutputLocalTarget(luigi_tools.target.OutputLocalTarget):
    __prefix = Path("statistics")


class ComputeStatistics(luigi_tools.task.WorkflowTask):
    morph_dir = luigi.Parameter(
        description="Folder containing the input morphologies.",
        default=None,
    )
    output_dataset = luigi.Parameter(description="Output dataset file", default="statistics.json")

    def requires(self):
        if self.morph_dir is None:
            return RepairDataset()
        return None

    def run(self):
        morph_dir = Path(self.morph_dir or self.input().pathlib_path)
        output_file = self.output().pathlib_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        pop = nm.core.Population([f for f in morph_dir.iterdir()])

        # Statistics we want to check
        section_tortuosity = np.array(nm.get("section_tortuosity", pop)).tolist()
        section_radial_distances = np.array(nm.get("section_radial_distances", pop)).tolist()
        terminal_path_lengths = np.array(nm.get("terminal_path_lengths", pop)).tolist()
        section_term_radial_distances = np.array(nm.get("section_term_radial_distances", pop)).tolist()
        neurite_tortuosity = (
            np.array(terminal_path_lengths) /
            np.array(section_term_radial_distances)
        ).tolist()
        local_bifurcation_angles = np.array(nm.get("local_bifurcation_angles", pop)).tolist()
        remote_bifurcation_angles = np.array(nm.get("remote_bifurcation_angles", pop)).tolist()

        result = {
            "section_tortuosity": section_tortuosity,
            "neurite_tortuosity": neurite_tortuosity,
            "local_bifurcation_angles": local_bifurcation_angles,
            "remote_bifurcation_angles": remote_bifurcation_angles,
            "section_radial_distances": section_radial_distances,
            "section_term_radial_distances": section_term_radial_distances,
            "terminal_path_lengths": terminal_path_lengths,
        }

        with open(self.output().path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        return result


    def output(self):
        return StatisticsOutputLocalTarget(self.output_dataset)


class PlotStatistics(luigi_tools.task.WorkflowTask):
    output_dir = luigi.Parameter(description="Output directory", default="figures")
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
