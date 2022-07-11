"""The main workflows.

These workflows should be run with the following command:
python -m luigi --local-scheduler --module workflows <workflow>

TODO: Make a real package and update imports
"""
import luigi
import luigi_tools
from morphology_processing_workflow.tasks.workflows import Curate

from create_dataset import CreateDatasetForRepair
from statistics import PlotStatistics
from statistics import StatisticsOutputLocalTarget
from PCSF.create_graph import CreateGraph
from PCSF.plot_steiner import PlotSolutions
from PCSF.steiner_morphologies import SteinerMorphologies


class GeneralConfig(luigi.Config):
    output_dir = luigi.Parameter(
        description="The directory in which all the results will be exported",
        default=None,
    )
    statistics_dir = luigi.Parameter(
        description="The directory in which all the statistics will be exported",
        default="raw_statistics",
    )


luigi_tools.target.OutputLocalTarget.set_default_prefix(GeneralConfig().output_dir)
StatisticsOutputLocalTarget.set_default_prefix(GeneralConfig().statistics_dir)


class DiscoverRawData(luigi_tools.task.WorkflowWrapperTask):
    """This is the first workflow: curate and plot the raw data."""

    def requires(self):
        dataset = CreateDatasetForRepair()
        return [dataset, Curate(dataset_df=dataset.output().path)]


class ExploreStatistics(luigi_tools.task.WorkflowWrapperTask):
    """This workflow creates and plots statistics about data."""

    def requires(self):
        return PlotStatistics()


class PrepareSteinerData(luigi_tools.task.WorkflowWrapperTask):
    """This workflow prepares the data used for Steiner Tree computation."""

    def requires(self):
        return CreateGraph()


class ComputeSteiner(luigi_tools.task.WorkflowWrapperTask):
    """This workflow performs the Steiner Tree computation."""

    def requires(self):
        return SteinerMorphologies()


class PlotSteiner(luigi_tools.task.WorkflowWrapperTask):
    """This workflow plots the results from the Steiner Tree computation."""

    def requires(self):
        return PlotSolutions()
