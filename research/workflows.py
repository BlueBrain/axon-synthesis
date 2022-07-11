"""The main workflows.

These workflows should be run with the following command:
python -m luigi --local-scheduler --module workflows <workflow>

TODO: Make a real package and update imports
"""
import luigi
import luigi_tools
from morphology_processing_workflow.tasks.workflows import Curate

from create_dataset import CreateDatasetForRepair
from PCSF.create_graph import CreateGraph
from PCSF.steiner_tree import SteinerTree
from PCSF.plot_steiner import PlotSolutions


class GeneralConfig(luigi.Config):
    output_dir = luigi.Parameter(description="The directory in which all the results will be exported", default=None)


luigi_tools.target.OutputLocalTarget.set_default_prefix(GeneralConfig().output_dir)


class DiscoverRawData(luigi_tools.task.WorkflowWrapperTask):
    """This is the first workflow: curate and plot the raw data."""

    def requires(self):
        dataset = CreateDatasetForRepair()
        return [dataset, Curate(dataset_df=dataset.output().path)]


class PrepareSteinerData(luigi_tools.task.WorkflowWrapperTask):
    """This workflow prepares the data used for Steiner Tree computation."""

    def requires(self):
        return CreateGraph()


class ComputeSteiner(luigi_tools.task.WorkflowWrapperTask):
    """This workflow prepares the data used for Steiner Tree computation."""

    def requires(self):
        return SteinerTree()


class PlotSteiner(luigi_tools.task.WorkflowWrapperTask):
    """This workflow prepares the data used for Steiner Tree computation."""

    def requires(self):
        return PlotSolutions()
