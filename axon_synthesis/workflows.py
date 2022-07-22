"""The main workflows.

These workflows should be run with the following command:
python -m luigi --local-scheduler --module workflows <workflow>

TODO: Make a real package and update imports

Requirements:
- "data-validation-framework>=0.3.1"
- "morphology-workflows>=0.2.0"
"""
import luigi_tools
from data_validation_framework.target import TaggedOutputLocalTarget

from axon_synthesis.add_tufts import AddTufts
from axon_synthesis.config import Config
from axon_synthesis.create_dataset import FetchWhiteMatterRecipe
from axon_synthesis.create_dataset import RepairDataset
from axon_synthesis.PCSF.clustering import ClusterTerminals
from axon_synthesis.PCSF.create_graph import CreateGraph
from axon_synthesis.PCSF.plot_steiner import PlotSolutions
from axon_synthesis.PCSF.steiner_morphologies import SteinerMorphologies
from axon_synthesis.statistics import CompareStatistics
from axon_synthesis.statistics import PlotStatistics

TaggedOutputLocalTarget.set_default_prefix(Config().output_dir)


class DiscoverRawData(luigi_tools.task.WorkflowWrapperTask):
    """This is the first workflow: curate and plot the raw data."""

    def requires(self):
        return RepairDataset()


class ExploreStatistics(luigi_tools.task.WorkflowWrapperTask):
    """This workflow creates and plots statistics about data."""

    def requires(self):
        return PlotStatistics()


class CreateInputs(luigi_tools.task.WorkflowWrapperTask):
    """This workflow creates inputs for long-range axon synthesis."""

    def requires(self):
        return {
            "clusters": ClusterTerminals(),
            "WMR": FetchWhiteMatterRecipe(),
        }


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


class BuildTufts(luigi_tools.task.WorkflowWrapperTask):
    """This workflow add tufts to the long range trunk built from the Stein Tree."""

    def requires(self):
        return AddTufts()


class ValidateSolutions(luigi_tools.task.WorkflowWrapperTask):
    """This workflow validates the results from the Steiner Tree computation."""

    def requires(self):
        return CompareStatistics()
