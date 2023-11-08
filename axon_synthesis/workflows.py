"""The main workflows of the AxonSynthesis package."""
import luigi_tools

from axon_synthesis.add_tufts import AddTufts
from axon_synthesis.create_dataset import FetchWhiteMatterRecipe
from axon_synthesis.create_dataset import RepairDataset
from axon_synthesis.PCSF.create_graph import CreateGraph
from axon_synthesis.PCSF.plot_steiner import PlotSolutions
from axon_synthesis.PCSF.steiner_morphologies import SteinerMorphologies
from axon_synthesis.pop_neuron_numbers import PickPopulationNeuronNumbers
from axon_synthesis.prepare_atlas import PrepareAtlas
from axon_synthesis.statistics import CompareStatistics
from axon_synthesis.statistics import PlotStatistics
from axon_synthesis.trunk_properties import LongRangeTrunkProperties


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
            "atlas": PrepareAtlas(),
            "trunk_properties": LongRangeTrunkProperties(),
            "WMR": FetchWhiteMatterRecipe(),
            "pop_neuron_numbers": PickPopulationNeuronNumbers(),
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


class Synthesis(luigi_tools.task.WorkflowWrapperTask):
    """This workflow synthesize morphologies according to given inputs."""

    def requires(self):
        return {
            "inputs": CreateInputs(),
            "tufts": AddTufts(),
        }


class ValidateSolutions(luigi_tools.task.WorkflowWrapperTask):
    """This workflow validates the results from the Steiner Tree computation."""

    def requires(self):
        return CompareStatistics()
