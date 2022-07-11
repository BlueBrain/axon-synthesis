"""The main workflows."""
import luigi_tools

from morphology_processing_workflow.tasks.workflows import Curate

from create_dataset import CreateDatasetForRepair
from PCSF.extract_terminals import ExtractTerminals
from PCSF.clustering import ClusterTerminals


class DiscoverRawData(luigi_tools.task.WorkflowWrapperTask):
    """This is the first workflow: curate and plot the raw data."""

    def requires(self):
        dataset = CreateDatasetForRepair()
        return [dataset, Curate(dataset_df=dataset.output().path)]


class PrepareSteinerData(luigi_tools.task.WorkflowWrapperTask):
    """This workflow prepares the data used for Steiner Tree computation."""

    def requires(self):
        return ClusterTerminals()
