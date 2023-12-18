"""Define the class to store the inputs."""
import json
import logging
from pathlib import Path
from typing import ClassVar

import h5py
import pandas as pd
from neurots.validator import ValidationError
from neurots.validator import validate_neuron_distribs
from neurots.validator import validate_neuron_params

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.base_path_builder import FILE_SELECTION
from axon_synthesis.base_path_builder import BasePathBuilder
from axon_synthesis.input_creation import pop_neuron_numbers
from axon_synthesis.input_creation.clustering import Clustering
from axon_synthesis.typing import FileType
from axon_synthesis.typing import Self
from axon_synthesis.utils import recursive_to_str
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe
from axon_synthesis.white_matter_recipe import WmrConfig

LOGGER = logging.getLogger(__name__)


class Inputs(BasePathBuilder):
    """Class to store the Inputs."""

    _filenames: ClassVar[dict] = {
        "BRAIN_REGIONS_MASK_FILENAME": "region_masks.h5",
        "CLUSTERING_DIRNAME": "Clustering",
        "METADATA_FILENAME": "metadata.json",
        "POPULATION_NEURON_NUMBERS_FILENAME": "neuron_density.csv",
        "POPULATION_PROBABILITIES_FILENAME": "population_probabilities.csv",
        "PROJECTION_PROBABILITIES_FILENAME": "projection_probabilities.csv",
        "TUFT_DISTRIBUTIONS_FILENAME": "tuft_distributions.json",
        "TUFT_PARAMETERS_FILENAME": "tuft_parameters.json",
        "WMR_DIRNAME": "WhiteMatterRecipe",
    }

    def __init__(
        self,
        path: FileType,
        morphology_path: FileType | None = None,
        pop_probabilities: FileType | None = None,
        proj_probabilities: FileType | None = None,
        neuron_density: float | None = None,
        **kwargs,
    ):
        """Create a new Inputs object.

        Args:
            path: The base path used to build the relative paths.
            morphology_path: The path of the directory containing the input morphologies.
            pop_probabilities: The path to the file containing the population probabilities.
            proj_probabilities: The path to the file containing the projection probabilities.
            neuron_density: The mean neuron density used to compute the expected total number of
                neurons in target regions.
            **kwargs: The keyword arguments are passed to the base constructor.
        """
        super().__init__(path, **kwargs)

        self.atlas = None
        self.brain_regions_mask_file = None
        self.clustering_data = None
        self.neuron_density = neuron_density
        self._pop_neuron_numbers = None
        self.pop_probabilities = None
        self.proj_probabilities = None
        self.tuft_distributions = None
        self.tuft_parameters = None
        self.wmr = None

        if self.METADATA_FILENAME.exists():
            self.load_metadata()
        else:
            self._metadata = {
                "clustering": self.CLUSTERING_DIRNAME,
                "neuron_density": self.neuron_density,
                "path": self.path,
                "WMR": self.WMR_DIRNAME,
            }
            if morphology_path is not None:
                self._metadata["morphology_path"] = Path(morphology_path)
            if pop_probabilities is not None:
                self._metadata["population_probabilities"] = Path(pop_probabilities)
            if proj_probabilities is not None:
                self._metadata["projection_probabilities"] = Path(proj_probabilities)
            self.metadata_to_attributes()

    @staticmethod
    def _format_metadata(metadata: dict) -> dict:
        # Format metadata elements
        metadata["clustering"] = Path(metadata["clustering"])
        metadata["morphology_path"] = Path(metadata["morphology_path"])
        metadata["path"] = Path(metadata["path"])
        metadata["WMR"] = Path(metadata["WMR"])

        if "atlas" in metadata:
            atlas_config = AtlasConfig.from_dict(metadata["atlas"])
            metadata["atlas"] = atlas_config.to_dict()

        return metadata

    @staticmethod
    def _unformat_metadata(metadata: dict) -> dict:
        return recursive_to_str(metadata)

    def _update_atlas_metadata(self) -> None:
        """Update the metadata with atlas config if possible."""
        if "atlas" not in self._metadata and self.atlas is not None:
            self._metadata["atlas"] = self.atlas.config.to_dict()

    @property
    def metadata(self) -> dict:
        """Return the metadata and automatically update them when an atlas is added."""
        self._update_atlas_metadata()
        return self._metadata

    def save_metadata(self):
        """Save the metadata."""
        with self.METADATA_FILENAME.open("w") as f:
            json.dump(
                self._unformat_metadata(self.metadata),
                f,
                indent=4,
                sort_keys=True,
            )

    def load_metadata(self):
        """Load the metadata."""
        try:
            with self.METADATA_FILENAME.open() as f:
                self._metadata = self._format_metadata(json.load(f))
            self.metadata_to_attributes()
        except Exception as exc:  # noqa: BLE001
            msg = "Could not load the inputs"
            raise RuntimeError(msg) from exc

    def metadata_to_attributes(self):
        """Propagate metadata to attributes."""
        self.reset_path(self.metadata["path"])
        self._filenames["CLUSTERING_DIRNAME"] = Path(self.metadata["clustering"]).name
        self._filenames["WMR_DIRNAME"] = Path(self.metadata["WMR"]).name
        if "morphology_path" in self.metadata:
            self.MORPHOLOGY_DIRNAME = Path(self.metadata["morphology_path"])
        self.neuron_density = self.metadata["neuron_density"]
        self._reset_attributes()

    def load_atlas(self, atlas_config=None):
        """Load the Atlas."""
        if "atlas" not in self.metadata and atlas_config is None:
            msg = "Could not load the Atlas because no atlas configuration was provided."
            raise ValueError(msg)
        if atlas_config is None:
            atlas_config = AtlasConfig.from_dict(self.metadata["atlas"])
        elif isinstance(atlas_config, dict):
            atlas_config = AtlasConfig.from_dict(atlas_config)
        self.atlas = AtlasHelper(atlas_config)
        self._update_atlas_metadata()

    def load_brain_regions_masks(self):
        """Load the brain region masks."""
        self.brain_regions_mask_file = h5py.File(self.BRAIN_REGIONS_MASK_FILENAME)

    def load_wmr(self, wmr_config: WmrConfig | None = None):
        """Load the Atlas."""
        self.wmr = WhiteMatterRecipe(self.WMR_DIRNAME, load=False)
        if self.wmr.exists():
            self.wmr.load()
        elif wmr_config is None:
            msg = (
                f"The directory '{self.WMR_DIRNAME}' that should contain the White Matter Recipe "
                "does not exist and no config was provided to load a raw WMR"
            )
            raise FileNotFoundError(msg)
        elif self.atlas is None:
            msg = (
                f"The directory '{self.WMR_DIRNAME}' that should contain the White Matter Recipe "
                "does not exist and the atlas is not loaded yet"
            )
            raise FileNotFoundError(msg)
        else:
            self.wmr.load_from_raw_wmr(
                wmr_config,
                self.atlas,
            )
            self.wmr.save()

    def load_clustering_data(self, file_selection=FILE_SELECTION.REQUIRED_ONLY):
        """Load the Atlas."""
        self.clustering_data = Clustering.load(
            self.CLUSTERING_DIRNAME,
            file_selection,
        )

    @property
    def pop_neuron_numbers(self):
        """Load the population numbers."""
        if self._pop_neuron_numbers is None:
            if self.POPULATION_NEURON_NUMBERS_FILENAME.exists():
                self._pop_neuron_numbers = pd.read_csv(self.POPULATION_NEURON_NUMBERS_FILENAME)
            else:
                self._pop_neuron_numbers = pop_neuron_numbers.compute(
                    self.wmr.populations,
                    self.neuron_density,
                    self.POPULATION_NEURON_NUMBERS_FILENAME,
                )
        return self._pop_neuron_numbers

    def load_probabilities(self):
        """Load the population and projection probabilities."""
        self.population_probabilities = pd.read_csv(self.POPULATION_PROBABILITIES_FILENAME)
        self.projection_probabilities = pd.read_csv(self.PROJECTION_PROBABILITIES_FILENAME)

    def load_tuft_params_and_distrs(self):
        """Load the parameters and distributions used to generate the tufts."""
        missing_files = [
            i
            for i in [self.TUFT_DISTRIBUTIONS_FILENAME, self.TUFT_PARAMETERS_FILENAME]
            if not i.exists()
        ]
        if missing_files:
            msg = "The following file"
            if len(missing_files) > 1:
                msg += "s does"
            else:
                msg += " do"
            msg += " not exist: {missing_files}"
            raise FileNotFoundError(msg)
        with self.TUFT_PARAMETERS_FILENAME.open() as f:
            parameters = json.load(f)
            # if "axon" in parameters:
            #     parameters["basal_dendrite"] = parameters["axon"]
        with self.TUFT_DISTRIBUTIONS_FILENAME.open() as f:
            distributions = json.load(f)
            # if "axon" in distributions:
            #     distributions["basal_dendrite"] = distributions["axon"]

        try:
            validate_neuron_params(parameters)
        except ValidationError as exc:
            msg = "The given tuft parameters are not valid"
            raise ValidationError(msg) from exc

        try:
            validate_neuron_distribs(distributions)
        except ValidationError as exc:
            msg = "The given tuft distributions are not valid"
            raise ValidationError(msg) from exc

        self.tuft_parameters = parameters
        self.tuft_distributions = distributions

    @classmethod
    def load(cls, path: FileType, atlas_config: AtlasConfig | None = None) -> Self:
        """Load all the inputs from the given path."""
        obj = cls(path)
        obj.load_atlas(atlas_config)
        obj.load_wmr()
        obj.load_clustering_data()
        obj.load_brain_regions_masks()
        obj.load_probabilities()
        obj.load_tuft_params_and_distrs()
        if obj.pop_neuron_numbers is None:
            msg = "Could not load or compute the population numbers in target regions"
            raise RuntimeError(msg)
        return obj

    def compute_probabilities(self, source="WMR"):
        """Compute projection probabilities."""
        if (
            self.POPULATION_PROBABILITIES_FILENAME.exists()
            and self.PROJECTION_PROBABILITIES_FILENAME.exists()
        ):
            LOGGER.info(
                "The population and projection probabilities are not computed because they already "
                "exists in '%s' and '%s'",
                self.POPULATION_PROBABILITIES_FILENAME,
                self.PROJECTION_PROBABILITIES_FILENAME,
            )
            return None

        # TODO: For now we support only the WMR but latter other methods may come.
        if source == "WMR":
            (
                self.population_probabilities,
                self.projection_probabilities,
            ) = self.wmr.compute_probabilities(self.atlas)
        else:
            msg = f"The value '{source}' is not known."
            raise ValueError(msg)

        # Export the population probabilities
        self.population_probabilities.to_csv(self.POPULATION_PROBABILITIES_FILENAME, index=False)

        # Export the projection probabilities
        self.projection_probabilities.to_csv(self.PROJECTION_PROBABILITIES_FILENAME, index=False)

        return self.population_probabilities, self.projection_probabilities

    def compute_atlas_region_masks(self):
        """Compute all region masks of the Atlas."""
        if self.atlas is None:
            msg = "The Atlas must be loaded before computing the region masks."
            raise RuntimeError(msg)
        self.atlas.compute_region_masks(self.BRAIN_REGIONS_MASK_FILENAME)