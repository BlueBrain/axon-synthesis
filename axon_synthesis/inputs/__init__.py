"""Define the class to store the inputs."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import json
import logging
from importlib import resources
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
from axon_synthesis.inputs import pop_neuron_numbers
from axon_synthesis.inputs.clustering import Clustering
from axon_synthesis.typing import FileType
from axon_synthesis.typing import Self
from axon_synthesis.utils import merge_json_files
from axon_synthesis.utils import recursive_to_str
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe
from axon_synthesis.white_matter_recipe import WmrConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_TUFT_DISTRIBUTIONS = resources.files("axon_synthesis") / "data/tuft_distributions.json"
DEFAULT_TUFT_PARAMETERS = resources.files("axon_synthesis") / "data/tuft_parameters.json"


class Inputs(BasePathBuilder):
    """Class to store the Inputs."""

    _filenames: ClassVar[dict] = {
        "BRAIN_REGIONS_MASK_FILENAME": "region_masks.h5",
        "CLUSTERING_DIRNAME": "Clustering",
        "METADATA_FILENAME": "metadata.json",
        "POPULATION_NEURON_NUMBERS_FILENAME": "neuron_density.csv",
        "POPULATION_PROBABILITIES_FILENAME": "population_probabilities.csv",
        "POPULATION_TUFT_NUMBER_FILENAME": "tufts_numbers_distribution.json",
        "PROJECTION_PROBABILITIES_FILENAME": "projection_probabilities.csv",
        "TUFT_DISTRIBUTIONS_FILENAME": "tuft_distributions.json",
        "TUFT_PARAMETERS_FILENAME": "tuft_parameters.json",
        "WMR_DIRNAME": "WhiteMatterRecipe",
    }

    def __init__(
        self,
        path: FileType,
        morphology_path: FileType | None = None,
        *,
        pop_probability_path: FileType | None = None,
        proj_probability_path: FileType | None = None,
        brain_regions_mask_path: FileType | None = None,
        neuron_density: float | None = None,
        **kwargs,
    ):
        """Create a new Inputs object.

        Args:
            path: The base path used to build the relative paths.
            morphology_path: The path of the directory containing the input morphologies.
            pop_probability_path: The path to the file containing the population probabilities.
            proj_probability_path: The path to the file containing the projection probabilities.
            brain_regions_mask_path: The path to the file containing the pre-computed brain
                region masks.
            neuron_density: The mean neuron density used to compute the expected total number of
                neurons in target regions.
            **kwargs: The keyword arguments are passed to the base constructor.
        """
        super().__init__(path, **kwargs)

        self._brain_regions_mask_file: h5py.File | None = None
        self._pop_neuron_numbers: pd.DataFrame | None = None
        self._metadata: dict
        self.atlas: AtlasHelper | None = None
        self.clustering_data: Clustering | None = None
        self.neuron_density = neuron_density
        self.pop_probability_path = None
        self.population_probabilities: pd.DataFrame | None = None
        self.proj_probability_path = None
        self.projection_probabilities: pd.DataFrame | None = None
        self.tuft_distributions: dict | None = None
        self.tuft_parameters: dict | None = None
        self.tuft_number: pd.DataFrame | None = None
        self.wmr: WhiteMatterRecipe | None = None

        if self.METADATA_FILENAME.exists():
            self.load_metadata()
        else:
            self._metadata = {
                "clustering": self.CLUSTERING_DIRNAME.relative_to(self.path),
                "neuron_density": self.neuron_density,
                "path": self.path,
                "WMR": self.WMR_DIRNAME.relative_to(self.path),
            }
            if morphology_path is not None:
                self._metadata["morphology_path"] = Path(morphology_path)
            if pop_probability_path is not None:
                self._metadata["population_probabilities"] = Path(pop_probability_path)
            if proj_probability_path is not None:
                self._metadata["projection_probabilities"] = Path(proj_probability_path)
            if brain_regions_mask_path is not None:
                self._metadata["brain_regions_mask"] = Path(brain_regions_mask_path)
            self.metadata_to_attributes()

    @staticmethod
    def _format_metadata(metadata: dict) -> dict:
        # Format metadata elements
        metadata["path"] = Path(metadata["path"])
        metadata["clustering"] = Path(metadata["clustering"])
        metadata["morphology_path"] = Path(metadata["morphology_path"])
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
        with self.METADATA_FILENAME.open("w", encoding="utf-8") as f:
            json.dump(
                self._unformat_metadata(self.metadata),
                f,
                indent=4,
                sort_keys=True,
            )

    def load_metadata(self):
        """Load the metadata."""
        try:
            with self.METADATA_FILENAME.open(encoding="utf-8") as f:
                self._metadata = self._format_metadata(json.load(f))
            self.metadata_to_attributes()
        except Exception as exc:
            msg = "Could not load the inputs"
            raise RuntimeError(msg) from exc

    def metadata_to_attributes(self):
        """Propagate metadata to attributes."""
        self._filenames["CLUSTERING_DIRNAME"] = Path(self.metadata["clustering"])
        self._filenames["WMR_DIRNAME"] = Path(self.metadata["WMR"])
        if "brain_regions_mask" in self.metadata:
            self._filenames["BRAIN_REGIONS_MASK_FILENAME"] = str(
                self.metadata["brain_regions_mask"]
            )
        if "morphology_path" in self.metadata:
            self.MORPHOLOGY_DIRNAME = self.metadata["morphology_path"]
        self.neuron_density = self.metadata["neuron_density"]
        self._reset_attributes()

    def load_atlas(self, atlas_config=None):
        """Load the Atlas."""
        if "atlas" not in self.metadata and atlas_config is None:
            msg = "Did not load the Atlas because no atlas configuration was provided."
            LOGGER.debug(msg)
            return
        if atlas_config is None:
            atlas_config = AtlasConfig.from_dict(self.metadata["atlas"])
        elif isinstance(atlas_config, dict):
            atlas_config = AtlasConfig.from_dict(atlas_config)
        # if not getattr(atlas_config, "load_region_map", True):
        #     LOGGER.debug("The 'load_region_map' attribute is forced to 'True'")
        #     atlas_config.load_region_map = True
        if isinstance(atlas_config, AtlasHelper):
            self.atlas = atlas_config
        else:
            self.atlas = AtlasHelper(atlas_config)
        self._update_atlas_metadata()

    def load_brain_regions_masks(self):
        """Load the brain region masks."""
        if self.BRAIN_REGIONS_MASK_FILENAME.exists():
            self._brain_regions_mask_file = h5py.File(self.BRAIN_REGIONS_MASK_FILENAME)
        elif self.atlas is not None:
            LOGGER.debug(
                (
                    "Computing the brain regions masks because the file '%s' is missing but an "
                    "atlas was provided."
                ),
                self.BRAIN_REGIONS_MASK_FILENAME,
            )
            self.compute_atlas_region_masks()
        else:
            LOGGER.debug(
                (
                    "Did not load the brain region masks because the file %s does not exist and "
                    "no atlas was provided."
                ),
                self.BRAIN_REGIONS_MASK_FILENAME,
            )

    def unload_brain_regions_masks(self):
        """Unload the brain region masks."""
        self._brain_regions_mask_file = None

    @property
    def brain_regions_mask_file(self):
        """Return the brain_regions_mask_file attribute or load if it's None."""
        if self._brain_regions_mask_file is None:
            self.load_brain_regions_masks()
        return self._brain_regions_mask_file

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

    def load_clustering_data(self, file_selection=FILE_SELECTION.REQUIRED_ONLY, **kwargs):
        """Load the Atlas."""
        self.clustering_data = Clustering.load(
            self.CLUSTERING_DIRNAME,
            file_selection,
            **kwargs,
        )

    @property
    def pop_neuron_numbers(self):
        """Load the population numbers."""
        if self._pop_neuron_numbers is None:
            if self.POPULATION_NEURON_NUMBERS_FILENAME.exists():
                self._pop_neuron_numbers = pd.read_csv(self.POPULATION_NEURON_NUMBERS_FILENAME)
            elif self.wmr is not None:
                self._pop_neuron_numbers = pop_neuron_numbers.compute(
                    self.wmr.populations,
                    self.neuron_density,
                    self.POPULATION_NEURON_NUMBERS_FILENAME,
                )
            else:
                return pd.DataFrame(
                    columns=[
                        "pop_raw_name",
                        "atlas_region_id",
                        "atlas_region_volume",
                        "pop_neuron_numbers",
                    ]
                )
        return self._pop_neuron_numbers

    def load_probabilities(self, **kwargs):
        """Load the population and projection probabilities."""
        self.update_from_dict(
            "population_probabilities_file", "POPULATION_PROBABILITIES_FILENAME", kwargs
        )
        if self.POPULATION_PROBABILITIES_FILENAME.exists():
            self.population_probabilities = pd.read_csv(self.POPULATION_PROBABILITIES_FILENAME)
            self.population_probabilities = self.population_probabilities.astype(
                {
                    "probability": float,
                    **{
                        col: str
                        for col in self.population_probabilities.columns
                        if col.endswith("population_id")
                    },
                }
            )

        self.update_from_dict(
            "projection_probabilities_file", "PROJECTION_PROBABILITIES_FILENAME", kwargs
        )
        if self.PROJECTION_PROBABILITIES_FILENAME.exists():
            self.projection_probabilities = pd.read_csv(self.PROJECTION_PROBABILITIES_FILENAME)
            self.projection_probabilities = self.projection_probabilities.astype(
                {
                    "probability": float,
                    **{
                        col: str
                        for col in self.projection_probabilities.columns
                        if col.endswith("population_id")
                    },
                }
            )

        self.update_from_dict(
            "population_tuft_number_file", "POPULATION_TUFT_NUMBER_FILENAME", kwargs
        )
        if self.POPULATION_TUFT_NUMBER_FILENAME.exists():
            with self.POPULATION_TUFT_NUMBER_FILENAME.open(mode="r", encoding="utf-8") as f:
                number = pd.DataFrame(json.load(f)).set_index("target_population_id")
            if (number["mean_tuft_number"] <= 0).any():
                LOGGER.error(
                    (
                        "The number of tuft must be greater than 0 for all populations but it's "
                        "not the case for the following ones: %s"
                    ),
                    number.loc[number["mean_tuft_number"] < 0].index.tolist(),
                )
            self.tuft_number = number

    def load_tuft_params_and_distrs(self, **kwargs):
        """Load and validate the parameters and distributions used to generate the tufts."""
        self.update_from_dict("tuft_parameters_file", "TUFT_PARAMETERS_FILENAME", kwargs)
        parameters = merge_json_files(DEFAULT_TUFT_PARAMETERS, self.TUFT_PARAMETERS_FILENAME)
        try:
            validate_neuron_params(parameters)
        except ValidationError as exc:
            msg = "The given tuft parameters are not valid"
            raise ValidationError(msg) from exc
        self.tuft_parameters = parameters

        self.update_from_dict("tuft_distributions_file", "TUFT_DISTRIBUTIONS_FILENAME", kwargs)
        distributions = merge_json_files(
            DEFAULT_TUFT_DISTRIBUTIONS, self.TUFT_DISTRIBUTIONS_FILENAME
        )
        try:
            validate_neuron_distribs(distributions)
        except ValidationError as exc:
            msg = "The given tuft distributions are not valid"
            raise ValidationError(msg) from exc
        self.tuft_distributions = distributions

    @classmethod
    def load(
        cls,
        path: FileType | None,
        atlas_config: AtlasConfig | AtlasHelper | None = None,
        **kwargs: dict,
    ) -> Self:
        """Load all the inputs from the given path."""
        if path is None:
            missing_keys = {
                "projection_probabilities_file",
                "population_probabilities_file",
                "trunk_properties_file",
                "tuft_properties_file",
            }.difference(kwargs.keys())
            if missing_keys:
                msg = (
                    "The following keys are missing in kwargs because the 'path' is not "
                    f"provided: {sorted(missing_keys)}"
                )
                raise RuntimeError(msg)
        obj = cls(path or "")
        obj.load_atlas(atlas_config)
        obj.load_clustering_data(**kwargs)
        obj.load_brain_regions_masks()
        obj.load_probabilities(**kwargs)
        obj.load_tuft_params_and_distrs(**kwargs)
        if obj.pop_neuron_numbers is None:
            msg = "Could not load or build the population numbers in target regions"
            raise RuntimeError(msg)
        return obj

    def export_probabilities(self):
        """Export the computed probabilities."""
        # Export the population probabilities
        if self.population_probabilities is not None:
            self.population_probabilities.to_csv(
                self.POPULATION_PROBABILITIES_FILENAME, index=False
            )

        # Export the projection probabilities
        if self.projection_probabilities is not None:
            self.projection_probabilities.to_csv(
                self.PROJECTION_PROBABILITIES_FILENAME, index=False
            )

    def compute_probabilities(self, source="WMR", *, export=True):
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

        # TODO: For now we support only the WMR but other methods may come later.
        if source == "WMR" and self.wmr is not None and self.atlas is not None:
            (
                self.population_probabilities,
                self.projection_probabilities,
            ) = self.wmr.compute_probabilities(self.atlas)
        else:
            msg = f"The value '{source}' is not known."
            raise ValueError(msg)

        if export:
            self.export_probabilities()
        return self.population_probabilities, self.projection_probabilities

    def compute_atlas_region_masks(self):
        """Compute all region masks of the Atlas."""
        if self.atlas is None:
            msg = "The Atlas must be loaded before computing the region masks."
            raise RuntimeError(msg)
        self.atlas.compute_region_masks(self.BRAIN_REGIONS_MASK_FILENAME)
        self._brain_regions_mask_file = h5py.File(self.BRAIN_REGIONS_MASK_FILENAME)
