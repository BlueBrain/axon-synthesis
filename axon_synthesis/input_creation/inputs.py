"""Define the class to store the inputs."""
import json
from pathlib import Path
from typing import ClassVar

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.atlas import AtlasHelper
from axon_synthesis.base_path_builder import BasePathBuilder
from axon_synthesis.input_creation.clustering import LOADING_TYPE
from axon_synthesis.input_creation.clustering import Clustering
from axon_synthesis.typing import FileType
from axon_synthesis.typing import Self
from axon_synthesis.utils import recursive_to_str
from axon_synthesis.white_matter_recipe import WhiteMatterRecipe


class Inputs(BasePathBuilder):
    """Class to store the Inputs."""

    _filenames: ClassVar[dict] = {
        "BRAIN_REGIONS_MASK_FILENAME": "region_masks.h5",
        "CLUSTERING_DIRNAME": "Clustering",
        "METADATA_FILENAME": "metadata.json",
        "POP_NEURON_NUMBERS_FILENAME": "neuron_density.csv",
        "WMR_DIRNAME": "WhiteMatterRecipe",
    }

    def __init__(self, path: FileType, morphology_path: FileType):
        """Create a new Inputs object.

        Args:
            path: The base path used to build the relative paths.
            morphology_path: The path of the directory containing the input morphologies.
        """
        super().__init__(path)

        self.atlas = None
        self.wmr = None
        self.clustering_data = None

        if self.METADATA_FILENAME.exists():
            self.load_metadata()
        else:
            self._metadata = {
                "clustering": self.CLUSTERING_DIRNAME,
                "morphology_path": Path(morphology_path),
                "path": self.path,
                "WMR": self.WMR_DIRNAME,
            }

        self.MORPHOLOGY_DIRNAME = self.metadata["morphology_path"]

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
        except Exception as exc:  # noqa: BLE001
            msg = "Could not load the inputs"
            raise RuntimeError(msg) from exc

    def load_atlas(self, atlas_config):
        """Load the Atlas."""
        self.atlas = AtlasHelper(atlas_config)
        self._update_atlas_metadata()

    def load_wmr(self):
        """Load the Atlas."""
        self.wmr = WhiteMatterRecipe.load(self.WMR_DIRNAME)

    def load_clustering_data(self, loading_type=LOADING_TYPE.REQUIRED_ONLY):
        """Load the Atlas."""
        self.clustering_data = Clustering.load(
            self.CLUSTERING_DIRNAME,
            loading_type,
        )

    @classmethod
    def load(cls, path: FileType) -> Self:
        """Load all the inputs from the given path."""
        obj = cls(path)
        obj.load_atlas()
        obj.load_wmr()
        obj.load_clustering_data()
        return obj
