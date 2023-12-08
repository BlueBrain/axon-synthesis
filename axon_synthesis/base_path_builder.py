"""Module to define a base class for relative paths storage and processing."""
from enum import IntEnum
from pathlib import Path
from typing import ClassVar

from axon_synthesis.typing import FileType

FILE_SELECTION = IntEnum("LoadingType", ["ALL", "REQUIRED_ONLY", "OPTIONAL_ONLY", "NONE"])


class BasePathBuilder:
    """A base class to store relative file paths."""

    _filenames: ClassVar[dict] = {}
    _optional_keys: ClassVar[set[str]] = set()

    def __init__(self, path: FileType, *, exists: bool = False, create: bool = False):
        """Create a new BasePathBuilder object.

        Args:
            path: The base path used to build the relative paths.
            exists: If set to True, the given path must already exist.
            create: If set to True, the given path will be automatically created.
        """
        self._path = Path(path)
        self._reset_attributes()

        if exists and not self.path.exists():
            msg = f"The directory {self.path} does not exist"
            raise FileNotFoundError(msg)

        if create:
            self.create_root()

    @property
    def path(self) -> Path:
        """Return the associated path."""
        return self._path

    def reset_path(self, path):
        """Return the associated path."""
        self._path = Path(path)
        self._reset_attributes()

    def create_root(self):
        """Ensure the root path exists."""
        self.path.mkdir(parents=True, exist_ok=True)

    def __iter__(self):
        """Return a generator to the paths to the associated data files."""
        yield from self.build_paths(self.path).items()

    def _reset_attributes(self) -> None:
        """Reset path attributes."""
        for k, v in self:
            setattr(self, k, v)

    @classmethod
    def build_paths(cls, path) -> dict[str, Path]:
        """Build the paths to the associated data files."""
        path = Path(path)
        return {k: path / v for k, v in cls._filenames.items()}

    @property
    def filenames(self):
        """Return the paths to the associated data files."""
        return dict(self)

    @property
    def optional_filenames(self):
        """Return the optional files."""
        return {k: v for k, v in self if k in self._optional_keys}

    @property
    def required_filenames(self):
        """Return the required files."""
        return {k: v for k, v in self if k not in self._optional_keys}

    def filenames_from_type(
        self, *, file_selection: FILE_SELECTION = FILE_SELECTION.ALL
    ) -> dict[str, Path]:
        """Return the associated files according to the given filter tag."""
        if file_selection == FILE_SELECTION.ALL:
            return self.filenames
        if file_selection == FILE_SELECTION.REQUIRED_ONLY:
            return self.required_filenames
        if file_selection == FILE_SELECTION.OPTIONAL_ONLY:
            return self.optional_filenames
        return {}

    def missing_files(self, *, file_selection=FILE_SELECTION.REQUIRED_ONLY):
        """Get the list of missing files based on a given type (required, optional or all)."""
        files = self.filenames_from_type(file_selection=file_selection)
        return {k: v for k, v in files.items() if not v.exists()}

    def exists(self, *, file_selection=FILE_SELECTION.REQUIRED_ONLY):
        """Check if all the paths exist."""
        return not self.missing_files(file_selection=file_selection)

    def raise_missing_files(
        self, *, file_selection=FILE_SELECTION.REQUIRED_ONLY, missing_files=None
    ):
        """Raise a 'FileNotFoundError' exception with the relevant list of missing files."""
        msg = "The following files are missing: %s"
        if missing_files is None:
            missing_files = list(self.missing_files(file_selection).keys())
        raise FileNotFoundError(msg, missing_files)

    def assert_exists(self, *, file_selection=FILE_SELECTION.REQUIRED_ONLY):
        """Raise a 'FileNotFoundError' exception if the relevant files do not exist."""
        files = list(self.missing_files(file_selection).keys())
        if files:
            self.raise_missing_files(missing_files=files)
