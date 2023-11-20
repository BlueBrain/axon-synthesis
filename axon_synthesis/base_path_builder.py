"""Module to define a base class for relative paths storage and processing."""
from pathlib import Path
from typing import ClassVar

from axon_synthesis.typing import FileType


class BasePathBuilder:
    """A base class to store relative file paths."""

    _filenames: ClassVar[dict] = {}
    _optional_keys: ClassVar[set[str]] = set()

    def __init__(self, path: FileType):
        """Create a new BasePathBuilder object.

        Args:
            path: The base path used to build the relative paths.
        """
        self._path = Path(path)

        # Set attributes
        for k, v in self:
            setattr(self, k, v)

    @property
    def path(self) -> Path:
        """Return the associated path."""
        return self._path

    def __iter__(self):
        """Return a generator to the paths to the associated data files."""
        yield from self.build_paths(self.path).items()

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

    def exists(self, *, require_optional=False):
        """Check if all the paths exist."""
        files = self.required_filenames if require_optional else self.filenames
        return self.path.exists() and all(v.exists() for k, v in files.items())
