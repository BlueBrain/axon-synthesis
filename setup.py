"""Setup for the axon-synthesis package."""

import importlib.util

from setuptools import setup

spec = importlib.util.spec_from_file_location(
    "axon_synthesis.version",
    "axon_synthesis/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

setup_kwargs = {
    "name": "axon-synthesis",
    "version": VERSION,
}

if __name__ == "__main__":
    setup(**setup_kwargs)
