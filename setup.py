"""Setup for the AxonSynthesis package."""
import importlib.util
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

spec = importlib.util.spec_from_file_location(
    "axon_synthesis.version",
    "axon_synthesis/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

with open("requirements.txt", encoding="utf-8") as f:
    reqs = f.read().splitlines()

setup(
    name="axon-synthesis",
    author="Adrien Berchet",
    author_email="adrien.berchet@epfl.ch",
    version=VERSION,
    description="A package to synthesize artificial axons",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://bbpteam.epfl.ch/documentation/projects/axon-synthesis",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "git@bbpgitlab.epfl.ch:neuromath/user/aberchet/axon-synthesis.git",
    },
    license="BBP-internal-confidential",
    install_requires=reqs,
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    extras_require={
        "docs": ["m2r2", "sphinx", "sphinx-bluebrain-theme"],
        "test": [
            "brainbuilder",
            "mock",
            "pytest",
            "pytest-cov",
            "pytest-depends",
            "pytest-html",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
