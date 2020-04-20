"""Axon synthesis"""

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Python < 3.6 is not supported. Please update the environment")


VERSION = imp.load_source("", "axon_synthesis/version.py").__version__

config = {
    "description": "axon-synthesis: a python package synthesizing axon morphologies",
    "name": "axon-synthesis",
    "version": VERSION,
    "author_email": "valerii.souchoroukov@epfl.ch",
    # "url": "https://bbpteam.epfl.ch/documentation/projects/axon-synthesis",
    "project_urls": {
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "https://bbpgitlab.epfl.ch/neuromath/axon-synthesis",
    },
    "install_requires": [
        "tmd==2.0.8",
        'neurom>=3.0,<4.0',
        "numpy>=1.18.1",
        "scipy>=1.4.1",
        "matplotlib>=1.3.1",
    ],
    "packages": find_packages(),
    "include_package_data": True,
}

setup(**config)
