"""Package to create inputs."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import logging

from axon_synthesis.atlas import AtlasConfig
from axon_synthesis.inputs import Inputs
from axon_synthesis.inputs.clustering import cluster_morphologies
from axon_synthesis.typing import FileType
from axon_synthesis.typing import SeedType
from axon_synthesis.utils import ParallelConfig
from axon_synthesis.white_matter_recipe import WmrConfig

LOGGER = logging.getLogger(__name__)


def create_inputs(
    morphology_dir: FileType,
    output_dir: FileType,
    clustering_parameters: dict,
    wmr_config: WmrConfig | None = None,
    atlas_config: AtlasConfig | None = None,
    neuron_density: float | None = None,
    bouton_density: float | None = None,
    *,
    debug: bool = False,
    rng: SeedType = None,
    parallel_config: ParallelConfig | None = None,
):
    """Create all inputs required to synthesize long-range axons."""
    inputs = Inputs(output_dir, morphology_dir, neuron_density=neuron_density, create=True)

    if atlas_config is not None:
        atlas_config.load_region_map = True

        # Load the Atlas
        inputs.load_atlas(atlas_config)

        # Pre-compute atlas data
        inputs.compute_atlas_region_masks()

    if wmr_config is not None:
        # Process the White Matter Recipe
        inputs.load_wmr(wmr_config)

        # Compute the population and projection probabilities
        inputs.compute_probabilities()

    # Define the tufts and main trunks in input morphologies and compute the properties of the long
    # range trunk and the tufts of each morphology
    inputs.clustering_data = cluster_morphologies(
        inputs.MORPHOLOGY_DIRNAME,
        clustering_parameters,
        inputs.CLUSTERING_DIRNAME,
        atlas=inputs.atlas,
        wmr=inputs.wmr,
        pop_neuron_numbers=inputs.pop_neuron_numbers,
        bouton_density=bouton_density,
        debug=debug,
        rng=rng,
        parallel_config=parallel_config,
    )
    inputs.clustering_data.save()

    # Export the input metadata
    inputs.save_metadata()

    return inputs
