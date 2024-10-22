"""Create number of neurons in each population."""

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
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def compute(populations, neuron_density, output_path=None):
    """Compute the number of neurons in each brain region."""
    if output_path is not None:
        output_path = Path(output_path)

        if output_path.exists():
            LOGGER.info(
                "The population neuron numbers are not computed because they already exist in '%s'",
                output_path,
            )
            return None

    populations["pop_neuron_numbers"] = (
        populations["atlas_region_volume"] * neuron_density
    ).astype(int)

    # Format the population numbers
    populations["pop_neuron_numbers"] = populations["pop_neuron_numbers"].clip(lower=2)
    res = populations[
        [
            "pop_raw_name",
            "atlas_region_id",
            "atlas_region_volume",
            "pop_neuron_numbers",
        ]
    ]

    # Export the results
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(output_path, index=False)

    return res
