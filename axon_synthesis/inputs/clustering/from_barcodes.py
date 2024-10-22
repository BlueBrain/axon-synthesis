"""Clustering from barcodes."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

# mypy: ignore-errors
# import json
from collections import defaultdict
from itertools import pairwise

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from morphio import IterType
from scipy import stats
from tmd.io.io import load_neuron_from_morphio
from tmd.Topology.analysis import barcode_bin_centers
from tmd.Topology.analysis import histogram_horizontal
from tmd.Topology.analysis import histogram_stepped
from tmd.Topology.methods import tree_to_property_barcode
from tmd.view.plot import barcode as plot_barcode

from axon_synthesis.utils import COORDS_COLS


def barcode_mins(barcode, nb_bins=100, threshold=0.1):
    """Compute min values of a barcode."""
    bin_centers, data = barcode_bin_centers(barcode, num_bins=nb_bins)

    # Gaussian kernel to smooth distribution of bars
    kde = stats.gaussian_kde(data)
    minimas = []

    # Compute first and second derivatives
    der1 = np.gradient(kde(bin_centers))
    der2 = np.gradient(der1)

    # Compute minimas of distribution
    while len(minimas) == 0:
        minimas = np.where(abs(der1) < threshold * np.max(abs(der1)))[0]
        minimas = minimas[der2[minimas] > 0]
        threshold *= 2.0  # if threshold was too small, increase and retry

    def _get_min_indices(mins, der) -> np.ndarray:
        # Compute where the derivative crosses the X axis
        der_zero_intervals = np.append(
            np.insert(
                np.where(np.diff(np.sign(der)))[0],
                0,
                -len(der) * 2,
            ),
            len(der) * 2,
        )

        # Find in which interval is each value
        zero_interval_indices = np.digitize(mins, der_zero_intervals)
        _tmp = pd.DataFrame(
            {
                "min_indices": mins,
                "interval_idx": zero_interval_indices,
            },
        ).groupby("interval_idx")

        # Get the median value
        return _tmp.quantile(interpolation="higher")["min_indices"].astype(int).to_numpy()

    # Keep one minimum per der1 and der2 interval
    min_indices = _get_min_indices(minimas, der1)
    min_indices = _get_min_indices(min_indices, der2)
    min_positions = bin_centers[min_indices]

    return min_indices, min_positions, bin_centers, der1, der2


def compute_clusters(
    config_str,  # noqa: ARG001
    morph,
    axon,  # noqa: ARG001
    axon_id,  # noqa: ARG001
    group_name,  # noqa: ARG001
    group_path,
    group,
    output_cols,  # noqa: ARG001
    *,
    debug=False,
    **_kwargs,
):
    """The points must be inside the ball to be merged."""
    # pylint: disable=too-many-locals
    # pylint: disable=unused-argument
    msg = "This mode is not implemented yet."
    raise NotImplementedError(msg)
    # pylint: disable=unreachable
    # pylint: disable=unused-variable
    new_terminal_points = []

    # config_str = json.dumps(config)

    # Get the complete morphology
    # morph = load_morphology(group_name)
    soma_center = morph.soma.center
    # axons = [i for i in neuron.neurites if i.type == NeuriteType.axon]

    neuron = load_neuron_from_morphio(group_path)
    origin = morph.soma.center
    nb_bins = 100

    for neuron_axon in neuron.axon:
        barcode, bars_to_points = tree_to_property_barcode(
            neuron_axon,
            # lambda tree: tree.get_point_path_distances(),
            lambda tree: tree.get_point_radial_distances(point=origin),
        )

        min_indices, min_positions, bin_centers, der1, der2 = barcode_mins(
            barcode,
            nb_bins,
        )

        # Plot
        if debug:
            fig, (ax_barcode, ax_hist, ax_der) = plt.subplots(1, 3, figsize=(12, 9))

            # Plot barcode
            plt.sca(ax_barcode)
            plot_barcode(barcode, new_fig=False)
            ax_barcode.vlines(
                min_positions,
                0,
                ax_barcode.get_ylim()[1],
                color="red",
                linestyle="--",
                label="In-between tufts",
            )

            # Plot histograms
            plt.sca(ax_hist)
            hist_data_horizontal = histogram_horizontal(barcode, num_bins=nb_bins)
            ax_hist.plot(
                hist_data_horizontal[0][:-1],
                hist_data_horizontal[1],
                color="chocolate",
                alpha=0.7,
                label="Histogram horizontal",
            )
            hist_data_stepped = histogram_stepped(barcode)
            ax_hist.plot(
                hist_data_stepped[0][:-1],
                hist_data_stepped[1],
                color="blue",
                alpha=0.7,
                label="Histogram stepped",
            )
            ax_hist.vlines(
                min_positions,
                0,
                np.max(np.concatenate([hist_data_horizontal[1], hist_data_stepped[1]])),
                color="red",
                linestyle="--",
                label="In-between tufts",
            )
            ax_hist.legend()

            # Plot derivatives
            ax_der.plot(bin_centers, der1, color="chocolate", alpha=0.7, label="1st derivative")
            ax_der.plot(bin_centers, der2, color="blue", alpha=0.7, label="2nd derivative")
            ax_der.vlines(
                min_positions,
                np.min(np.concatenate([der1, der2])),
                np.max(np.concatenate([der1, der2])),
                color="red",
                linestyle="--",
                label="In-between tufts",
            )
            ax_der.legend()

            plt.show()

        group["radial_dist"] = np.linalg.norm(group[COORDS_COLS] - soma_center, axis=1)
        min_positions = np.append(np.insert(min_positions, 0, 0), group["radial_dist"].max() + 1)
        terminal_intervals = np.digitize(group["radial_dist"], min_positions)
        cluster_ids = []

        for num_interval, interval in enumerate(pairwise(min_positions)):
            cluster_terminals = group.loc[terminal_intervals == num_interval + 1]
            terminal_parents = defaultdict(list)
            crossing_sections = set()
            for term_sec in cluster_terminals["section_id"].to_numpy():
                for sec in morph.section(term_sec).iter(IterType.upstream):
                    if np.linalg.norm(sec.points[-1] - soma_center) < interval[0]:
                        break
                    if np.linalg.norm(sec.points[0] - soma_center) <= interval[0]:
                        crossing_sections.add(sec.id)
                    terminal_parents[sec.id].append(term_sec)
            if not crossing_sections:
                crossing_sections.add(min(terminal_parents.keys()))

            # raise NotImplementedError("This mode is not implemented yet.")
            # for sec in crossing_sections:
            #     print(sec)

    return new_terminal_points, cluster_ids
