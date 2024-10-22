"""Tests for the statistics functions."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

from diff_pdf_visually import pdf_similar
from morph_tool.utils import iter_morphology_files

from axon_synthesis.validation.statistics import default_config
from axon_synthesis.validation.statistics import plot_score_matrix


def test_plot_score_matrix(data_dir, tmp_path):
    """Test the plot_score_matrix function."""

    def get_morphs(input_dir) -> list:
        return [(i.name, [i]) for i in sorted(iter_morphology_files(input_dir))]

    ref_morphs_files = get_morphs(data_dir / "mimic_example" / "inputs" / "converted_morphologies")
    res_morphs_files = get_morphs(
        data_dir / "mimic_example" / "synthesis_basic_172992" / "Morphologies"
    ) + get_morphs(data_dir / "mimic_example" / "synthesis_basic_172993" / "Morphologies")

    output = tmp_path / "score_matrix.pdf"
    plot_score_matrix(ref_morphs_files, res_morphs_files, output, default_config())

    expected = data_dir / "score_matrix.pdf"
    pdf_similar(output, expected)
