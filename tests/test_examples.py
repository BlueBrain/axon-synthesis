"""Test the examples."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of Axon Synthesis.
# See https://github.com/BlueBrain/axon-synthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import shutil
from pathlib import Path

import pytest
from dir_content_diff import assert_equal_trees
from dir_content_diff import get_comparators
from dir_content_diff.comparators.morphio import MorphologyComparator
from dir_content_diff.comparators.pandas import HdfComparator
from dir_content_diff.comparators.voxcell import CellCollectionComparator

import axon_synthesis.cli


def _ignore_files(_src, names) -> list[str]:
    """Filter some files."""
    return [
        i for i in names if Path(i).name not in ["172992.asc", "172993.asc"] and "nested" not in i
    ]


# @pytest.mark.parametrize("nb_workers", [0, 2])
# def test_synthesis_example(testing_dir, data_dir, example_dir, cli_runner, nb_workers):
#     """Test the mimic workflow from the general example with 2 morphologies."""
#     shutil.copyfile(example_dir / "config.cfg", testing_dir / "config.cfg")
#     shutil.copyfile(example_dir / "WMR.yaml", testing_dir / "WMR.yaml")
#     shutil.copyfile(example_dir / "input_cells_lite.mvd3", testing_dir / "input_cells.mvd3")
#     shutil.copyfile(
#         example_dir / "axon_grafting_points.h5",
#         testing_dir / "axon_grafting_points.h5",
#     )

#     morph_dir = Path("morphologies") / "repair_release" / "asc"
#     shutil.copytree(example_dir / morph_dir, testing_dir / morph_dir, ignore=_ignore_files)

#     inputs_result_dir = testing_dir / "inputs"
#     result = cli_runner.invoke(
#         axon_synthesis.cli.main,
#         [
#             "-c",
#             "config.cfg",
#             "create-inputs",
#             "--output-dir",
#             str(inputs_result_dir),
#             "--nb-workers",
#             str(nb_workers),
#         ],
#     )
#     assert result.exit_code == 0, result.output

#     synthesis_result_dir = testing_dir / "out"
#     result = cli_runner.invoke(
#         axon_synthesis.cli.main,
#         [
#             "-c",
#             "config.cfg",
#             "synthesize",
#             "--input-dir",
#             inputs_result_dir,
#             "--morphology-data-file",
#             testing_dir / "input_cells.mvd3",
#             "--axon-grafting-points-file",
#             testing_dir / "axon_grafting_points.h5",
#             "--output-dir",
#             synthesis_result_dir,
#             "--nb-workers",
#             str(nb_workers),
#             "--target-max-tries",
#             "100",
#         ],
#     )
#     assert result.exit_code == 0, result.output

#     # Check the results
#     comparators = get_comparators()
#     comparators[".h5"] = MorphologyComparator(default_diff_kwargs={"rtol": 1e-3, "atol": 1e-3})
#     comparators[".asc"] = MorphologyComparator(default_diff_kwargs={"rtol": 1e-3, "atol": 1e-3})
#     out_dir_pattern = (str(synthesis_result_dir) + "/?", "")
#     assert_equal_trees()


@pytest.mark.parametrize("nb_workers", [0, 2])
def test_mimic_example(testing_dir, data_dir, example_dir, cli_runner, nb_workers):
    """Test the mimic workflow from the general example with 2 morphologies."""
    shutil.copyfile(example_dir / "config.cfg", testing_dir / "config.cfg")

    morph_dir = Path("morphologies") / "repair_release" / "asc"
    shutil.copytree(example_dir / morph_dir, testing_dir / morph_dir, ignore=_ignore_files)

    result_dir = testing_dir / "out"
    result = cli_runner.invoke(
        axon_synthesis.cli.main,
        [
            "-c",
            "config.cfg",
            "validation",
            "mimic",
            "--output-dir",
            str(result_dir),
            "--nb-workers",
            str(nb_workers),
        ],
    )
    assert result.exit_code == 0, result.output

    # Check the results
    comparators = get_comparators()
    comparators[".h5"] = MorphologyComparator(default_diff_kwargs={"rtol": 1e-3, "atol": 1e-3})
    comparators[".asc"] = MorphologyComparator(default_diff_kwargs={"rtol": 1e-3, "atol": 1e-3})
    out_dir_pattern = (str(result_dir) + "/?", "")
    assert_equal_trees(
        data_dir / "mimic_example",
        testing_dir / "out",
        comparators=comparators,
        specific_args={
            "inputs/circuit.h5": {
                "comparator": CellCollectionComparator(),
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "morph_file",
                        ],
                    },
                },
            },
            "GraphCreationData data": {
                "patterns": [r"synthesis_basic_\S*/GraphCreationData/\S*\.h5$"],
                "comparator": HdfComparator(),
                "load_kwargs": {"key": "nodes"},
            },
            "SteinerTreeSolutions data": {
                "patterns": [r"synthesis_basic_\S*/SteinerTreeSolutions/\S*\.h5$"],
                "comparator": HdfComparator(),
                "load_kwargs": {"key": "solution_nodes"},
            },
            "inputs/metadata.json": {
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "clustering",
                            "path",
                            "morphology_path",
                            "WMR",
                        ],
                    },
                }
            },
            "inputs/Clustering/clustered_morphologies_paths.csv": {
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "morph_file",
                            "morph_path",
                        ],
                    },
                }
            },
            "inputs/Clustering/clustered_terminals.csv": {
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "morph_file",
                        ],
                    },
                }
            },
            "inputs/Clustering/trunk_properties.json": {
                "tolerance": 1e-4,
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "[*].morph_file",
                        ],
                    },
                },
            },
            "inputs/Clustering/tuft_properties.json": {
                "tolerance": 1e-4,
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "[*].morph_file",
                        ],
                    },
                },
            },
            "inputs/projection_probabilities.csv": {
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "morph_file",
                        ],
                    },
                }
            },
            "Target points": {
                "patterns": [r"synthesis_basic_\S*/target_points.h5"],
                "comparator": HdfComparator(),
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "morph_file",
                        ],
                    },
                },
                "atol": 1e-6,
            },
        },
    )
