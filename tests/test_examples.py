"""Test the examples."""
import shutil
from pathlib import Path

import dir_content_diff.pandas
import pytest
from dir_content_diff import assert_equal_trees
from dir_content_diff import get_comparators
from dir_content_diff_plugins.morphologies import MorphologyComparator
from dir_content_diff_plugins.voxcell import CellCollectionComparator

import axon_synthesis.cli


def _ignore_files(_src, names) -> list[str]:
    """Filter some files."""
    return [i for i in names if Path(i).name not in ["172992.asc", "172993.asc"]]


@pytest.mark.parametrize("nb_workers", [0, 1, 2])
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
    comparators[".h5"] = MorphologyComparator()
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
                "patterns": [r"synthesis/GraphCreationData/\S*\.h5$"],
                "comparator": dir_content_diff.pandas.HdfComparator(),
                "load_kwargs": {"key": "nodes"},
            },
            "SteinerTreeSolutions data": {
                "patterns": [r"synthesis/SteinerTreeSolutions/\S*\.h5$"],
                "comparator": dir_content_diff.pandas.HdfComparator(),
                "load_kwargs": {"key": "solution_nodes"},
            },
            "inputs/metadata.json": {
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "clustering",
                            "path",
                            "WMR",
                        ],
                    },
                }
            },
            "inputs/Clustering/clustered_morphologies_paths.csv": {
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "morph_path",
                        ],
                    },
                }
            },
            "inputs/Clustering/trunk_properties.json": {
                "tolerance": 1e-4,
            },
            "inputs/Clustering/tuft_properties.json": {
                "tolerance": 1e-4,
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
            "synthesis/target_points.h5": {
                "comparator": dir_content_diff.pandas.HdfComparator(),
                "format_data_kwargs": {
                    "replace_pattern": {
                        out_dir_pattern: [
                            "morph_file",
                        ],
                    },
                },
            },
        },
    )