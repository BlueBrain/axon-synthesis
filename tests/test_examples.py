"""Test the examples."""
import shutil
from pathlib import Path

import dir_content_diff.pandas
from dir_content_diff import DefaultComparator
from dir_content_diff import assert_equal_trees
from dir_content_diff import get_comparators
from dir_content_diff_plugins.morphologies import MorphologyComparator

import axon_synthesis.cli


def _ignore_files(_src, names):
    """Filter some files."""
    return [i for i in names if Path(i).name not in ["172992.asc", "172993.asc"]]


def test_mimic_example(testing_dir, data_dir, example_dir, cli_runner):
    """Test the mimic workflow from the general example with 2 morphologies."""
    shutil.copyfile(example_dir / "config.cfg", testing_dir / "config.cfg")

    morph_dir = Path("morphologies") / "repair_release" / "asc"
    shutil.copytree(example_dir / morph_dir, testing_dir / morph_dir, ignore=_ignore_files)

    result = cli_runner.invoke(
        axon_synthesis.cli.main,
        ["-c", "config.cfg", "validation", "mimic", "--output-dir", str(testing_dir / "out")],
    )
    assert result.exit_code == 0, result.output

    # Check the results
    comparators = get_comparators()
    comparators[".h5"] = MorphologyComparator()
    assert_equal_trees(
        data_dir / "mimic_example",
        testing_dir / "out",
        comparators=comparators,
        specific_args={
            "inputs/circuit.h5": {
                "comparator": DefaultComparator(),
            },
            "synthesis/GraphCreationData/172992_0.h5": {
                "comparator": dir_content_diff.pandas.HdfComparator(),
                "load_kwargs": {"key": "nodes"},
            },
            "synthesis/GraphCreationData/172993_0.h5": {
                "comparator": dir_content_diff.pandas.HdfComparator(),
                "load_kwargs": {"key": "nodes"},
            },
            "synthesis/SteinerTreeSolutions/172992_0.h5": {
                "comparator": dir_content_diff.pandas.HdfComparator(),
                "load_kwargs": {"key": "solution_nodes"},
            },
            "synthesis/SteinerTreeSolutions/172993_0.h5": {
                "comparator": dir_content_diff.pandas.HdfComparator(),
                "load_kwargs": {"key": "solution_nodes"},
            },
            "synthesis/target_points.h5": {
                "comparator": dir_content_diff.pandas.HdfComparator(),
            },
        },
    )
