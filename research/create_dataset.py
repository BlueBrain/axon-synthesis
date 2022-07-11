"""Create the dataset that can be used for the Curate workflow from MPoW.

The workflow should be called using the luigi.cfg file from this directory and
"morphology-processing-workflow==0.0.5".
"""
import sys
from pathlib import Path

import pandas as pd


def create_dataset_for_repair(morph_dir, dataset_file="dataset.csv"):
    dataset = pd.DataFrame(columns=["morph_path", "mtype"])
    dataset.index.name = "morph_name"
    for morph in morph_dir.iterdir():
        if morph.suffix.lower() in [".asc", ".h5", ".swc"]:
            dataset.loc[morph.stem, "morph_path"] = morph
            dataset.loc[morph.stem, "mtype"] = "UNKOWN"
    dataset.sort_index(inplace=True)
    dataset.reset_index().to_csv(dataset_file, index=False)
    return dataset


def main(morph_dir, output_dataset):
    morph_dir = Path(morph_dir)
    output_dataset = Path(output_dataset)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    create_dataset_for_repair(morph_dir, output_dataset)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <morph_dir> <output_dataset>")
        exit(1)
    main(*sys.argv[1:])
