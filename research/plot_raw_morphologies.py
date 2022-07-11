"""Plot the repaired raw morphologies."""
import sys
from pathlib import Path

from neurom import load_neuron
from plotly_helper.neuron_viewer import NeuronBuilder


def plot_morphology(morph_path, output_dir, verbose=True):
    if verbose:
        print(f"Plot '{morph_path}' to '{output_dir}'")
    input_file = Path(morph_path)
    neuron = load_neuron(morph_path)
    builder = NeuronBuilder(neuron, "3d", line_width=4, title=f"{input_file.name}")

    output_filename = Path(output_dir) / input_file.name
    builder.plot(filename=str(output_filename.with_suffix(".html")), auto_open=False)


def main(morph_dir, output_dir):
    morph_dir = Path(morph_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in morph_dir.iterdir():
        plot_morphology(file, output_dir)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <morph_dir> <output_dir>")
        exit(1)
    main(*sys.argv[1:])
