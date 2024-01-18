"""Some plot utils for create graph."""

from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from axon_synthesis.utils import add_camera_sync


def plot_final_morph(morph, output_path, initial_morph=None, logger=None):
    """Plot the given morphology.

    If `initial_morph` is not None then the given morphology is also plotted for comparison.
    """
    title = "Final morphology"
    fig_builder = NeuronBuilder(morph, "3d", line_width=4, title=title)
    fig_data = [fig_builder.get_figure()["data"]]
    left_title = "Synthesized morphology"

    if initial_morph is not None:
        raw_builder = NeuronBuilder(initial_morph, "3d", line_width=4, title=title)

        fig = make_subplots(
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=[left_title, "Initial morphology"],
        )
        fig_data.append(raw_builder.get_figure()["data"])
    else:
        fig = make_subplots(cols=1, specs=[[{"type": "scene"}]], subplot_titles=[left_title])

    for col_num, data in enumerate(fig_data):
        fig.add_traces(data, rows=[1] * len(data), cols=[col_num + 1] * len(data))

    fig.update_scenes({"aspectmode": "data"})

    fig.layout.update(title=morph.name)

    # Export figure
    fig.write_html(output_path)

    if initial_morph is not None:
        add_camera_sync(output_path)

    if logger is not None:
        logger.info("Exported figure to %s", output_path)
