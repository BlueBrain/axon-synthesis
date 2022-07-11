"""Plot the Steiner Tree solutions."""
import logging
import re
from pathlib import Path

import luigi
import luigi_tools
import pandas as pd
import plotly.graph_objects as go
from neurom import load_neuron
from plotly.subplots import make_subplots
from plotly_helper.neuron_viewer import NeuronBuilder

from PCSF.steiner_tree import SteinerTree

logger = logging.getLogger(__name__)


class PlotSolutions(luigi_tools.task.WorkflowTask):
    nodes_path = luigi.Parameter(description="Path to the nodes CSV file.", default=None)
    edges_path = luigi.Parameter(description="Path to the edges CSV file.", default=None)
    output_dir = luigi.Parameter(
        description="Output folder for figures.", default="steiner_solutions"
    )

    def requires(self):
        return SteinerTree()

    def run(self):
        nodes = pd.read_csv(self.nodes_path or self.input()["nodes"].path)
        edges = pd.read_csv(self.edges_path or self.input()["edges"].path)
        output_dir = Path(self.output().path)
        output_dir.mkdir(parents=True, exist_ok=True)

        node_groups = nodes.groupby("morph_file")
        edge_groups = edges.groupby("morph_file")

        group_names = node_groups.groups.keys()
        assert set(group_names) == set(
            edge_groups.groups.keys()
        ), "The nodes and edges have different 'morph_file' entries"

        for group_name in group_names:
            group_nodes = node_groups.get_group(group_name)
            group_edges = edge_groups.get_group(group_name)

            logger.debug(f"{group_name}: {len(group_nodes)} nodes and {len(group_edges)} edges")

            # Build the neuron figure
            neuron = load_neuron(group_name)
            builder = NeuronBuilder(neuron, "3d", line_width=4, title=f"{group_name}")
            neuron_fig = builder.get_figure()

            # Build the solution figure
            node_trace = go.Scatter3d(
                x=group_nodes.loc[group_nodes["is_terminal"], "x"].values,
                y=group_nodes.loc[group_nodes["is_terminal"], "y"].values,
                z=group_nodes.loc[group_nodes["is_terminal"], "z"].values,
                mode="markers",
                name="markers",
                scene="scene2",
                marker={"size": 2},
            )

            group_edges_solution = group_edges.loc[group_edges["is_solution"]]
            edges_x = group_edges_solution[["x_from", "x_to"]].copy()
            edges_x["breaker"] = None
            edges_y = group_edges_solution[["y_from", "y_to"]].copy()
            edges_y["breaker"] = None
            edges_z = group_edges_solution[["z_from", "z_to"]].copy()
            edges_z["breaker"] = None

            edge_trace = go.Scatter3d(
                x=edges_x.values.flatten().tolist(),
                y=edges_y.values.flatten().tolist(),
                z=edges_z.values.flatten().tolist(),
                hoverinfo="none",
                mode="lines",
                scene="scene2",
            )

            # Export the solution
            fig = make_subplots(cols=2, specs=[[{"is_3d": True}, {"is_3d": True}]])
            fig.add_traces([node_trace, edge_trace], rows=[1, 1], cols=[1, 1])
            fig.add_traces(
                neuron_fig["data"],
                rows=[1] * len(neuron_fig["data"]),
                cols=[2] * len(neuron_fig["data"]),
            )
            fig_path = str((output_dir / Path(group_name).name).with_suffix(".html"))
            fig.write_html(fig_path)

            # Update the HTML file to synchronize the cameras between the two plots
            with open(fig_path) as f:
                tmp = f.read()
                fig_id = re.match('.*id="([^ ]*)" .*', tmp, flags=re.DOTALL).group(1)

            js = """
    <script>
    var gd = document.getElementById('{fig_id}');
    var isUnderRelayout = false

    gd.on('plotly_relayout', () => {{
      console.log('relayout', isUnderRelayout)
      if (!isUnderRelayout) {{
        Plotly.relayout(gd, 'scene2.camera', gd.layout.scene.camera)
          .then(() => {{ isUnderRelayout = false }}  )
      }}

      isUnderRelayout = true;
    }})
    </script>
    """.format(
                fig_id=fig_id
            )

            with open(fig_path, "w") as f:
                f.write(tmp.replace("</body>", js + "</body>"))

            logger.info(f"{group_name}: exported to {fig_path}")

    def output(self):
        return luigi_tools.target.OutputLocalTarget(self.output_dir)
