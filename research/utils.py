import re

import networkx as nx
import pandas as pd
from neurom import NeuriteType


def add_camera_sync(fig_path):
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


def get_axons(morph):
    return [i for i in morph.neurites if i.type == NeuriteType.axon]


def neurite_to_graph(neurite, graph_cls=nx.DiGraph, **graph_kwargs):
    graph_nodes = []
    graph_edges = []
    for section in neurite.iter_sections():
        is_terminal = not bool(section.children)
        if section.parent is None:
            graph_nodes.append((-1, *section.points[0, :3], True))
            graph_edges.append((-1, section.id))

        graph_nodes.append((section.id, *section.points[-1, :3], is_terminal))

        for child in section.children:
            graph_edges.append((section.id, child.id))

    nodes = pd.DataFrame(graph_nodes, columns=["id", "x", "y", "z", "is_terminal"])
    nodes.set_index("id", inplace=True)

    edges = pd.DataFrame(graph_edges, columns=["source", "target"])
    graph = nx.from_pandas_edgelist(edges, create_using=graph_cls, **graph_kwargs)
    nx.set_node_attributes(graph, nodes[["x", "y", "z", "is_terminal"]].to_dict("index"))

    return nodes, edges, graph


def append_section_recursive(source, target):

    current_sections = [(source, target)]

    # Add kept sections
    while current_sections:
        source_parent, target_child = current_sections.pop()
        source_child = source_parent.append_section(target_child)
        for child in target_child.children:
            current_sections.append((source_child, child))
        #     if child.id in kept_path:
        #         new_section = PointLevel(
        #             child.points[:, COLS.XYZ].tolist(),
        #             (child.points[:, COLS.R] * 2).tolist(),
        #         )
        #         current_sections.append(
        #             (current_child, current_child.append_section(new_section))
        #         )

        # if current_parent.id in sections_to_add:
        #     for new_sec in sections_to_add[current_parent.id]:
        #         current_child.append_section(new_sec)
