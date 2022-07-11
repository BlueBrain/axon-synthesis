import re

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
