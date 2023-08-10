from luigi_tools.util import get_dependency_graph
from luigi_tools.util import graphviz_dependency_graph
from luigi_tools.util import render_dependency_graph

from axon_synthesis import workflows

deps = get_dependency_graph(workflows.CreateInputs())
graph = graphviz_dependency_graph(deps)
render_dependency_graph(graph, "CreateInputs_dependency_graph.png")

deps = get_dependency_graph(workflows.BuildTufts())
graph = graphviz_dependency_graph(deps)
render_dependency_graph(graph, "BuildTufts_dependency_graph.png")

deps = get_dependency_graph(workflows.Synthesis())
graph = graphviz_dependency_graph(deps)
render_dependency_graph(graph, "Synthesis_dependency_graph.png")
