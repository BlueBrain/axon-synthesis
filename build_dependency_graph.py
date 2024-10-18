# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of AxonSynthesis.
# See https://github.com/BlueBrain/AxonSynthesis for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""This script creates a diagram of the axon-synthesis workflow."""

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
