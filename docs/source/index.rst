.. SPDX-License-Identifier: Apache-2.0

Axon Synthesis
==============

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-line: 1

.. toctree::
   :hidden:
   :maxdepth: 2

   Home <self>
   general_principles
   cli
   config_file
   input_files
   api_ref
   changelog
   contributing


Workflows
---------

The ``axon-synthesis`` tool contains 2 main workflows:

* one to create and format the inputs required to actually synthesizing long-range axons.
* one to synthesize the long-range axons.

Create inputs workflow
~~~~~~~~~~~~~~~~~~~~~~

.. graphviz::

   digraph {
      "Atlas" -> "Brain region masks";
      "White Matter Recipe" -> "Probabilities";
      "Morphologies" -> "Clustering";
      "Clustering" -> "Tuft properties";
      "Clustering" -> "Main trunk properties";
      "Tuft properties" -> "Synthesis inputs"
      "Main trunk properties" -> "Synthesis inputs"
      "Brain region masks" -> "Synthesis inputs"
      "Probabilities" -> "Synthesis inputs"
   }

Synthesize axons workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

.. graphviz::

   digraph {
      "Inputs" -> "Source points";
      "Inputs" -> "Target points";
      "Source points" -> "3D graph";
      "Target points" -> "3D graph";
      "3D graph" -> "Steiner tree";
      "Steiner tree" -> "Main trunk";
      "Inputs" -> "Tufts";
      "Main trunk" -> "Tufts";
      "Tufts" -> "Morphology";
   }

See :ref:`the CLI page <cli>` for details on how to run each workflow.
