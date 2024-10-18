.. SPDX-License-Identifier: Apache-2.0

.. _cli:

Command Line Interface
======================

Entry point and global parameters
---------------------------------

.. click:: axon_synthesis.cli:main
  :prog: axon-synthesis

Sub-commands
------------

.. _cli_wmr:

.. click:: axon_synthesis.cli.input_creation:fetch_white_matter_recipe
  :prog: axon-synthesis fetch-white-matter-recipe
  :nested: full

.. _cli_create_inputs:

.. click:: axon_synthesis.cli.input_creation:create_inputs
  :prog: axon-synthesis create-inputs
  :nested: full

.. _cli_synthesis:

.. click:: axon_synthesis.cli.synthesis:synthesize
  :prog: axon-synthesis synthesize
  :nested: full

Validation sub-commands
-----------------------

.. _cli_validation_mimic:

.. click:: axon_synthesis.cli.validation:mimic
  :prog: axon-synthesis validation mimic
  :nested: full
