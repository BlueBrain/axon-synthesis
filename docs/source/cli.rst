.. _cli:

Command Line Interface
======================

Entry point and global parameters
---------------------------------

.. click:: axon_synthesis.cli:main
  :prog: axon-synthesis

Sub-commands
------------

.. _wmr:

.. click:: axon_synthesis.cli.input_creation:fetch_white_matter_recipe
  :prog: axon-synthesis fetch-white-matter-recipe
  :nested: full

.. _create_inputs:

.. click:: axon_synthesis.cli.input_creation:create_inputs
  :prog: axon-synthesis create-inputs
  :nested: full

.. _synthesis:

.. click:: axon_synthesis.cli.synthesis:synthesize
  :prog: axon-synthesis synthesize
  :nested: full

Validation sub-commands
-----------------------

.. _validation_mimic:

.. click:: axon_synthesis.cli.validation:mimic
  :prog: axon-synthesis validation mimic
  :nested: full
