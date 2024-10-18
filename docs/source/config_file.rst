.. SPDX-License-Identifier: Apache-2.0

.. _config_file:

Configuration file
==================


The parameters can be stored in a configuration file which in then parsed by the CLI using the
``-c/--config`` argument. This configuration file is a ``*.cfg`` file and is organized in the
following way:

* the root items are only used by the entry point.
* the items in sections are used by the corresponding sub-command (e.g. the items of the
  ``[create-inputs]`` section are only used by the ``axon-synthesis create-inputs`` sub-command.

Global section
~~~~~~~~~~~~~~

The items of the ``[global]`` section can be used by all sub-commands. If an item is present in
both the ``[global]`` section and in a sub-section, the second one takes precedence. For example,
the following configuration:

.. code-block:: cfg

  [create-inputs]
  atlas_path = /the/atlas/path
  atlas_region_filename = brain_regions.nrrd

is equivalent to:

.. code-block:: cfg

  [global]
  path = /the/atlas/path
  atlas_region_filename = the_global_brain_regions.nrrd

  [create-inputs]
  atlas_region_filename = brain_regions.nrrd


Parameter groups
~~~~~~~~~~~~~~~~

Some parameters can be grouped in a sub-section and are then expanded as individual parameters.
For example, the following configuration:

.. code-block:: cfg

  [create-inputs]
  atlas_path = /the/atlas/path
  atlas_region_filename = brain_regions.nrrd

is equivalent to:

.. code-block:: cfg

  [create-inputs]

  [[atlas]]
  path = /the/atlas/path
  region_filename = brain_regions.nrrd


Example
~~~~~~~

A complete example is given in the ``examples`` directory that contains the following configuration
file:

.. literalinclude:: ../../examples/config.cfg
   :language: cfg
