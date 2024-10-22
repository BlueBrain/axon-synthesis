.. SPDX-License-Identifier: Apache-2.0

.. _input_files:

Input files
===========

The synthesis workflow requires several input files and can optionally use a few other ones.

.. _inputs_circuit:

Circuit file (required)
-----------------------

The first required file is a circuit file, as described here: .
It's a HDF5 (or MVD3 but is is deprecated) file that must
contain the following columns:

* ``morphology``: the name of the morphology.
* ``mtype``: the mtype of the morphology.
* ``x``, ``y`` and ``z``: the coordinates of the soma center of the morphology.
* ``orientation``: the orientation of the morphology.

.. _inputs_axon_grafting_points:

Axon grafting points (optional)
-------------------------------

It is possible to provide a HDF5 file that specify where each synthesized axon should be grafted
on the morphology. The file should contain the following columns:

* morphology: the name of the morphology.
* grafting_section_id: the mtype of the morphology.
* source_x, source_y and source_z: the coordinates of the first point of the synthesized axon (if
  these columns are omitted the coordinates will be computed from the section IDs but it may take
  some time).
* population_id: the ID of the source population used for the axon (if this column is not provided
  it will be computed from the atlas).

If other columns are present they will be ignored.

.. _inputs_tuft_parameters:

Tuft parameters (optional)
--------------------------

It is possible to create a ``tuft_parameters.json`` file in the input directory to store the
parameter values used during the tuft generation. The schema and an example of the format of this
file is described in :external+neurots:ref:`params`.
Note that this file should not contain the ``basal_dendrite`` and ``apical_dendrite`` entries and
that the ``axon/orientation/values/orientations`` entry will be overridden during the tuft
generation.

.. _inputs_tuft_distributions:

Tuft distributions (optional)
-----------------------------

It is possible to create a ``tuft_distributions.json`` file in the input directory to store the
distribution values used during the tuft generation. The schema and an example of the format of this
file is described in :external+neurots:ref:`distrs`.
Note that this file should not contain the ``basal_dendrite`` and ``apical_dendrite`` entries and
that the ``axon/persistence_diagram`` entry will be overridden during the tuft generation.

.. _inputs_input_folder:

Input folder (required)
-----------------------

A folder containing several files is also needed. Usually, this folder should be created using the
:ref:`'axon-synthesis create-inputs' <cli_create_inputs>` command but it's possible to create these
files manually. In this case, one should create a folder containing the following files:

* ``Clustering/trunk_properties.json``: a JSON file containing the set of trunk properties.
* ``Clustering/tuft_properties.json``: a JSON file containing the set of tuft properties.
* ``population_probabilities.json``: a JSON file containing the probabilities to pick each population.
* ``projection_probabilities.json``: a JSON file containing the probabilities to pick each projection.
* ``region_masks.h5``: This file is optional and will be automatically computed if an atlas is
  provided to the ``synthesis`` workflow.
