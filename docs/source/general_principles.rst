.. SPDX-License-Identifier: Apache-2.0

.. _synthesis_general_principles:

General principles of the synthesis algorithm
=============================================

.. _synthesis_source_target_pts:

Computing the source and target points of an axon
-------------------------------------------------

The long-range axon synthesis algorithm starts by collecting the given morphologies from the given
cell collection file and the morphology directory. Then the starting point of each axon are
computed. These starting points are called the **source points** of the axons and can be given
explicitly using the :ref:`axon grafting points file<inputs_axon_grafting_points>` (see also
:ref:`the related CLI argument <cli_synthesis>`).

The next step is to associate a **source population** to each long-range axon. These populations
can also be arbitrarily associated to each axon using the
:ref:`axon grafting points file<inputs_axon_grafting_points>`. If they are not given, the brain
region of the atlas in which the cell is located is computed. Then a source population is randomly
picked among the ones associated to this brain region (the probabilities are given in the
``population_probabilities.json`` file of the :ref:`input folder<inputs_input_folder>`).

After that, the **target populations** of each axon are computed. The probabilities of each target
population depending on the source population are given in the ``projection_probabilities.json``
file of the :ref:`input folder<inputs_input_folder>`).

Once the target populations have been chosen, a voxell is randomly chosen in the brain region
associated to each target population. Then a random shift is applied inside this voxel. This
final location is called a **target point** of the axon.

Here is a schema of the data workflow to generate the source and target points used for synthesis:

.. graphviz::

    digraph {
    	{
    		atlas [label="atlas"]
    		soma_center [label="soma center"]
    		axon_grafting_pts [label="axon grafting points", style=dotted]
    		pop_probs [label="population probabilities"]
    		source_pop [label="source population"]
    		source_pt [label="source point", shape=box, style=rounded, color=blue]
    		proj_probs [label="projection probabilities"]
    		target_pops [label="target populations"]
    		target_pts [label="target points", shape=box, style=rounded, color=blue]
    		synthesis [label="Synthesis", shape=box, color=red]
    	}
   	    soma_center -> source_pop;
   	    atlas -> source_pop;
        pop_probs -> source_pop;
   	    soma_center -> source_pt;
   	    axon_grafting_pts -> source_pt [style=dotted];
   	    axon_grafting_pts -> source_pop [style=dotted];
        source_pop -> target_pops;
        proj_probs -> target_pops;
        target_pops -> target_pts;
        atlas -> target_pts;
        source_pt -> synthesis
        target_pts -> synthesis
   	}


.. _synthesis_main_trunk:

Synthesizing the main trunk of an axon
--------------------------------------

The main trunk of the axon is the part the connects the source point to all the target points. The
synthesis of this main trunk will be performed using a Steiner Tree algorithm. This algorithm is
NP-complete, which means that is can be very long to compute a solution. In our case, we suppose
that a reasonable approximation is sufficient (the code
`pcst_fast <https://github.com/fraenkel-lab/pcst_fast>`_ will be used for this task).

In order to use this approximation, the first step is to transform the problem in the Euclidean
space into a graph problem. To do this, some intermediate points are created between the source and
target points in order to refine the future graph. Then, for the same reason, some random points
are created in the 3D bounding box containing the source and target points. After that, the Voronoï
points of all the previous points are created, in order to add points with more realistic angles
because the Voronoï points have angle properties close to the ones of the Steiner points. Once all
the points are created, a triangulation is computed to create the edges that connect them.

The edges used to compute the Steiner Tree must have positive weights. So the first step is to
consider that each edge has a weight equal to its length in the Euclidean space. Then several
optional penalties can be applied to these weights in order to guide the Steiner Tree algorithm.
Here are the possible penalties:

* a penalty to increase the weights if the edge is not in the radial direction of the source point.
  This is to ensure that the main trunk does not deviate too much from the target point directions.
* a penalty to increase the weights if the edge direction is not perpendicular to the depths fields
  direction. This is to ensure that the main trunk follows the shape of the atlas layers when it is
  possible.
* a penalty to decrease the weights if the edge is located in a set of specific brain regions. This
  is to guide the main trunk in the preferred regions for axonal development, like the fiber tracts
  for example.

Once the edges all have a weight, the Steiner Tree algorithm is computed in order to selected the
edges that connect the source point to all the target points using the shortest possible paths.

.. image:: scripts/graph_creation_solution.png

The final step to build the main trunk is to perform a guided random walk that follows the edges
selected by the Steiner Tree algorithm. The random walk is tinkered in a way that ensures that the
final result of this step is a main trunk whose morphometrics are realistic.


.. _preferred_regions:

Considering preferred regions
-----------------------------

It is possible to make the main trunk prefer some regions during the Steiner Tree process. To do
this, the preferred regions are modeled by attractor points positioned in space. These attractors
are used to update the weights of the edges that are used by the Steiner Tree algorithm to compute
the solution which minimizes the total weight of the edges selected in the solution. In this case,
the edge weights will decrease as their distance to the attractor decreases, making them more
likely to be selected by the Stein Tree algorithm.

.. image:: scripts/graph_creation_solution_preferred_regions.png


.. _synthesis_tufts:

Synthesizing the tufts of an axon
---------------------------------

Once the main trunk of the axon has been synthesized, a **tuft** is grown from each target point.
These tufts aims at connecting the axon to the dentrites surrounding the target points. The
generation of these tufts has twow steps:

* a barcode representing the future tuft is randomly chosen among the possible ones, based one the
  target population (these barcodes are given in the ``Clustering/tuft_properties.json`` file of
  the :ref:`input folder<inputs_input_folder>`).
* the tuft is grown based on this barcode using the
  `NeuroTS code <https://neurots.readthedocs.io>`_.
