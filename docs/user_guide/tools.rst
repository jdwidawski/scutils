Tools (``scutils.tl``)
======================

Overview
--------

The ``scutils.tl`` module provides analytical tools for refining and
extending Scanpy clustering results.  All functions operate in place
on an :class:`~anndata.AnnData` object and return ``None`` unless
they produce a diagnostic figure.

.. note::
   Run :func:`scanpy.pp.neighbors` and a base clustering (e.g.
   :func:`scanpy.tl.leiden`) before using the tools below.

Iterative Subclustering
-----------------------

:func:`~scutils.tools.iterative_subcluster` re-clusters individual
clusters at a higher resolution, then merges the results back into
a single column.  This is useful for resolving heterogeneous clusters
that contain multiple cell types.

.. code-block:: python

   import scanpy as sc
   import scutils

   sc.pp.neighbors(adata)
   sc.tl.leiden(adata, key_added="leiden", resolution=0.5)

   # Re-cluster clusters "3" and "7" at higher resolution
   scutils.tl.iterative_subcluster(
       adata,
       cluster_col="leiden",
       subcluster_resolutions={"3": 0.6, "7": 0.4},
       key_added="leiden_sub",
   )
   # adata.obs["leiden_sub"] now contains the merged labels

Renaming Subcluster Labels
--------------------------

After iterative subclustering and manual annotation, use
:func:`~scutils.tools.rename_subcluster_labels` to replace numeric
sub-labels with biologically meaningful names.

.. code-block:: python

   label_map = {
       "3_0": "CD4 T cell",
       "3_1": "Regulatory T cell",
       "7_0": "NK cell",
       "7_1": "NKT cell",
   }

   scutils.tl.rename_subcluster_labels(
       adata,
       cluster_col="leiden_sub",
       label_map=label_map,
       key_added="cell_type",
   )

Spatial Cluster Splitting
--------------------------

:func:`~scutils.tools.spatial_split_clusters` divides clusters that
span spatially distinct regions (e.g. two tissue sections) into
separate sub-populations based on their spatial coordinates.

.. code-block:: python

   scutils.tl.spatial_split_clusters(
       adata,
       cluster_col="leiden",
       spatial_key="spatial",
       key_added="leiden_spatial",
   )

Diagnostic Plots
----------------

:func:`~scutils.tools.plot_spatial_split_diagnostics` produces a
figure showing the spatial distribution of clusters before and after
splitting — useful for validating the split boundaries.

.. code-block:: python

   fig = scutils.tl.plot_spatial_split_diagnostics(
       adata,
       cluster_col="leiden",
       spatial_key="spatial",
   )
   fig.savefig("spatial_split_qc.png", dpi=150)
