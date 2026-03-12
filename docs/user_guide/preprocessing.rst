Preprocessing (``scutils.pp``)
==============================

Overview
--------

The ``scutils.pp`` module provides helpers for assembling and
pre-processing multi-dataset single-cell experiments.  The main
challenge it addresses is **gene-set mismatch**: when datasets are
profiled on different gene panels, naïve concatenation drops genes
present in only a subset of datasets.  ``scutils.pp`` fills the
missing entries with zeros so that every dataset retains all genes.

.. note::
   Functions that modify an :class:`~anndata.AnnData` in place
   (e.g. those adding ``obs`` columns) return ``None``.  Functions
   that produce a new object return it explicitly.

Concatenating Datasets with Zero-Filling
-----------------------------------------

Use :func:`~scutils.preprocessing.concat_anndata_with_zeros` to
concatenate a list of :class:`~anndata.AnnData` objects, padding
missing genes with zeros.

.. code-block:: python

   import scutils

   # Concatenate two datasets; genes missing from one are filled with zeros
   combined = scutils.pp.concat_anndata_with_zeros(
       adata_a, adata_b,
       left_name="sample_A",
       right_name="sample_B",
   )
   print(combined.shape)       # (all_cells, union_of_genes)

   # Chain additional datasets one at a time
   combined = scutils.pp.concat_anndata_with_zeros(
       combined, adata_c,
       left_name=None,          # left is already a combined object
       right_name="sample_C",
   )

Inspecting Zero-Filling Statistics
------------------------------------

Before combining, it can be useful to understand which genes are
missing from each dataset.

.. code-block:: python

   # Which datasets are missing a specific gene?
   missing = scutils.pp.get_datasets_missing_gene(combined, gene="GAPDH")
   print(missing)

   # Summary table of zero-filled genes per dataset
   stats = scutils.pp.get_zero_filling_stats(combined)
   scutils.pp.print_zero_filling_summary(combined)

Filtering Genes by Presence
-----------------------------

Remove genes that are absent from too many datasets — useful for
reducing noise from panel-specific genes.

.. code-block:: python

   # Keep only genes present in at least 3 datasets
   filtered = scutils.pp.filter_genes_by_presence(
       combined,
       min_datasets=3,
   )
