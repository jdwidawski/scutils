Plotting (``scutils.pl``)
=========================

Overview
--------

The ``scutils.pl`` module provides high-level plotting utilities for
single-cell RNA-seq data.  All functions accept an
:class:`~anndata.AnnData` object and return a
:class:`~matplotlib.figure.Figure`; they never call ``plt.show()``
internally, leaving display control to the caller.

All plotting functions share these conventions:

- **``figsize``** — explicit ``(width, height)`` in inches.
- **``palette``** — accepts a single colour string, a list of colours,
  a Matplotlib colormap name, or ``None`` (falls back to
  ``adata.uns["{col}_colors"]`` when available).
- **``**kwargs``** — forwarded to the underlying Scanpy or Matplotlib
  call where applicable.

Embedding Plots
---------------

Visualise cells in a 2-D embedding (UMAP, PCA, …) coloured by
categorical or continuous variables.

.. code-block:: python

   import scutils

   # Single-category multiplot (one panel per category value)
   fig = scutils.pl.embedding_category_multiplot(
       adata,
       column="leiden",   # one panel per leiden cluster value
       basis="umap",
       figsize=(4, 4),
   )
   fig.savefig("embeddings.png", dpi=150)

   # Gene expression overlay split by a categorical column
   fig = scutils.pl.embedding_gene_expression_multiplot(
       adata,
       column="leiden",   # one panel per leiden cluster value
       feature="GAPDH",   # colour by this gene
       basis="umap",
   )

Boxplots
--------

Compare feature distributions across groups using violin/box plots.

.. code-block:: python

   # Single feature grouped by a categorical column
   fig = scutils.pl.plot_feature_boxplot(
       adata,
       feature="n_genes_by_counts",
       x="leiden",
   )

   # One panel per x-category value (splits the x axis into panels)
   fig = scutils.pl.plot_feature_boxplot_multiplot(
       adata,
       feature="n_genes_by_counts",
       x="sample",
   )

   # Pseudo-bulk: one data point per sample, grouped by condition
   fig = scutils.pl.plot_feature_boxplot_aggregated(
       adata,
       feature="CD3E",
       x="condition",
       sample_col="donor",
   )

Dotplots
--------

Show mean expression and fraction expressing for a gene set across
cell types.

.. code-block:: python

   # Each function shows one gene across two categorical axes
   fig = scutils.pl.dotplot_expression_two_categories(
       adata,
       feature="CD3E",
       category_x="leiden",
       category_y="condition",
   )

Heatmaps
--------

Heatmaps of mean gene expression per cluster, optionally split by a
second variable.

.. code-block:: python

   fig = scutils.pl.heatmap_expression_two_categories(
       adata,
       feature="CD3E",
       category_x="leiden",
       category_y="condition",
   )

Density Plots
-------------

KDE density outlines overlaid on an embedding — useful for comparing
the spatial distribution of cell populations across conditions.

.. code-block:: python

   # Density outlines require pre-computed embedding density:
   # sc.tl.embedding_density(adata, basis="umap", groupby="condition")
   fig = scutils.pl.plot_density_outlines(
       adata,
       category_dict={"condition": ["control", "treated"]},
       basis="umap",
   )

   # Multi-panel layout — one panel per AOI column value
   fig = scutils.pl.aoi_density_outlines_multiplot(
       adata,
       category_dict={"condition": ["control", "treated"]},
       column="tissue_region",
   )

Volcano Plots
-------------

Visualise differential expression results.

.. code-block:: python

   # de_results: pd.DataFrame with gene names in a "names" column or index
   fig = scutils.pl.volcano_plot(
       de_results,
       lfc_col="logfoldchanges",
       pval_col="pvals_adj",
       pval_cutoff=0.05,
       lfc_cutoff=1.0,
   )
   fig.savefig("volcano.png", dpi=150)
