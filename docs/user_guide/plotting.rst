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
       color_cols=["leiden", "sample", "condition"],
       basis="umap",
       figsize=(12, 4),
   )
   fig.savefig("embeddings.png", dpi=150)

   # Gene expression overlay
   fig = scutils.pl.embedding_gene_expression_multiplot(
       adata,
       genes=["GAPDH", "CD3E", "CD19"],
       basis="umap",
   )

Boxplots
--------

Compare feature distributions across groups using violin/box plots.

.. code-block:: python

   # Single feature, multiple groups
   fig = scutils.pl.plot_feature_boxplot(
       adata,
       feature="n_genes_by_counts",
       groupby="leiden",
   )

   # Multi-feature panel layout
   fig = scutils.pl.plot_feature_boxplot_multiplot(
       adata,
       features=["n_genes_by_counts", "total_counts", "pct_counts_mt"],
       groupby="sample",
   )

   # Aggregated per-group summary (mean ± std)
   fig = scutils.pl.plot_feature_boxplot_aggregated(
       adata,
       feature="CD3E",
       groupby="condition",
       aggregate_by="donor",
   )

Dotplots
--------

Show mean expression and fraction expressing for a gene set across
cell types.

.. code-block:: python

   marker_genes = ["CD3E", "CD19", "MS4A1", "GNLY", "NKG7"]

   fig = scutils.pl.dotplot_expression_two_categories(
       adata,
       genes=marker_genes,
       category_col="leiden",
       split_col="condition",
   )

Heatmaps
--------

Heatmaps of mean gene expression per cluster, optionally split by a
second variable.

.. code-block:: python

   fig = scutils.pl.heatmap_expression_two_categories(
       adata,
       genes=marker_genes,
       category_col="leiden",
       split_col="condition",
   )

Density Plots
-------------

KDE density outlines overlaid on an embedding — useful for comparing
the spatial distribution of cell populations across conditions.

.. code-block:: python

   fig = scutils.pl.plot_density_outlines(
       adata,
       group_col="condition",
       basis="umap",
   )

   # Multi-panel layout per area of interest
   fig = scutils.pl.aoi_density_outlines_multiplot(
       adata,
       group_col="condition",
       aoi_col="tissue_region",
   )

Volcano Plots
-------------

Visualise differential expression results.

.. code-block:: python

   fig = scutils.pl.volcano_plot(
       de_results,            # pd.DataFrame with logFC and p-value columns
       log2fc_col="logfoldchanges",
       pval_col="pvals_adj",
       gene_col="names",
       title="Condition A vs B",
   )
   fig.savefig("volcano.png", dpi=150)
