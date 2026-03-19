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
   )
   # result is stored in adata.obs["leiden_subclustered"]

Renaming Subcluster Labels
--------------------------

After iterative subclustering and manual annotation, use
:func:`~scutils.tools.rename_subcluster_labels` to replace numeric
sub-labels with biologically meaningful names.

.. code-block:: python

   # label_map: new_label -> [old_labels]
   label_map = {
       "CD4 T cell":        ["3_0"],
       "Regulatory T cell": ["3_1"],
       "NK cell":           ["7_0"],
       "NKT cell":          ["7_1"],
   }

   scutils.tl.rename_subcluster_labels(
       adata,
       col="leiden_subclustered",
       label_map=label_map,
   )
   # result is written back to adata.obs["leiden_subclustered"] in place

Spatial Cluster Splitting
--------------------------

:func:`~scutils.tools.spatial_split_clusters` divides clusters that
span spatially distinct regions (e.g. two tissue sections) into
separate sub-populations based on their spatial coordinates.

.. code-block:: python

   scutils.tl.spatial_split_clusters(
       adata,
       cluster_col="leiden",
       categories=["3", "7"],   # clusters to evaluate for spatial separation
       basis="X_umap",
   )
   # result is stored in adata.obs["leiden_spatial_split"]

Gene Scoring
------------

Two methods are available for scoring a gene set against every cell,
both working in place on ``adata.obs``.

Hotspot module scores
~~~~~~~~~~~~~~~~~~~~~

:func:`~scutils.tools.compute_hotspot_scores` computes a Hotspot
auto-correlation score for a set of genes based on their spatial
auto-correlation in a neighbourhood graph built from a low-dimensional
representation.

.. note::
   Requires the optional ``scoring`` extras:
   ``pip install 'scutils[scoring]'``

.. code-block:: python

   import scutils

   scutils.tl.compute_hotspot_scores(
       adata,
       genes=["SELL", "CCR7", "TCF7"],
       score_name="naive_T_score",
       use_rep="X_pca",
   )
   # Score is stored in adata.obs["naive_T_score"]

ULM transcription-factor activity scores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~scutils.tools.compute_ulm_scores` runs Univariate Linear Model
(ULM) from the :mod:`decoupler` package on each source (e.g.
transcription factor) in a prior knowledge network.  One activity column
per source is added to ``adata.obs`` under the prefix ``score_name``.

.. note::
   Requires the optional ``decoupler`` extra:
   ``pip install 'scutils[decoupler]'``

.. code-block:: python

   import pandas as pd
   import scutils

   # net must have source / target / weight columns
   net = pd.read_csv("collectri.csv")

   scutils.tl.compute_ulm_scores(
       adata,
       net=net,
       score_name="tf_activity",
       return_pvals=True,
   )
   # Each TF gets a column in adata.obs, e.g. adata.obs["tf_activity_STAT3"]
   # P-value matrix: adata.obsm["tf_activity_pvals"]

Diagnostic Plots
----------------

:func:`~scutils.tools.plot_spatial_split_diagnostics` produces a
figure showing the spatial distribution of clusters before and after
splitting — useful for validating the split boundaries.

.. code-block:: python

   fig = scutils.tl.plot_spatial_split_diagnostics(
       adata,
       cluster_col="leiden",
       categories=["3", "7"],   # clusters to inspect
       basis="X_umap",
   )
   fig.savefig("spatial_split_qc.png", dpi=150)

Functional Enrichment
---------------------

:func:`~scutils.tools.get_enriched_terms` wraps the g:Profiler API
(via :func:`scanpy.queries.enrich`) to run gene-set enrichment analysis
and returns a filtered, annotated results table ready for plotting.

.. code-block:: python

   import scutils

   # Retrieve top marker genes for a cluster
   markers = adata.uns["rank_genes_groups"]["names"]["3"].tolist()

   enrich_df = scutils.tl.get_enriched_terms(
       markers,
       sources=["GO:BP", "REAC", "KEGG"],
       pval_adjust_sign=0.05,
       min_term_size=20,
       max_term_size=500,
   )
   enrich_df.head()

   # Save results to disk
   scutils.tl.get_enriched_terms(
       markers,
       output_file="enrichment_results.csv",
   )
