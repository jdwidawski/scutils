scutils
=======

.. image:: https://img.shields.io/badge/python-3.11%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.11+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: MIT License

.. image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://github.com/astral-sh/ruff
   :alt: Code style: ruff

----

**scutils** is a Python package of high-level utility functions for
single-cell RNA-sequencing (scRNA-seq) analysis, built on top of
`Scanpy <https://scanpy.readthedocs.io>`_ and
`AnnData <https://anndata.readthedocs.io>`_.

Features
--------

🎨 **Plotting** (``scutils.pl``)
   Rich visualisation helpers — UMAP/PCA embeddings with multi-panel
   layouts, feature boxplots, dotplots, heatmaps, density outlines,
   and volcano plots.

🔧 **Preprocessing** (``scutils.pp``)
   Dataset concatenation with zero-filling for missing genes, filtering
   utilities, and summary statistics for multi-dataset workflows.

🧬 **Tools** (``scutils.tl``)
   Iterative subclustering, cluster label renaming, and spatial cluster
   splitting with diagnostic visualisations.

Quick Start
-----------

Install from source using `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   git clone https://github.com/jakub_widawski/single_cell_utilities.git
   cd single_cell_utilities
   uv sync

Or with pip:

.. code-block:: bash

   pip install .

Basic usage:

.. code-block:: python

   import scanpy as sc
   import scutils

   adata = sc.read_h5ad("my_data.h5ad")

   # Plotting — accessible via scutils.pl
   fig = scutils.pl.embedding_category_multiplot(adata, color_cols=["leiden", "sample"])

   # Preprocessing — accessible via scutils.pp
   combined = scutils.pp.concat_anndata_with_zeros([adata1, adata2], dataset_col="batch")

   # Tools — accessible via scutils.tl
   scutils.tl.iterative_subcluster(adata, cluster_col="leiden", subcluster_resolutions={"3": 0.4})

----

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/getting_started
   user_guide/plotting
   user_guide/preprocessing
   user_guide/tools

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/plotting
   api/preprocessing
   api/tools

.. toctree::
   :maxdepth: 1
   :caption: Developer Notes

   changelog
   deployment

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
