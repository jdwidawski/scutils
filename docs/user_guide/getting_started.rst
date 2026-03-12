Getting Started
===============

Prerequisites
-------------

- Python ≥ 3.11
- `uv <https://github.com/astral-sh/uv>`_ (recommended) or pip

Installation
------------

**From source with uv (recommended)**

.. code-block:: bash

   git clone https://github.com/jakub_widawski/single_cell_utilities.git
   cd single_cell_utilities

   # Install runtime dependencies + package in editable mode
   uv sync

   # Also install dev dependencies (pytest, ruff)
   uv sync --extra dev

   # Also install documentation dependencies (sphinx, rtd-theme)
   uv sync --extra docs

**From source with pip**

.. code-block:: bash

   git clone https://github.com/jakub_widawski/single_cell_utilities.git
   cd single_cell_utilities
   pip install -e .

   # With optional groups
   pip install -e ".[dev,docs]"

Verifying the Installation
--------------------------

.. code-block:: python

   import scutils

   print(scutils.__version__)   # e.g. "0.1.0"

Package Structure
-----------------

``scutils`` mirrors the `Scanpy <https://scanpy.readthedocs.io>`_ API
conventions.  The three subpackages can be accessed via shortcuts:

.. code-block:: python

   import scutils

   scutils.pl   # plotting  — scutils.plotting
   scutils.pp   # preprocessing — scutils.preprocessing
   scutils.tl   # tools — scutils.tools

You can also import functions directly:

.. code-block:: python

   from scutils.plotting import embedding_category_multiplot
   from scutils.preprocessing import concat_anndata_with_zeros
   from scutils.tools import iterative_subcluster

Running Tests
-------------

.. code-block:: bash

   uv run pytest
   uv run pytest tests/plotting/ -q           # run a specific subset
   uv run pytest --cov=scutils --cov-report=html  # with coverage

Building the Docs
-----------------

.. code-block:: bash

   uv run sphinx-build docs/ docs/_build/html
   # or
   cd docs && make html

Open ``docs/_build/html/index.html`` in a browser to preview.
