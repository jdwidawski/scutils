# Configuration file for the Sphinx documentation builder.
#
# Full list of configuration options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "scutils"
author = "scutils contributors"
try:
    release = importlib.metadata.version("scutils")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",   # Google / NumPy docstring parsing
    "sphinx.ext.viewcode",   # [source] links in API docs
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",           # Markdown source files
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autosummary: generate stub files automatically
autosummary_generate = True

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# Autodoc
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Intersphinx: links to upstream libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
}
