# Configuration file for the Sphinx documentation builder.
#
# Full list of configuration options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "scutils"
author = "Jakub Widawski"
copyright = "2026, Jakub Widawski"

try:
    import importlib.metadata
    release = importlib.metadata.version("scutils")
except Exception:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "installation.md", "quickstart.md"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Napoleon settings — support both Google and NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# Autodoc
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Mock heavy optional imports so autodoc works without a full environment
autodoc_mock_imports = ["hdbscan"]

# Intersphinx: links to upstream libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
