"""scutils — Single-cell RNA-seq utility functions.

Subpackages
-----------
plotting (pl)
    Visualisation utilities: embeddings, boxplots, dotplots, heatmaps,
    density plots, and volcano plots.
preprocessing (pp)
    Data loading, concatenation, and pre-processing helpers.
tools (tl)
    Analytical tools: clustering, subclustering, and spatial splitting.

Shortcuts
---------
``scutils.pl``, ``scutils.pp``, and ``scutils.tl`` are aliases for the
three subpackages, mirroring the Scanpy convention.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("scutils")
except PackageNotFoundError:
    __version__ = "unknown"

from scutils import plotting as pl
from scutils import preprocessing as pp
from scutils import tools as tl

__all__ = ["plotting", "preprocessing", "tools", "pl", "pp", "tl"]
