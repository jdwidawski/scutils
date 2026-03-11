"""scutils — Single-cell RNA-seq utility functions.

Subpackages
-----------
plotting
    Visualisation utilities: embeddings, boxplots, dotplots, heatmaps,
    density plots, and volcano plots.
preprocessing
    Data loading, concatenation, and pre-processing helpers.
tools
    Analytical tools: clustering, subclustering, and spatial splitting.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("scutils")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["plotting", "preprocessing", "tools"]
