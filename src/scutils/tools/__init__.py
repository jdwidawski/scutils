"""Analytical tools for single-cell data."""

from scutils.tools.clustering import (
    iterative_subcluster,
    rename_subcluster_labels,
    spatial_split_clusters,
    plot_spatial_split_diagnostics,
)
from scutils.tools.differential_expression import (
    deseq2,
    format_deseq2_results,
)
from scutils.tools.gene_scoring import (
    compute_aucell_scores,
    compute_hotspot_scores,
    compute_ulm_scores,
)
from scutils.tools.functional import get_enriched_terms

__all__ = [
    # clustering
    "iterative_subcluster",
    "rename_subcluster_labels",
    "spatial_split_clusters",
    "plot_spatial_split_diagnostics",
    # differential expression
    "deseq2",
    "format_deseq2_results",
    # gene scoring
    "compute_aucell_scores",
    "compute_hotspot_scores",
    "compute_ulm_scores",
    # functional / pathway enrichment
    "get_enriched_terms",
]
