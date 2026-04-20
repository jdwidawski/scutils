"""Analytical tools for single-cell data."""

from scutils.tools.clustering import (
    iterative_subcluster,
    rename_subcluster_labels,
    spatial_split_clusters,
    plot_spatial_split_diagnostics,
    spatial_split_reassign_subclusters,
    plot_spatial_split_reassign_subclusters_diagnostics,
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
from scutils.tools.disease_subclusters import (
    detect_disease_enriched_subclusters,
    plot_disease_enriched_subclusters,
)
from scutils.tools.functional import get_enriched_terms
from scutils.tools.score_hubs import (
    find_high_score_subclusters,
    find_score_hubs,
    plot_subcluster_score_diagnostics,
    plot_score_hub_diagnostics,
)

__all__ = [
    # clustering
    "iterative_subcluster",
    "rename_subcluster_labels",
    "spatial_split_clusters",
    "plot_spatial_split_diagnostics",
    "spatial_split_reassign_subclusters",
    "plot_spatial_split_reassign_subclusters_diagnostics",
    # differential expression
    "deseq2",
    "format_deseq2_results",
    # gene scoring
    "compute_aucell_scores",
    "compute_hotspot_scores",
    "compute_ulm_scores",
    # functional / pathway enrichment
    "get_enriched_terms",
    # disease-enriched subclusters
    "detect_disease_enriched_subclusters",
    "plot_disease_enriched_subclusters",
    # score hub detection
    "find_high_score_subclusters",
    "find_score_hubs",
    "plot_subcluster_score_diagnostics",
    "plot_score_hub_diagnostics",
]
