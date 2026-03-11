"""Plotting utilities for single-cell data visualisation."""

from scutils.plotting.boxplots import (
    plot_feature_boxplot,
    plot_feature_boxplot_multiplot,
    plot_feature_boxplot_aggregated,
    plot_feature_boxplot_aggregated_multiplot,
)
from scutils.plotting.dotplots import (
    dotplot_expression_two_categories,
    dotplot_expression_two_categories_multiplot,
)
from scutils.plotting.heatmaps import (
    heatmap_expression_two_categories,
    heatmap_expression_two_categories_multiplot,
)
from scutils.plotting.embeddings import (
    embedding_category_multiplot,
    embedding_gene_expression_multiplot,
)
from scutils.plotting.volcano_plot import volcano_plot
from scutils.plotting.density_plotting import (
    plot_embedding_categories,
    plot_density_embedding,
    plot_density_embedding_multiplot,
    plot_density_embedding_comparison,
)

__all__ = [
    # boxplots
    "plot_feature_boxplot",
    "plot_feature_boxplot_multiplot",
    "plot_feature_boxplot_aggregated",
    "plot_feature_boxplot_aggregated_multiplot",
    # dotplots
    "dotplot_expression_two_categories",
    "dotplot_expression_two_categories_multiplot",
    # heatmaps
    "heatmap_expression_two_categories",
    "heatmap_expression_two_categories_multiplot",
    # embeddings
    "embedding_category_multiplot",
    "embedding_gene_expression_multiplot",
    # volcano
    "volcano_plot",
    # density
    "plot_embedding_categories",
    "plot_density_embedding",
    "plot_density_embedding_multiplot",
    "plot_density_embedding_comparison",
]
