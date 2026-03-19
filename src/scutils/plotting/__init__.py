"""Plotting utilities for single-cell data visualisation."""

from scutils.plotting.barplots import cell_count_barplot
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
    heatmap_feature_aggregated_three_categories,
)
from scutils.plotting.embeddings import (
    embedding_category_multiplot,
    embedding_gene_expression_multiplot,
)
from scutils.plotting.volcano_plot import volcano_plot
from scutils.plotting.density_plotting import (
    plot_embedding_categories,
    plot_density_outlines,
    plot_area_of_interest_density_outlines,
    aoi_density_outlines_multiplot,
)
from scutils.plotting.sankey import sankey_plot
from scutils.plotting.upset import upset_plot
from scutils.plotting._utils import show_plotly
from scutils.plotting.functional import create_pathway_dotplot, load_pathway_data

__all__ = [
    # barplots
    "cell_count_barplot",
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
    "heatmap_feature_aggregated_three_categories",
    # embeddings
    "embedding_category_multiplot",
    "embedding_gene_expression_multiplot",
    # volcano
    "volcano_plot",
    # density
    "plot_embedding_categories",
    "plot_density_outlines",
    "plot_area_of_interest_density_outlines",
    "aoi_density_outlines_multiplot",
    # sankey
    "sankey_plot",
    # upset
    "upset_plot",
    # display helpers
    "show_plotly",
    # functional / pathway enrichment
    "create_pathway_dotplot",
    "load_pathway_data",
]
