"""Plotting utilities for functional / pathway enrichment results."""

from __future__ import annotations

import logging
from textwrap import wrap
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------




def create_pathway_dotplot(
    data: pd.DataFrame,
    source_colors: Dict[str, str],
    figsize: Tuple[int, int] = (12, 8),
    max_pathways: Optional[int] = 20,
    min_dot_size: float = 20,
    max_dot_size: float = 200,
    variable_size: bool = True,
    title: str = "Functional Pathway Analysis",
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """Create a dot-plot visualising functional pathway enrichment results.

    Pathways are shown on the y-axis (most significant at the top), and
    :math:`-\\log_{10}(p)` on the x-axis.  Dot colour encodes the source
    database and, optionally, dot size encodes significance.

    Args:
        data: DataFrame with at minimum the columns ``source``, ``name``, and
            ``p_value``.  Typically the output of
            :func:`scutils.tl.get_enriched_terms`.
        source_colors: Mapping from source database names to hex colour codes
            (e.g. ``{"GO:BP": "#1f77b4", "REAC": "#ff7f0e"}``).  Sources
            missing from the mapping receive default fallback colours.
        figsize: ``(width, height)`` of the figure in inches.  Defaults to
            ``(12, 8)``.
        max_pathways: Maximum number of pathways to display; the
            *max_pathways* most significant terms are kept.  ``None`` shows
            all terms.  Defaults to ``20``.
        min_dot_size: Smallest marker area when ``variable_size=True``.
            Defaults to ``20``.
        max_dot_size: Largest marker area when ``variable_size=True``.
            Defaults to ``200``.
        variable_size: When ``True``, marker size scales with significance
            (smaller p-value → larger dot).  When ``False``, all dots have
            the same size.  Defaults to ``True``.
        title: Plot title.  Defaults to ``"Functional Pathway Analysis"``.
        save_path: Path to save the figure.  When ``None``, no file is
            written.  Defaults to ``None``.
        dpi: Resolution (dots per inch) for the saved figure.  Defaults to
            ``300``.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the dot-plot.

    Raises:
        TypeError: If *data* is not a :class:`pandas.DataFrame` or
            *source_colors* is not a :class:`dict`.
        ValueError: If required columns are missing or *data* is empty.

    Example:
        >>> import scutils
        >>> enrich_df = scutils.tl.get_enriched_terms(gene_list)
        >>> colors = {"GO:BP": "#1f77b4", "REAC": "#ff7f0e", "KEGG": "#2ca02c"}
        >>> fig = create_pathway_dotplot(enrich_df, colors, max_pathways=15)
        >>> fig.savefig("pathways.png", dpi=150)
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
    
    required_columns = ['source', 'name', 'p_value']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if data.empty:
        raise ValueError("Input DataFrame cannot be empty")
    
    if not isinstance(source_colors, dict):
        raise TypeError("source_colors must be a dictionary")
    
    if not isinstance(variable_size, bool):
        raise TypeError("variable_size must be a boolean")
    
    # Prepare data
    plot_data = data.copy()
    
    # Sort by p-value (most significant first)
    plot_data = plot_data.sort_values('p_value', ascending=True)
    
    # Limit number of pathways if specified
    if max_pathways is not None and len(plot_data) > max_pathways:
        plot_data = plot_data.head(max_pathways)
    
    # Calculate -log10(p_value) for x-axis
    plot_data = plot_data.copy()
    plot_data['-log10_pvalue'] = -np.log10(plot_data['p_value'])
    
    # Check for sources not in color mapping
    unique_sources = plot_data['source'].unique()
    missing_colors = [source for source in unique_sources if source not in source_colors]
    if missing_colors:
        logging.warning("No colour specified for sources: %s", missing_colors)
        _fallback = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        for i, source in enumerate(missing_colors):
            source_colors[source] = _fallback[i % len(_fallback)]
    
    # Create figure with space for legends on the right
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate dot sizes based on variable_size parameter
    if variable_size:
        min_p = plot_data['p_value'].min()
        max_p = plot_data['p_value'].max()
        
        if max_p == min_p:
            # All p-values are the same
            dot_sizes = np.full(len(plot_data), (min_dot_size + max_dot_size) / 2)
        else:
            # Scale dot sizes inversely proportional to p-value (smaller p = larger size)
            # Use log scale for better visualization
            log_min_p = np.log10(min_p)
            log_max_p = np.log10(max_p)
            log_p_values = np.log10(plot_data['p_value'])
            
            # Invert the scaling so smaller p-values get larger sizes
            size_range = max_dot_size - min_dot_size
            normalized_sizes = (log_max_p - log_p_values) / (log_max_p - log_min_p)
            dot_sizes = min_dot_size + normalized_sizes * size_range
    else:
        # Use uniform dot size (average of min and max)
        uniform_size = (min_dot_size + max_dot_size) / 2
        dot_sizes = np.full(len(plot_data), uniform_size)
    
    # Create scatter plot for each source
    y_positions = range(len(plot_data))
    
    # Plot all points at once, grouped by source for proper coloring
    for source in unique_sources:
        source_mask = plot_data['source'] == source
        source_data = plot_data[source_mask]
        source_y_positions = [i for i, mask in enumerate(source_mask) if mask]
        source_dot_sizes = dot_sizes[source_mask]
        
        ax.scatter(
            source_data['-log10_pvalue'], 
            source_y_positions,
            s=source_dot_sizes,
            c=source_colors[source],
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=source
        )
    
    # Customize plot
    ax.set_xlabel('-log10(p-value)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Functional Pathway', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis labels to pathway names
    ax.set_yticks(y_positions)
    pathway_labels = plot_data['name'].tolist()
    
    max_label_length = 50
    truncated_labels = [
        "\n".join(wrap(label, max_label_length)) if len(label) > max_label_length else label
        for label in pathway_labels
    ]
    
    ax.set_yticklabels(truncated_labels, fontsize=12)
    
    # Invert y-axis so most significant pathways are at the top
    ax.invert_yaxis()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout to make room for legends on the right
    plt.tight_layout()
    
    # Create source color legend with all provided colors (not just visible ones)
    source_legend_handles = _create_source_legend_handles(source_colors)
    source_legend = ax.legend(
        handles=source_legend_handles,
        title='Source',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True
    )
    source_legend.get_title().set_fontweight('bold')
    
    # Create size legend for p-values only if variable_size is True
    if variable_size:
        min_p = plot_data['p_value'].min()
        max_p = plot_data['p_value'].max()
        size_legend_handles = _create_size_legend_handles(
            min_p, max_p, min_dot_size, max_dot_size
        )
        
        if size_legend_handles:
            size_legend = ax.legend(
                handles=size_legend_handles,
                title='Significance\n(p-value)',
                bbox_to_anchor=(1.02, 0.5),
                loc='upper left',
                frameon=True,
                fancybox=True,
                shadow=True
            )
            size_legend.get_title().set_fontweight('bold')
            
            # Add the size legend to the plot while preserving the source legend
            ax.add_artist(source_legend)
    
    # Adjust the figure to accommodate legends with appropriate space
    legend_space = 0.65 if variable_size else 0.75
    plt.subplots_adjust(right=legend_space)
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


def _create_source_legend_handles(source_colors: Dict[str, str]) -> list:
    """Build legend handles for the pathway-source colour legend.

    Args:
        source_colors: Mapping from source database name to hex colour code.

    Returns:
        List of :class:`matplotlib.lines.Line2D` handles for use in a legend.
    """
    legend_handles = []
    uniform_size = 8  # Fixed size for all legend dots
    
    for source, color in source_colors.items():
        handle = plt.Line2D(
            [0], [0], 
            marker='o', 
            color='w', 
            markerfacecolor=color,
            markeredgecolor='black',
            markeredgewidth=0.5,
            markersize=uniform_size,
            alpha=0.7,
            label=source,
            linestyle='None'
        )
        legend_handles.append(handle)
    
    return legend_handles


def _create_size_legend_handles(
    min_p: float,
    max_p: float,
    min_dot_size: float,
    max_dot_size: float,
) -> list:
    """Build legend handles representing the p-value → dot-size mapping.

    Args:
        min_p: Smallest p-value present in the plotted data.
        max_p: Largest p-value present in the plotted data.
        min_dot_size: Smallest scatter marker area used in the plot.
        max_dot_size: Largest scatter marker area used in the plot.

    Returns:
        List of :class:`matplotlib.lines.Line2D` handles (three representative
        sizes), or an empty list when all p-values are identical or an error
        occurs.
    """
    try:
        # Create size legend with representative sizes that reflect actual min/max
        if max_p > min_p:
            # Create three representative sizes: min, middle, max
            legend_sizes = [max_dot_size, (min_dot_size + max_dot_size) / 2, min_dot_size]
            legend_p_values = [min_p, np.sqrt(min_p * max_p), max_p]
            
            legend_handles = []
            
            for size, p_val in zip(legend_sizes, legend_p_values):
                # Format p-value as p<1e-X format
                exponent = int(np.floor(np.log10(p_val)))
                label = f'p<1e{exponent:+d}'
                # Create scatter handle for legend with actual size reflecting plot sizes
                handle = plt.Line2D(
                    [0], [0],
                    marker='o', 
                    color='w', 
                    markerfacecolor='gray',
                    markeredgecolor='black',
                    markeredgewidth=0.5,
                    markersize=np.sqrt(size),  # Convert area to radius for markersize
                    alpha=0.7,
                    label=label,
                    linestyle='None'
                )
                legend_handles.append(handle)
            
            return legend_handles
        
        return []
        
    except Exception as e:
        logging.warning(f"Could not create size legend handles: {e}")
        return []


def load_pathway_data(file_path: str) -> pd.DataFrame:
    """Load pathway enrichment results from a CSV file.

    Validates that the required columns are present and that the ``p_value``
    column contains numeric values in the range ``(0, 1]``.

    Args:
        file_path: Path to the CSV file containing pathway analysis results.

    Returns:
        DataFrame with at least the columns ``source``, ``name``, and
        ``p_value``.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If required columns are missing or the file cannot be
            parsed.

    Example:
        >>> data = load_pathway_data("enrichment_results.csv")
        >>> data.head()
    """
    try:
        data = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['source', 'name', 'p_value']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # Convert p_value to numeric if it's not already
        data['p_value'] = pd.to_numeric(data['p_value'], errors='coerce')
        
        # Remove rows with invalid p-values
        invalid_pvalues = data['p_value'].isna() | (data['p_value'] <= 0) | (data['p_value'] > 1)
        if invalid_pvalues.any():
            logging.warning("Removing %d rows with invalid p-values.", invalid_pvalues.sum())
            data = data[~invalid_pvalues]

        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Pathway data file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading pathway data: {str(e)}")

