
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from typing import Optional, Union, Dict, List, Tuple
from scipy.ndimage import gaussian_filter

def plot_embedding_categories(
    adata: AnnData,
    category_dict: Dict[str, List[str]],
    basis: str = "umap",
    palette: Optional[Union[str, Dict[str, Union[str, List[str], Dict[str, str]]]]] = None,
    figsize: Tuple[float, float] = (8, 6),
    size: float = 10.0,
    title: Optional[str] = None,
    legend_loc: Optional[str] = 'right margin',
    show_others: bool = True,
    others_color: str = 'lightgray',
    others_alpha: float = 0.3,
    alpha: float = 0.8
) -> Figure:
    """
    Plot embedding with cells colored by categories from multiple columns.
    
    This function visualizes cells in a 2D embedding space, coloring them according to
    their category assignments from one or more columns in adata.obs.
    
    Args:
        adata: AnnData object containing embedding coordinates.
        category_dict: Dictionary where keys are column names in adata.obs (e.g., 'leiden', 'cell_type')
                      and values are lists of category values to plot from that column.
                      Example: {'leiden': ['0', '1', '2'], 'cell_type': ['T cell', 'B cell']}
        basis: Key in adata.obsm for the embedding to plot (default: 'umap').
        palette: Color palette for categories. Pass ``None`` to use
            ``adata.uns['{category}_colors']`` when available, a Matplotlib
            colormap name string to apply one map to all categories, or a
            ``dict`` mapping column names to individual colour specifications
            (colormap string, list of colours, or value-to-colour dict).
        figsize: Figure size as (width, height) tuple (default: (8, 6)).
        size: Size of scatter plot points (default: 10.0).
        title: Plot title. If None, uses 'Embedding Categories'.
        legend_loc: Legend location. Options are:
                   - None or 'right margin': Place legend outside plot area on the right
                   - 'on data': Place legend inside plot area at best location
        show_others: Whether to show cells not in specified categories (default: True).
        others_color: Color for cells not in specified categories (default: 'lightgray').
        others_alpha: Alpha transparency for cells not in specified categories (default: 0.3).
        alpha: Alpha transparency for cells in specified categories (default: 0.8).
        
    Returns:
        matplotlib.figure.Figure: The generated figure object.
        
    Raises:
        KeyError: If category or basis not found.
        ValueError: If invalid input parameters.
        
    Examples:
        >>> import scanpy as sc
        >>> from density_outline_plot import plot_embedding_categories
        >>> 
        >>> # Load data
        >>> adata = sc.read_h5ad('data.h5ad')
        >>> 
        >>> # Plot single category
        >>> fig = plot_embedding_categories(
        ...     adata,
        ...     category_dict={'leiden': ['0', '1', '2']},
        ...     basis='umap'
        ... )
        >>> 
        >>> # Plot multiple categories from different columns
        >>> fig = plot_embedding_categories(
        ...     adata,
        ...     category_dict={
        ...         'leiden': ['0', '1', '2'],
        ...         'cell_type': ['T cell', 'B cell']
        ...     },
        ...     palette={
        ...         'leiden': 'tab10',
        ...         'cell_type': {'T cell': 'red', 'B cell': 'blue'}
        ...     }
        ... )
        >>> 
        >>> # Plot without showing other cells
        >>> fig = plot_embedding_categories(
        ...     adata,
        ...     category_dict={'leiden': ['0', '1', '2']},
        ...     show_others=False
        ... )
    """
    # Input validation
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData object")
    
    if not isinstance(category_dict, dict):
        raise TypeError("category_dict must be a dictionary")
    
    if not category_dict:
        raise ValueError("category_dict cannot be empty")
    
    # Validate all categories exist
    for category, categories in category_dict.items():
        if category not in adata.obs.columns:
            raise KeyError(f"Category '{category}' not found in adata.obs")
    
    embedding_key = f"X_{basis}"
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding '{embedding_key}' not found in adata.obsm")
    
    # Validate legend_loc
    valid_legend_locs = [None, 'right margin', 'on data']
    if legend_loc not in valid_legend_locs:
        raise ValueError(f"legend_loc must be one of {valid_legend_locs}, got '{legend_loc}'")
    
    # Ensure all categories are categorical
    for category in category_dict.keys():
        if not pd.api.types.is_categorical_dtype(adata.obs[category]):
            adata.obs[category] = pd.Categorical(adata.obs[category])
    
    # Build color mapping for all categories
    color_mapping = {}
    
    for category, categories in category_dict.items():
        n_categories = len(categories)
        
        # Handle palette for this category
        category_palette = None
        
        if palette is None:
            # Try to get colors from adata.uns
            colors_key = f"{category}_colors"
            if colors_key in adata.uns:
                all_cats = adata.obs[category].cat.categories.tolist()
                all_colors = adata.uns[colors_key]
                palette_colors = [all_colors[all_cats.index(cat)] for cat in categories if cat in all_cats]
            else:
                # Set default colors
                sc.pl._utils._set_default_colors_for_categorical_obs(adata, category)
                all_cats = adata.obs[category].cat.categories.tolist()
                all_colors = adata.uns[colors_key]
                palette_colors = [all_colors[all_cats.index(cat)] for cat in categories if cat in all_cats]
        elif isinstance(palette, str):
            # Single colormap for all
            cmap = plt.cm.get_cmap(palette)
            palette_colors = [cmap(i / n_categories) for i in range(n_categories)]
        elif isinstance(palette, dict):
            category_palette = palette.get(category)
            
            if category_palette is None:
                # Use default colors for this category
                colors_key = f"{category}_colors"
                if colors_key in adata.uns:
                    all_cats = adata.obs[category].cat.categories.tolist()
                    all_colors = adata.uns[colors_key]
                    palette_colors = [all_colors[all_cats.index(cat)] for cat in categories if cat in all_cats]
                else:
                    sc.pl._utils._set_default_colors_for_categorical_obs(adata, category)
                    all_cats = adata.obs[category].cat.categories.tolist()
                    all_colors = adata.uns[colors_key]
                    palette_colors = [all_colors[all_cats.index(cat)] for cat in categories if cat in all_cats]
            elif isinstance(category_palette, str):
                # Colormap for this category
                cmap = plt.cm.get_cmap(category_palette)
                palette_colors = [cmap(i / n_categories) for i in range(n_categories)]
            elif isinstance(category_palette, list):
                if len(category_palette) != n_categories:
                    raise ValueError(
                        f"Palette list length ({len(category_palette)}) must match number of categories "
                        f"({n_categories}) for '{category}'"
                    )
                palette_colors = category_palette
            elif isinstance(category_palette, dict):
                palette_colors = [category_palette.get(str(cat), '#000000') for cat in categories]
            else:
                raise TypeError(f"Invalid palette type for category '{category}'")
        else:
            raise TypeError("palette must be None, str, or dict")
        
        # Store colors for this category
        for i, cat in enumerate(categories):
            color_mapping[(category, cat)] = palette_colors[i]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get embedding
    embedding = adata.obsm[embedding_key]
    
    # Create a mask for all cells that belong to any specified category
    all_specified_mask = np.zeros(adata.n_obs, dtype=bool)
    for category, categories in category_dict.items():
        for cat in categories:
            cat_mask = adata.obs[category] == cat
            all_specified_mask |= cat_mask
    
    # Plot cells not in specified categories first (background)
    if show_others:
        other_mask = ~all_specified_mask
        if other_mask.any():
            ax.scatter(
                embedding[other_mask, 0],
                embedding[other_mask, 1],
                c=others_color,
                s=size,
                alpha=others_alpha,
                rasterized=True,
                label='Others'
            )
    
    # Track legend elements
    legend_elements = []
    
    # Plot each category
    for category, categories in category_dict.items():
        for cat in categories:
            # Get mask for this category
            cat_mask = adata.obs[category] == cat
            
            if not cat_mask.any():
                continue
            
            # Get color for this category
            color = color_mapping[(category, cat)]
            
            # Plot cells in this category
            ax.scatter(
                embedding[cat_mask, 0],
                embedding[cat_mask, 1],
                c=color,
                s=size,
                alpha=alpha,
                rasterized=True,
                label=f"{category}: {cat}" if len(category_dict) > 1 else str(cat)
            )
            
            # Add to legend
            label = f"{category}: {cat}" if len(category_dict) > 1 else str(cat)
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=color, markersize=8, label=label)
            )
    
    # Set labels and title
    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")
    
    if title is None:
        ax.set_title('Embedding Categories')
    else:
        ax.set_title(title)
    
    # Remove grid
    ax.grid(False)
    
    # Add legend
    if legend_elements:
        if legend_loc == 'right margin' or legend_loc is None:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=True
            )
        elif legend_loc == 'on data':
            ax.legend(
                handles=legend_elements,
                loc='best',
                frameon=True
            )
    
    plt.tight_layout()
    
    return fig

"""
Density-based Cluster Outline Visualization Module

This module provides functionality for visualizing high-density regions of cell clusters
in single-cell RNA-seq embeddings using contour outlines.
"""


def plot_density_outlines(
    adata: AnnData,
    category_dict: Dict[str, List[str]],
    density_colname: str = "umap_density",
    density_cutoff: float = 1.0,
    basis: str = "umap",
    palette: Optional[Union[str, Dict[str, Union[str, List[str], Dict[str, str]]]]] = None,
    linewidth: float = 2.0,
    figsize: Tuple[float, float] = (8, 6),
    size: float = 1.0,
    title: Optional[str] = None,
    legend_loc: Optional[str] = None,
    show_labels: bool = True,
    contour_squeeze: float = 0.1
) -> Figure:
    """
    Plot embedding with density-based outlines for multiple categories.
    
    This function visualizes high-density regions of different categories from multiple
    columns in a 2D embedding space by drawing contour outlines around areas where cells
    of each category are densely packed. The density values should be pre-computed using
    scanpy.tl.embedding_density.
    
    Args:
        adata: AnnData object containing pre-computed embedding and density values.
        category_dict: Dictionary where keys are column names in adata.obs (e.g., 'leiden', 'cell_type')
                      and values are lists of category values to plot from that column.
                      Example: {'leiden': ['0', '1', '2'], 'cell_type': ['T cell', 'B cell']}
        density_colname: Column name in adata.obs containing density values. The full column name should be
                        formatted as '{density_colname}_{category}' (e.g., 'umap_density_leiden').
        density_cutoff: Threshold for defining high-density regions. Only cells with density
                       above this value will be included in the outline.
        basis: Key in adata.obsm for the embedding to plot (default: 'umap').
        palette: Color palette for category outlines. Pass ``None`` to use
            ``adata.uns['{category}_colors']`` when available, a Matplotlib
            colormap name string to apply one map to all categories, or a
            ``dict`` mapping column names to individual colour specifications
            (colormap string, list of colours, or value-to-colour dict).
        linewidth: Width of the outline contours (default: 2.0).
        figsize: Figure size as (width, height) tuple (default: (8, 6)).
        size: Size of scatter plot points (default: 1.0).
        title: Plot title. If None, uses 'Density Outlines'.
        legend_loc: Legend location. Options are:
                   - None or 'right margin': Place legend outside plot area on the right
                   - 'on data': Place legend inside plot area at best location
        show_labels: Whether to show category labels on the contours (default: True).
        contour_squeeze: how far should the contour go from the cells (extra padding)
        
    Returns:
        matplotlib.figure.Figure: The generated figure object.
        
    Raises:
        KeyError: If category, basis, or density column not found.
        ValueError: If invalid input parameters.
        
    Examples:
        >>> import scanpy as sc
        >>> from density_outline_plot import plot_density_outlines
        >>> 
        >>> # Load data and compute embeddings
        >>> adata = sc.read_h5ad('data.h5ad')
        >>> sc.pp.neighbors(adata)
        >>> sc.tl.umap(adata)
        >>> sc.tl.leiden(adata)
        >>> 
        >>> # Compute embedding density for multiple groupings
        >>> sc.tl.embedding_density(adata, basis='umap', groupby='leiden')
        >>> sc.tl.embedding_density(adata, basis='umap', groupby='cell_type')
        >>> 
        >>> # Plot outlines from multiple columns
        >>> fig = plot_density_outlines(
        ...     adata,
        ...     category_dict={
        ...         'leiden': ['0', '1', '2'],
        ...         'cell_type': ['T cell', 'B cell']
        ...     },
        ...     density_colname='umap_density',
        ...     density_cutoff=1.5,
        ...     basis='umap'
        ... )
        >>> 
        >>> # Plot with custom palettes per column
        >>> fig = plot_density_outlines(
        ...     adata,
        ...     category_dict={
        ...         'leiden': ['0', '1', '2'],
        ...         'cell_type': ['T cell', 'B cell']
        ...     },
        ...     palette={
        ...         'leiden': 'tab10',
        ...         'cell_type': {'T cell': 'red', 'B cell': 'blue'}
        ...     },
        ...     density_cutoff=1.5
        ... )
    """
    # Input validation
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData object")
    
    if not isinstance(category_dict, dict):
        raise TypeError("category_dict must be a dictionary")
    
    if not category_dict:
        raise ValueError("category_dict cannot be empty")
    
    # Validate all categories exist
    for category, categories in category_dict.items():
        if category not in adata.obs.columns:
            raise KeyError(f"Category '{category}' not found in adata.obs")
        
        # Check density column exists
        density_col = f"{density_colname}_{category}"
        if density_col not in adata.obs.columns:
            raise KeyError(
                f"Density column '{density_col}' not found in adata.obs. "
                f"Run sc.tl.embedding_density(adata, basis='{basis}', groupby='{category}') first."
            )
    
    embedding_key = f"X_{basis}"
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding '{embedding_key}' not found in adata.obsm")
    
    # Validate legend_loc
    valid_legend_locs = [None, 'right margin', 'on data']
    if legend_loc not in valid_legend_locs:
        raise ValueError(f"legend_loc must be one of {valid_legend_locs}, got '{legend_loc}'")
    
    # Ensure all categories are categorical
    for category in category_dict.keys():
        if not pd.api.types.is_categorical_dtype(adata.obs[category]):
            adata.obs[category] = pd.Categorical(adata.obs[category])
    
    # Build color mapping for all categories
    color_mapping = {}
    
    for category, categories in category_dict.items():
        n_categories = len(categories)
        
        # Handle palette for this category
        category_palette = None
        
        if palette is None:
            # Try to get colors from adata.uns
            colors_key = f"{category}_colors"
            if colors_key in adata.uns:
                all_cats = adata.obs[category].cat.categories.tolist()
                all_colors = adata.uns[colors_key]
                palette_colors = [all_colors[all_cats.index(cat)] for cat in categories if cat in all_cats]
            else:
                # Set default colors
                sc.pl._utils._set_default_colors_for_categorical_obs(adata, category)
                all_cats = adata.obs[category].cat.categories.tolist()
                all_colors = adata.uns[colors_key]
                palette_colors = [all_colors[all_cats.index(cat)] for cat in categories if cat in all_cats]
        elif isinstance(palette, str):
            # Single colormap for all
            cmap = plt.cm.get_cmap(palette)
            palette_colors = [cmap(i / n_categories) for i in range(n_categories)]
        elif isinstance(palette, dict):
            category_palette = palette.get(category)
            
            if category_palette is None:
                # Use default colors for this category
                colors_key = f"{category}_colors"
                if colors_key in adata.uns:
                    all_cats = adata.obs[category].cat.categories.tolist()
                    all_colors = adata.uns[colors_key]
                    palette_colors = [all_colors[all_cats.index(cat)] for cat in categories if cat in all_cats]
                else:
                    sc.pl._utils._set_default_colors_for_categorical_obs(adata, category)
                    all_cats = adata.obs[category].cat.categories.tolist()
                    all_colors = adata.uns[colors_key]
                    palette_colors = [all_colors[all_cats.index(cat)] for cat in categories if cat in all_cats]
            elif isinstance(category_palette, str):
                # Colormap for this category
                cmap = plt.cm.get_cmap(category_palette)
                palette_colors = [cmap(i / n_categories) for i in range(n_categories)]
            elif isinstance(category_palette, list):
                if len(category_palette) != n_categories:
                    raise ValueError(
                        f"Palette list length ({len(category_palette)}) must match number of categories "
                        f"({n_categories}) for '{category}'"
                    )
                palette_colors = category_palette
            elif isinstance(category_palette, dict):
                palette_colors = [category_palette.get(str(cat), '#000000') for cat in categories]
            else:
                raise TypeError(f"Invalid palette type for category '{category}'")
        else:
            raise TypeError("palette must be None, str, or dict")
        
        # Store colors for this category
        for i, cat in enumerate(categories):
            color_mapping[(category, cat)] = palette_colors[i]
    
    # Create base embedding plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot embedding points in light gray
    embedding = adata.obsm[embedding_key]
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c='lightgray',
        s=size,
        alpha=0.3,
        rasterized=True
    )
    
    # Get embedding bounds for grid
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    
    # Add padding
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Create grid
    grid_size = 100
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Track legend elements
    legend_elements = []
    
    # Plot density outlines for each category from each column
    for category, categories in category_dict.items():
        density_col = f"{density_colname}_{category}"
        
        for cat in categories:
            # Subset data for this category
            cat_mask = adata.obs[category] == cat
            
            # Get density values for cells in this category
            cat_densities = adata.obs.loc[cat_mask, density_col]
            
            # Get cells with density above cutoff
            high_density_mask = cat_densities > density_cutoff
            
            if not high_density_mask.any():
                continue
            
            # Get coordinates of high-density cells
            cat_indices = np.where(cat_mask)[0]
            high_density_indices = cat_indices[high_density_mask.values]
            high_density_coords = embedding[high_density_indices]
            
            # Get color for this category
            color = color_mapping[(category, cat)]
            
            # Create contour plot
            try:
                # Create density map on grid
                density_grid = np.zeros((grid_size, grid_size))
                
                for x_coord, y_coord in high_density_coords:
                    x_idx = int((x_coord - x_min) / (x_max - x_min) * (grid_size - 1))
                    y_idx = int((y_coord - y_min) / (y_max - y_min) * (grid_size - 1))
                    
                    if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                        density_grid[y_idx, x_idx] += 1
                
                # Smooth the density grid
                density_grid = gaussian_filter(density_grid, sigma=2)
                
                if density_grid.max() > 0:
                    # Plot contour outline
                    contour_levels = [density_grid.max() * contour_squeeze]
                    contours = ax.contour(
                        Xi, Yi, density_grid,
                        levels=contour_levels,
                        colors=[color],
                        linewidths=linewidth,
                        alpha=0.8
                    )
                    
                    # Add to legend if contour was created
                    if len(contours.allsegs) > 0 and len(contours.allsegs[0]) > 0:
                        label = f"{category.split('_')[-1]}: {cat}" # if len(category_dict) > 1 else str(cat)
                        legend_elements.append(
                            Line2D([0], [0], color=color, linewidth=linewidth, label=label)
                        )
                        
                        # Add label on plot
                        if show_labels:
                            paths = contours.allsegs[0]
                            if paths:
                                largest_path = max(paths, key=lambda p: len(p))
                                if len(largest_path) > 0:
                                    centroid = np.array(largest_path).mean(axis=0)
                                    ax.text(
                                        centroid[0], centroid[1], str(cat),
                                        fontsize=10,
                                        fontweight='bold',
                                        color=color,
                                        ha='center',
                                        va='center',
                                        bbox=dict(
                                            boxstyle='round,pad=0.3',
                                            facecolor='white',
                                            alpha=0.7,
                                            edgecolor=color
                                        )
                                    )
            
            except Exception as e:
                print(f"Warning: Could not create outline for {category}={cat}: {e}")
                continue
    
    # Set labels and title
    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")
    
    if title is None:
        ax.set_title('Density Outlines')
    else:
        ax.set_title(title)
    
    # Remove grid
    ax.grid(False)
    
    # Add legend
    if legend_elements:
        if legend_loc == 'right margin' or (legend_loc is None and show_labels == False):
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=True
            )
    
    plt.tight_layout()
    
    return fig


def plot_area_of_interest_density_outlines(
    adata: AnnData,
    category_dict: Dict[str, List[str]],
    aoi_category: Optional[str] = None,
    aoi_values: Optional[List[str]] = None,
    density_colname: str = "umap_density",
    density_cutoff: Union[float, Dict[str, float]] = 1.0,
    basis: str = "umap",
    palette: Optional[
        Union[str, Dict[str, Union[str, List[str], Dict[str, str]]]]
    ] = None,
    aoi_color_by: Optional[str] = None,
    aoi_palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    show_aoi_outline: bool = False,
    aoi_outline_color: str = "black",
    aoi_outline_linewidth: float = 1.5,
    aoi_outline_linestyle: str = "--",
    aoi_outline_sigma: float = 3.0,
    aoi_outline_squeeze: float = 0.05,
    aoi_size: float = 1.0,
    aoi_alpha: float = 0.3,
    aoi_default_color: str = "lightgray",
    linewidth: float = 2.0,
    figsize: Tuple[float, float] = (8, 6),
    background_size: float = 1.0,
    background_color: str = "lightgray",
    background_alpha: float = 0.3,
    background_color_by: Optional[str] = None,
    background_palette: Optional[
        Union[str, List[str], Dict[str, str]]
    ] = None,
    show_background_labels: bool = False,
    background_label_fontsize: float = 9.0,
    exclude_aoi_from_background: bool = False,
    show_background_legend: bool = False,
    show_density_legend: bool = True,
    legend_remove_category_name: bool = False,
    annotate_aoi: bool = False,
    aoi_annotation_fontsize: float = 10.0,
    aoi_groups: Optional[Dict[str, Dict[str, List[str]]]] = None,
    aoi_groups_colors: Optional[Union[str, List[str], Dict[str, str]]] = None,
    aoi_groups_linewidth: float = 1.5,
    aoi_groups_linestyle: str = "--",
    aoi_groups_sigma: float = 3.0,
    aoi_groups_squeeze: float = 0.05,
    show_aoi_groups_legend: bool = True,
    title: Optional[str] = None,
    legend_loc: Optional[str] = None,
    show_labels: bool = True,
    contour_squeeze: Union[float, Dict[str, float]] = 0.1,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """
    Plot embedding with an area-of-interest highlight and density outlines.

    Layers (bottom to top):
      1. **Background** – all cells as small gray dots.
      2. **Area of interest (AOI)** – cells matching *aoi_category / aoi_values*
         drawn larger and optionally colored by a metadata column.  An optional
         dashed outline can be drawn around the entire AOI region.
      3. **Density outlines** – contour outlines for the categories specified in
         *category_dict* (same logic as ``plot_density_outlines``).

    Args:
        adata: AnnData object with pre-computed embedding and density values.
        category_dict: Columns → category values for density outlines.
            Example: ``{'leiden': ['0', '1', '2']}``.
        aoi_category: Column in ``adata.obs`` that defines the area of interest.
            If ``None``, the AOI layer is skipped entirely and only the
            background and density-outline layers are drawn.
        aoi_values: Values in *aoi_category* that belong to the AOI.
            Ignored when *aoi_category* is ``None``.
        density_colname: Base name for the density column in ``adata.obs``.
            Full column: ``{density_colname}_{category}``.
        density_cutoff: Density threshold for the outline contours. Either a
            single ``float`` applied to every category value, or a
            ``dict`` mapping category value strings to per-value thresholds
            (values absent from the dict fall back to ``1.0``).
        basis: Embedding key in ``adata.obsm`` (default: ``'umap'``).
        palette: Color palette for the density-outline categories (same
            semantics as in ``plot_density_outlines``).
        aoi_color_by: Optional column in ``adata.obs`` used to color the
            AOI cells.  If *None* all AOI cells use *aoi_default_color*.
        aoi_palette: Colors for the *aoi_color_by* column: a Matplotlib
            colormap name string, a list of colours (one per unique value),
            or a value-to-colour dict. Pass ``None`` to use ``adata.uns``
            colours or Scanpy defaults.
        show_aoi_outline: Draw a dashed outline around the whole AOI region
            (default ``False``).
        aoi_outline_color: Colour of the AOI outline (default ``'black'``).
        aoi_outline_linewidth: Line width of the AOI outline.
        aoi_outline_linestyle: Line style of the AOI outline (default ``'--'``).
        aoi_outline_sigma: Gaussian sigma for smoothing the AOI outline grid.
        aoi_outline_squeeze: Contour threshold fraction for the AOI outline.
        aoi_size: Dot size for AOI cells (default: 1.0).
        aoi_alpha: Alpha for AOI cells (default: 0.3).
        aoi_default_color: Uniform colour when *aoi_color_by* is None
            (default ``'lightgray'``).
        linewidth: Width of the density-outline contours.
        figsize: Figure size ``(width, height)``.
        background_size: Dot size for background cells.
        background_color: Colour for background cells (used when
            *background_color_by* is ``None``).
        background_alpha: Alpha for background cells.
        background_color_by: Optional column in ``adata.obs`` used to colour
            background cells by category instead of a uniform colour.
        background_palette: Colours for the *background_color_by* column:
            a Matplotlib colormap name string, a list of colours (one per
            unique value), or a value-to-colour dict. Pass ``None`` to use
            ``adata.uns`` colours or Scanpy defaults.
        show_background_labels: Place text labels at the centroid of each
            background category cluster (default ``False``).
        background_label_fontsize: Font size for background category labels
            (default ``9.0``).
        exclude_aoi_from_background: If ``True``, cells that belong to the
            area of interest are **not** drawn in the background layer, so
            only non-AOI cells are shown (default ``False``).
        show_background_legend: Show a second legend for the background
            categories (default ``False``).
        show_density_legend: Show a legend for the density outline categories
            (default ``True``).  The legend entries match the colour and line
            width of the corresponding contours.
        legend_remove_category_name: If ``True``, density-outline legend
            entries show only the category value (e.g. ``'0'``) instead of
            ``'leiden: 0'`` (default ``False``).
        annotate_aoi: Place a text label reading ``"Area of interest"``
            next to the AOI outline contour (default ``False``).  Only
            has an effect when *show_aoi_outline* is ``True``.
        aoi_annotation_fontsize: Font size for the AOI annotation text
            (default ``10.0``).
        aoi_groups: Draw **multiple** AOI outlines at once. Pass a dict
            mapping group labels (legend entries) to AOI definitions of the
            form ``{column: [values]}``, e.g.
            ``{'T cells': {'cell_type': ['CD4 T', 'CD8 T']}}``. When
            provided, *aoi_category* / *aoi_values* are ignored for outline
            drawing. Set to ``None`` to disable (default).
        aoi_groups_colors: Colors for the *aoi_groups* outlines. Pass
            ``None`` for black outlines, a single colour string, a list of
            colours (one per group in key order), or a group-label-to-colour
            dict.
        aoi_groups_linewidth: Line width for the group outlines
            (default ``1.5``).
        aoi_groups_linestyle: Line style for the group outlines
            (default ``'--'``).
        aoi_groups_sigma: Gaussian smoothing sigma for the group
            outline grids (default ``3.0``).
        aoi_groups_squeeze: Contour threshold fraction for the group
            outlines (default ``0.05``).
        show_aoi_groups_legend: Show a legend for the *aoi_groups*
            outlines (default ``True``).
        title: Plot title.  Defaults to ``'Area of Interest – Density Outlines'``.
        legend_loc: ``None`` / ``'right margin'`` / ``'on data'``.
        show_labels: Show category labels on density contours.
        contour_squeeze: Contour threshold fraction for density outlines.
            Either a single ``float`` applied to every category value or a
            ``dict`` mapping category value strings to per-value fractions
            (values absent from the dict fall back to ``0.1``).

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        KeyError: If a required column or embedding is missing.
        ValueError: If input parameters are invalid.
        TypeError: If argument types are wrong.

    Examples:
        >>> import scanpy as sc
        >>> from density_plotting import plot_area_of_interest_density_outlines
        >>>
        >>> adata = sc.read_h5ad('data.h5ad')
        >>> sc.tl.embedding_density(adata, basis='umap', groupby='leiden')
        >>>
        >>> # Highlight a cell-type region and show density outlines
        >>> fig = plot_area_of_interest_density_outlines(
        ...     adata,
        ...     category_dict={'leiden': ['0', '1', '2']},
        ...     aoi_category='cell_type',
        ...     aoi_values=['T cell', 'NK cell'],
        ...     aoi_color_by='cell_type',
        ...     show_aoi_outline=True,
        ... )
        >>>
        >>> # Minimal: just highlight AOI without colouring
        >>> fig = plot_area_of_interest_density_outlines(
        ...     adata,
        ...     category_dict={'leiden': ['3', '4']},
        ...     aoi_category='cell_type',
        ...     aoi_values=['B cell'],
        ... )
        >>>
        >>> # Density outlines only, no AOI
        >>> fig = plot_area_of_interest_density_outlines(
        ...     adata,
        ...     category_dict={'leiden': ['0', '1', '2']},
        ... )
    """
    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData object")
    if not isinstance(category_dict, dict) or not category_dict:
        raise ValueError("category_dict must be a non-empty dictionary")

    embedding_key = f"X_{basis}"
    if embedding_key not in adata.obsm:
        raise KeyError(
            f"Embedding '{embedding_key}' not found in adata.obsm"
        )

    if aoi_category is not None:
        if aoi_category not in adata.obs.columns:
            raise KeyError(
                f"AOI category '{aoi_category}' not found in adata.obs"
            )
        if aoi_values is None:
            raise ValueError(
                "aoi_values must be provided when aoi_category is set"
            )

    for category, cat_values in category_dict.items():
        if category not in adata.obs.columns:
            raise KeyError(
                f"Category '{category}' not found in adata.obs"
            )
        density_col = f"{density_colname}_{category}"
        if density_col not in adata.obs.columns:
            raise KeyError(
                f"Density column '{density_col}' not found. "
                f"Run sc.tl.embedding_density(adata, basis='{basis}', "
                f"groupby='{category}') first."
            )

    if aoi_groups is not None:
        if not isinstance(aoi_groups, dict) or not aoi_groups:
            raise ValueError(
                "aoi_groups must be a non-empty dictionary when provided"
            )
        for grp_label, grp_def in aoi_groups.items():
            if not isinstance(grp_def, dict) or not grp_def:
                raise ValueError(
                    f"aoi_groups['{grp_label}'] must be a non-empty "
                    "dict mapping a column name to a list of values"
                )
            for col, vals in grp_def.items():
                if col not in adata.obs.columns:
                    raise KeyError(
                        f"Column '{col}' (in aoi_groups['{grp_label}']) "
                        "not found in adata.obs"
                    )

    if aoi_color_by is not None and aoi_color_by not in adata.obs.columns:
        raise KeyError(
            f"aoi_color_by column '{aoi_color_by}' not found in adata.obs"
        )

    if (
        background_color_by is not None
        and background_color_by not in adata.obs.columns
    ):
        raise KeyError(
            f"background_color_by column '{background_color_by}' "
            f"not found in adata.obs"
        )

    valid_legend_locs = [None, "right margin", "on data"]
    if legend_loc not in valid_legend_locs:
        raise ValueError(
            f"legend_loc must be one of {valid_legend_locs}, "
            f"got '{legend_loc}'"
        )

    # ------------------------------------------------------------------ #
    # Ensure categoricals
    # ------------------------------------------------------------------ #
    cols_to_categorize = list(category_dict.keys())
    if aoi_category is not None:
        cols_to_categorize.append(aoi_category)
    for col in cols_to_categorize:
        if not pd.api.types.is_categorical_dtype(adata.obs[col]):
            adata.obs[col] = pd.Categorical(adata.obs[col])

    if aoi_color_by is not None:
        if not pd.api.types.is_categorical_dtype(adata.obs[aoi_color_by]):
            adata.obs[aoi_color_by] = pd.Categorical(
                adata.obs[aoi_color_by]
            )

    if background_color_by is not None:
        if not pd.api.types.is_categorical_dtype(
            adata.obs[background_color_by]
        ):
            adata.obs[background_color_by] = pd.Categorical(
                adata.obs[background_color_by]
            )

    # ------------------------------------------------------------------ #
    # Build colour mapping for density outlines (reuse original logic)
    # ------------------------------------------------------------------ #
    color_mapping: Dict[Tuple[str, str], str] = {}

    for category, cat_values in category_dict.items():
        n_cat = len(cat_values)

        if palette is None:
            colors_key = f"{category}_colors"
            if colors_key not in adata.uns:
                sc.pl._utils._set_default_colors_for_categorical_obs(
                    adata, category
                )
            all_cats = adata.obs[category].cat.categories.tolist()
            all_colors = adata.uns[colors_key]
            pal_colors = [
                all_colors[all_cats.index(c)]
                for c in cat_values
                if c in all_cats
            ]
        elif isinstance(palette, str):
            cmap = plt.cm.get_cmap(palette)
            pal_colors = [cmap(i / n_cat) for i in range(n_cat)]
        elif isinstance(palette, dict):
            cat_pal = palette.get(category)
            if cat_pal is None:
                colors_key = f"{category}_colors"
                if colors_key not in adata.uns:
                    sc.pl._utils._set_default_colors_for_categorical_obs(
                        adata, category
                    )
                all_cats = adata.obs[category].cat.categories.tolist()
                all_colors = adata.uns[colors_key]
                pal_colors = [
                    all_colors[all_cats.index(c)]
                    for c in cat_values
                    if c in all_cats
                ]
            elif isinstance(cat_pal, str):
                cmap = plt.cm.get_cmap(cat_pal)
                pal_colors = [cmap(i / n_cat) for i in range(n_cat)]
            elif isinstance(cat_pal, list):
                if len(cat_pal) != n_cat:
                    raise ValueError(
                        f"Palette list length ({len(cat_pal)}) must match "
                        f"number of categories ({n_cat}) for '{category}'"
                    )
                pal_colors = cat_pal
            elif isinstance(cat_pal, dict):
                pal_colors = [
                    cat_pal.get(str(c), "#000000") for c in cat_values
                ]
            else:
                raise TypeError(
                    f"Invalid palette type for category '{category}'"
                )
        else:
            raise TypeError("palette must be None, str, or dict")

        for i, c in enumerate(cat_values):
            color_mapping[(category, c)] = pal_colors[i]

    # ------------------------------------------------------------------ #
    # Build AOI colour mapping
    # ------------------------------------------------------------------ #
    aoi_color_map: Optional[Dict[str, str]] = None

    if aoi_category is not None and aoi_color_by is not None:
        aoi_unique = (
            adata.obs.loc[
                adata.obs[aoi_category].isin(aoi_values), aoi_color_by
            ]
            .dropna()
            .unique()
            .tolist()
        )
        n_aoi = len(aoi_unique)

        if aoi_palette is None:
            colors_key = f"{aoi_color_by}_colors"
            if colors_key not in adata.uns:
                sc.pl._utils._set_default_colors_for_categorical_obs(
                    adata, aoi_color_by
                )
            all_cats = adata.obs[aoi_color_by].cat.categories.tolist()
            all_colors = adata.uns[colors_key]
            aoi_color_map = {
                c: all_colors[all_cats.index(c)]
                for c in aoi_unique
                if c in all_cats
            }
        elif isinstance(aoi_palette, str):
            cmap = plt.cm.get_cmap(aoi_palette)
            aoi_color_map = {
                c: cmap(i / max(n_aoi, 1))
                for i, c in enumerate(aoi_unique)
            }
        elif isinstance(aoi_palette, list):
            if len(aoi_palette) != n_aoi:
                raise ValueError(
                    f"aoi_palette list length ({len(aoi_palette)}) must "
                    f"match unique AOI colour-by values ({n_aoi})"
                )
            aoi_color_map = {
                c: aoi_palette[i] for i, c in enumerate(aoi_unique)
            }
        elif isinstance(aoi_palette, dict):
            aoi_color_map = {
                c: aoi_palette.get(str(c), aoi_default_color)
                for c in aoi_unique
            }
        else:
            raise TypeError(
                "aoi_palette must be None, str, list, or dict"
            )

    # ------------------------------------------------------------------ #
    # Build background colour mapping
    # ------------------------------------------------------------------ #
    bg_color_map: Optional[Dict[str, str]] = None

    if background_color_by is not None:
        bg_cats = adata.obs[background_color_by].cat.categories.tolist()
        n_bg = len(bg_cats)

        if background_palette is None:
            colors_key = f"{background_color_by}_colors"
            if colors_key not in adata.uns:
                sc.pl._utils._set_default_colors_for_categorical_obs(
                    adata, background_color_by
                )
            all_colors = adata.uns[colors_key]
            bg_color_map = {
                c: all_colors[i] for i, c in enumerate(bg_cats)
            }
        elif isinstance(background_palette, str):
            cmap = plt.cm.get_cmap(background_palette)
            bg_color_map = {
                c: cmap(i / max(n_bg, 1))
                for i, c in enumerate(bg_cats)
            }
        elif isinstance(background_palette, list):
            if len(background_palette) != n_bg:
                raise ValueError(
                    f"background_palette list length "
                    f"({len(background_palette)}) must match unique "
                    f"background categories ({n_bg})"
                )
            bg_color_map = {
                c: background_palette[i]
                for i, c in enumerate(bg_cats)
            }
        elif isinstance(background_palette, dict):
            bg_color_map = {
                c: background_palette.get(str(c), background_color)
                for c in bg_cats
            }
        else:
            raise TypeError(
                "background_palette must be None, str, list, or dict"
            )

    # ------------------------------------------------------------------ #
    # Create figure
    # ------------------------------------------------------------------ #
    _external_ax = ax is not None
    if _external_ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)
    embedding = adata.obsm[embedding_key]
    bg_legend_elements: List[Line2D] = []
    density_legend_elements: List[Line2D] = []

    # ---- Layer 1: background (all cells) ---- #
    # Build a mask selecting which cells to include in the background
    if aoi_category is not None and aoi_values is not None and exclude_aoi_from_background:
        aoi_mask_bg = adata.obs[aoi_category].isin(aoi_values)
        bg_mask = ~aoi_mask_bg.values
    else:
        bg_mask = np.ones(adata.n_obs, dtype=bool)

    if background_color_by is not None and bg_color_map is not None:
        bg_labels = adata.obs[background_color_by]
        for cat_val, color in bg_color_map.items():
            cat_mask = (bg_labels == cat_val).values & bg_mask
            if not cat_mask.any():
                continue
            ax.scatter(
                embedding[cat_mask, 0],
                embedding[cat_mask, 1],
                c=color,
                s=background_size,
                alpha=background_alpha,
                rasterized=True,
                zorder=1,
            )
            if show_background_legend:
                bg_legend_elements.append(
                    Line2D(
                        [0], [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=6,
                        alpha=background_alpha,
                        label=str(cat_val),
                    )
                )
            if show_background_labels:
                cat_coords = embedding[cat_mask]
                centroid = cat_coords.mean(axis=0)
                ax.text(
                    centroid[0],
                    centroid[1],
                    str(cat_val),
                    fontsize=background_label_fontsize,
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    zorder=2,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.5,
                        edgecolor="none",
                    ),
                )
    else:
        bg_indices = np.where(bg_mask)[0]
        ax.scatter(
            embedding[bg_indices, 0],
            embedding[bg_indices, 1],
            c=background_color,
            s=background_size,
            alpha=background_alpha,
            rasterized=True,
            zorder=1,
        )

    # ---- Layer 2: area of interest ---- #
    if aoi_category is not None and aoi_values is not None:
        aoi_mask = adata.obs[aoi_category].isin(aoi_values)

        if aoi_mask.any():
            aoi_coords = embedding[aoi_mask.values]

            if aoi_color_by is not None and aoi_color_map is not None:
                aoi_labels = adata.obs.loc[aoi_mask, aoi_color_by]
                for val, color in aoi_color_map.items():
                    val_mask = (aoi_labels == val).values
                    if not val_mask.any():
                        continue
                    ax.scatter(
                        aoi_coords[val_mask, 0],
                        aoi_coords[val_mask, 1],
                        c=color,
                        s=aoi_size,
                        alpha=aoi_alpha,
                        rasterized=True,
                        zorder=3,
                    )
            else:
                ax.scatter(
                    aoi_coords[:, 0],
                    aoi_coords[:, 1],
                    c=aoi_default_color,
                    s=aoi_size,
                    alpha=aoi_alpha,
                    rasterized=True,
                    zorder=3,
                )

            # Optional AOI outline
            if show_aoi_outline:
                x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
                y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
                x_pad = (x_max - x_min) * 0.05
                y_pad = (y_max - y_min) * 0.05
                x_min -= x_pad
                x_max += x_pad
                y_min -= y_pad
                y_max += y_pad

                grid_size = 100
                xi = np.linspace(x_min, x_max, grid_size)
                yi = np.linspace(y_min, y_max, grid_size)
                Xi, Yi = np.meshgrid(xi, yi)

                aoi_grid = np.zeros((grid_size, grid_size))
                for x_c, y_c in aoi_coords:
                    x_idx = int(
                        (x_c - x_min) / (x_max - x_min) * (grid_size - 1)
                    )
                    y_idx = int(
                        (y_c - y_min) / (y_max - y_min) * (grid_size - 1)
                    )
                    if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                        aoi_grid[y_idx, x_idx] += 1

                aoi_grid = gaussian_filter(
                    aoi_grid, sigma=aoi_outline_sigma
                )

                if aoi_grid.max() > 0:
                    aoi_contours = ax.contour(
                        Xi, Yi, aoi_grid,
                        levels=[
                            aoi_grid.max() * aoi_outline_squeeze
                        ],
                        colors=[aoi_outline_color],
                        linewidths=aoi_outline_linewidth,
                        linestyles=aoi_outline_linestyle,
                        alpha=0.6,
                        zorder=4,
                    )

                    # Annotate the outline with text along the contour
                    if annotate_aoi:
                        level_val = (
                            aoi_grid.max() * aoi_outline_squeeze
                        )
                        clabels = ax.clabel(
                            aoi_contours,
                            levels=[level_val],
                            fmt={level_val: "Area of interest"},
                            fontsize=aoi_annotation_fontsize,
                            inline=True,
                            inline_spacing=5,
                            colors=aoi_outline_color,
                        )
                        for txt in clabels:
                            txt.set_fontweight("bold")
                            txt.set_zorder(7)

    # ---- Layer 2b: multiple AOI group outlines ---- #
    aoi_groups_legend_elements: List[Line2D] = []

    if aoi_groups is not None:
        # Resolve per-group colours
        grp_labels = list(aoi_groups.keys())
        n_grps = len(grp_labels)

        if aoi_groups_colors is None:
            _grp_colors = {lbl: "black" for lbl in grp_labels}
        elif isinstance(aoi_groups_colors, str):
            _grp_colors = {lbl: aoi_groups_colors for lbl in grp_labels}
        elif isinstance(aoi_groups_colors, list):
            if len(aoi_groups_colors) < n_grps:
                raise ValueError(
                    f"aoi_groups_colors list length "
                    f"({len(aoi_groups_colors)}) must be >= number of "
                    f"groups ({n_grps})"
                )
            _grp_colors = {
                lbl: aoi_groups_colors[i]
                for i, lbl in enumerate(grp_labels)
            }
        elif isinstance(aoi_groups_colors, dict):
            _grp_colors = {
                lbl: aoi_groups_colors.get(lbl, "black")
                for lbl in grp_labels
            }
        else:
            raise TypeError(
                "aoi_groups_colors must be None, str, list, or dict"
            )

        # Compute grid bounds once
        _x_min = embedding[:, 0].min()
        _x_max = embedding[:, 0].max()
        _y_min = embedding[:, 1].min()
        _y_max = embedding[:, 1].max()
        _x_pad = (_x_max - _x_min) * 0.05
        _y_pad = (_y_max - _y_min) * 0.05
        _x_min -= _x_pad
        _x_max += _x_pad
        _y_min -= _y_pad
        _y_max += _y_pad

        _gs = 100
        _xi = np.linspace(_x_min, _x_max, _gs)
        _yi = np.linspace(_y_min, _y_max, _gs)
        _Xi, _Yi = np.meshgrid(_xi, _yi)

        for grp_label, grp_def in aoi_groups.items():
            grp_color = _grp_colors[grp_label]

            # Build a boolean mask for all cells in this group
            grp_mask = np.zeros(adata.n_obs, dtype=bool)
            for col, vals in grp_def.items():
                grp_mask |= adata.obs[col].isin(vals).values

            if not grp_mask.any():
                continue

            grp_coords = embedding[grp_mask]

            # Build density grid
            grp_grid = np.zeros((_gs, _gs))
            for x_c, y_c in grp_coords:
                xi_idx = int(
                    (x_c - _x_min) / (_x_max - _x_min) * (_gs - 1)
                )
                yi_idx = int(
                    (y_c - _y_min) / (_y_max - _y_min) * (_gs - 1)
                )
                if 0 <= xi_idx < _gs and 0 <= yi_idx < _gs:
                    grp_grid[yi_idx, xi_idx] += 1

            grp_grid = gaussian_filter(
                grp_grid, sigma=aoi_groups_sigma
            )

            if grp_grid.max() > 0:
                ax.contour(
                    _Xi, _Yi, grp_grid,
                    levels=[grp_grid.max() * aoi_groups_squeeze],
                    colors=[grp_color],
                    linewidths=aoi_groups_linewidth,
                    linestyles=aoi_groups_linestyle,
                    alpha=0.6,
                    zorder=4,
                )

                if show_aoi_groups_legend:
                    aoi_groups_legend_elements.append(
                        Line2D(
                            [0], [0],
                            color=grp_color,
                            linewidth=aoi_groups_linewidth,
                            linestyle=aoi_groups_linestyle,
                            label=str(grp_label),
                        )
                    )

    # ---- Layer 3: density outlines ---- #
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    grid_size = 100
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    for category, cat_values in category_dict.items():
        density_col = f"{density_colname}_{category}"

        for cat in cat_values:
            # Resolve per-category density_cutoff
            _cutoff = (
                density_cutoff.get(str(cat), 1.0)
                if isinstance(density_cutoff, dict)
                else density_cutoff
            )
            # Resolve per-category contour_squeeze
            _squeeze = (
                contour_squeeze.get(str(cat), 0.1)
                if isinstance(contour_squeeze, dict)
                else contour_squeeze
            )

            cat_mask = adata.obs[category] == cat
            cat_densities = adata.obs.loc[cat_mask, density_col]
            high_density_mask = cat_densities > _cutoff

            if not high_density_mask.any():
                continue

            cat_indices = np.where(cat_mask)[0]
            hd_indices = cat_indices[high_density_mask.values]
            hd_coords = embedding[hd_indices]
            color = color_mapping[(category, cat)]

            try:
                density_grid = np.zeros((grid_size, grid_size))
                for x_c, y_c in hd_coords:
                    x_idx = int(
                        (x_c - x_min) / (x_max - x_min) * (grid_size - 1)
                    )
                    y_idx = int(
                        (y_c - y_min) / (y_max - y_min) * (grid_size - 1)
                    )
                    if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                        density_grid[y_idx, x_idx] += 1

                density_grid = gaussian_filter(density_grid, sigma=2)

                if density_grid.max() > 0:
                    contour_levels = [
                        density_grid.max() * _squeeze
                    ]
                    contours = ax.contour(
                        Xi, Yi, density_grid,
                        levels=contour_levels,
                        colors=[color],
                        linewidths=linewidth,
                        alpha=0.8,
                        zorder=5,
                    )

                    segs = contours.allsegs
                    if len(segs) > 0 and len(segs[0]) > 0:
                        label = (
                            str(cat)
                            if legend_remove_category_name
                            else f"{category.split('_')[-1]}: {cat}"
                        )
                        density_legend_elements.append(
                            Line2D(
                                [0], [0],
                                color=color,
                                linewidth=linewidth,
                                label=label,
                            )
                        )

                        if show_labels:
                            largest = max(segs[0], key=lambda p: len(p))
                            if len(largest) > 0:
                                centroid = np.array(largest).mean(axis=0)
                                ax.text(
                                    centroid[0],
                                    centroid[1],
                                    str(cat),
                                    fontsize=10,
                                    fontweight="bold",
                                    color=color,
                                    ha="center",
                                    va="center",
                                    zorder=6,
                                    bbox=dict(
                                        boxstyle="round,pad=0.3",
                                        facecolor="white",
                                        alpha=0.7,
                                        edgecolor=color,
                                    ),
                                )
            except Exception as e:
                print(
                    f"Warning: Could not create outline for "
                    f"{category}={cat}: {e}"
                )
                continue

    # ------------------------------------------------------------------ #
    # Labels, title, legend
    # ------------------------------------------------------------------ #
    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")
    ax.set_title(
        title if title is not None
        else "Area of Interest \u2013 Density Outlines"
    )
    ax.grid(False)

    # When drawn on an externally provided axes, skip figure-level
    # legend placement and layout adjustments – the caller handles those.
    if _external_ax:
        return fig

    # Use fig.legend() for right-margin legends so that multiple
    # legends accumulate without replacing each other (ax.legend()
    # replaces the previous one and add_artist() is fragile).
    _has_right_legends = False

    if density_legend_elements and show_density_legend:
        if legend_loc == "right margin" or legend_loc is None:
            fig.legend(
                handles=density_legend_elements,
                bbox_to_anchor=(1.0, 0.5),
                bbox_transform=ax.transAxes,
                loc="center left",
                frameon=True,
                #title="Density outlines",
            )
            _has_right_legends = True
        elif legend_loc == "on data":
            ax.legend(
                handles=density_legend_elements,
                loc="upper right",
                frameon=True,
                #title="Density outlines",
            )

    if aoi_groups_legend_elements and show_aoi_groups_legend:
        if legend_loc == "right margin" or legend_loc is None:
            fig.legend(
                handles=aoi_groups_legend_elements,
                bbox_to_anchor=(1.0, 0.85),
                bbox_transform=ax.transAxes,
                loc="upper left",
                frameon=True,
                title="AOI groups",
            )
            _has_right_legends = True
        elif legend_loc == "on data":
            ax.legend(
                handles=aoi_groups_legend_elements,
                loc="upper left",
                frameon=True,
                title="AOI groups",
            )

    if bg_legend_elements and show_background_legend:
        if legend_loc == "right margin" or legend_loc is None:
            fig.legend(
                handles=bg_legend_elements,
                bbox_to_anchor=(1.0, 0.0),
                bbox_transform=ax.transAxes,
                loc="lower left",
                frameon=True,
                title=str(background_color_by),
            )
            _has_right_legends = True
        elif legend_loc == "on data":
            ax.legend(
                handles=bg_legend_elements,
                loc="lower right",
                frameon=True,
                title=str(background_color_by),
            )

    if _has_right_legends:
        # Expand the figure so the *axes* keeps the requested figsize
        # and the legend occupies extra space on the right.
        _legend_pad = 2.0  # extra inches reserved for the legend
        fig.set_size_inches(figsize[0] + _legend_pad, figsize[1])
        fig.tight_layout(
            rect=[0, 0, figsize[0] / (figsize[0] + _legend_pad), 1]
        )
    else:
        fig.tight_layout()

    return fig


def aoi_density_outlines_multiplot(
    adata: AnnData,
    category_dict: Dict[str, List[str]],
    column: Optional[str] = None,
    categories: Optional[List[str]] = None,
    ncols: int = 2,
    figsize_per_panel: Tuple[float, float] = (6, 5),
    return_fig: bool = True,
    **kwargs,
) -> Optional[Figure]:
    """Plot a grid of AOI density-outline plots, one per category value.

    Two modes of operation:

    **With AOI** (*column* is provided):
        For each category value in *column* (or the subset given by
        *categories*) a panel is drawn using
        :func:`plot_area_of_interest_density_outlines` with
        ``aoi_category=column`` and ``aoi_values=[value]``.

    **Without AOI** (*column* is ``None``, the default):
        One panel per value found in *category_dict*.  Each panel shows
        the background embedding plus the density outline for that single
        value only (no area-of-interest layer is drawn).

    Args:
        adata: AnnData object with pre-computed embedding and density values.
        category_dict: Columns → category values forwarded to
            :func:`plot_area_of_interest_density_outlines` for the density
            outlines layer.
        column: Column in ``adata.obs`` whose categorical values define the
            individual panels **and** serve as the AOI category.  When
            ``None`` (the default), panels are derived from the values in
            *category_dict* and no AOI layer is drawn.
        categories: Subset of values to plot.  Interpreted as values from
            *column* when *column* is given, or as ``(category_key, value)``
            labels when *column* is ``None`` (not commonly needed).
            If ``None``, all values are used.
        ncols: Number of columns in the subplot grid (default: 2).
        figsize_per_panel: ``(width, height)`` of each individual panel.
            The total figure size is derived automatically.
        return_fig: Whether to return the ``Figure`` object (default ``True``).
            If ``False`` the figure is displayed and ``None`` is returned.
        **kwargs: Additional keyword arguments forwarded to
            :func:`plot_area_of_interest_density_outlines`.  Do **not** pass
            ``aoi_category``, ``aoi_values``, ``figsize``, ``ax``, or
            ``category_dict`` here – they are managed internally.

    Returns:
        ``matplotlib.figure.Figure`` | ``None``

    Raises:
        ValueError: If *column* is not found in ``adata.obs`` or no
            categories are available.

    Examples:
        >>> import scanpy as sc
        >>> from density_plotting import aoi_density_outlines_multiplot
        >>>
        >>> adata = sc.read_h5ad('data.h5ad')
        >>> sc.tl.embedding_density(adata, basis='umap', groupby='leiden')
        >>>
        >>> # --- Without AOI: one panel per density-outline category ---
        >>> fig = aoi_density_outlines_multiplot(
        ...     adata,
        ...     category_dict={'leiden': ['0', '1', '2']},
        ...     ncols=3,
        ... )
        >>>
        >>> # --- With AOI: one panel per cell-type, density outlines from leiden ---
        >>> fig = aoi_density_outlines_multiplot(
        ...     adata,
        ...     category_dict={'leiden': ['0', '1', '2']},
        ...     column='cell_type',
        ...     ncols=3,
        ... )
        >>>
        >>> # Only selected cell-type categories
        >>> fig = aoi_density_outlines_multiplot(
        ...     adata,
        ...     category_dict={'leiden': ['0', '1', '2']},
        ...     column='cell_type',
        ...     categories=['T cell', 'B cell'],
        ... )
    """
    # ------------------------------------------------------------------ #
    # Build the list of panels
    # ------------------------------------------------------------------ #
    #  _panels: list of dicts that are passed as **extra to the inner call
    _managed_keys = {
        "aoi_category", "aoi_values", "figsize", "ax", "category_dict",
    }
    clean_kwargs = {
        k: v for k, v in kwargs.items() if k not in _managed_keys
    }

    if column is not None:
        # ---------- AOI mode: one panel per value in column ---------- #
        if column not in adata.obs.columns:
            raise ValueError(
                f"Column '{column}' not found in adata.obs. "
                "Choose a valid column."
            )
        if not adata.obs[column].dtype.name == "category":
            adata.obs[column] = adata.obs[column].astype("category")

        if categories is None:
            categories = adata.obs[column].cat.categories.tolist()

        panels: List[dict] = [
            {
                "title": str(val),
                "category_dict": category_dict,
                "aoi_category": column,
                "aoi_values": [val],
            }
            for val in categories
        ]
    else:
        # ---- No-AOI mode: one panel per value in category_dict ---- #
        flat_values: List[Tuple[str, str]] = [
            (cat_key, val)
            for cat_key, vals in category_dict.items()
            for val in vals
        ]
        if categories is not None:
            # Allow filtering by value name
            flat_values = [
                (k, v) for k, v in flat_values if v in categories
            ]

        panels = [
            {
                "title": str(val),
                "category_dict": {cat_key: [val]},
                # no AOI
            }
            for cat_key, val in flat_values
        ]

    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError("No categories to plot.")

    # ------------------------------------------------------------------ #
    # Create subplot grid
    # ------------------------------------------------------------------ #
    nrows = int(np.ceil(n_panels / ncols))
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h)
    )

    # Ensure axs is always a flat array (even for 1-row or 1-col grids)
    axs = np.atleast_1d(axs).flat

    # ------------------------------------------------------------------ #
    # Fill each subplot
    # ------------------------------------------------------------------ #
    for panel, cur_ax in zip(panels, axs):
        call_kwargs = dict(clean_kwargs)
        call_kwargs["ax"] = cur_ax
        call_kwargs["title"] = panel["title"]
        call_kwargs["category_dict"] = panel["category_dict"]
        if "aoi_category" in panel:
            call_kwargs["aoi_category"] = panel["aoi_category"]
            call_kwargs["aoi_values"] = panel["aoi_values"]

        plot_area_of_interest_density_outlines(adata, **call_kwargs)

    # Turn off unused axes
    for idx in range(n_panels, nrows * ncols):
        axs[idx].axis("off")

    plt.tight_layout()

    if return_fig:
        return fig
    return None