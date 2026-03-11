from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.figure import Figure

try:
    from scanpy.plotting._utils import set_colors_for_categorical_obs, set_default_colors_for_categorical_obs
except ImportError:
    from scanpy.plotting._utils import _set_colors_for_categorical_obs as set_colors_for_categorical_obs
    from scanpy.plotting._utils import _set_default_colors_for_categorical_obs as set_default_colors_for_categorical_obs
    
def is_sequence(obj: Any) -> bool:
    """Return ``True`` when *obj* supports ``__len__`` and ``__getitem__``."""
    t = type(obj)
    return hasattr(t, "__len__") and hasattr(t, "__getitem__")


def flatten_list_of_lists(nested: list[list[Any]]) -> list[Any]:
    """Flatten one level of nesting from a list of lists."""
    return [item for sublist in nested for item in sublist]

def embedding_category_multiplot(
    adata: sc.AnnData,
    column: str,
    palette: Union[str, Sequence] = None,
    ncols: int = None,
    return_fig: bool = False,
    use_legend: bool = False,
    basis: str = "umap",
    figsize: tuple[float, float] = (4, 4),
    hspace: float = None,
    wspace: float = None,
    groups: Sequence = None,
    sort_order: bool = True,
    **kwargs,  # Keyword arguments forwarded to sc.pl.embedding()
) -> Figure:
    """Plot embedding panels showing one categorical value per subplot.

    Creates a matplotlib grid and fills each cell with an embedding produced
    by :func:`scanpy.pl.embedding`, highlighting only one category per panel.

    Args:
        adata: Annotated data matrix.
        column: Name of the ``adata.obs`` column with categorical values.
        palette: Palette to use.  Accepts a single colour string (all
            categories share that colour), a list of colours, or a matplotlib
            colormap / seaborn palette name.  When ``None``, colours from
            ``adata.uns`` are used if available, otherwise defaults are set.
        ncols: Number of columns in the grid.  Defaults to
            ``ceil(sqrt(n_groups))`` for an approximately square grid.
        return_fig: When ``True`` the figure is returned.  Deprecated — the
            figure is always returned regardless of this flag.
        use_legend: When ``True``, overlays the category label directly on the
            embedding (``legend_loc="on data"``).  Defaults to ``False``.
        basis: Embedding key in ``adata.obsm`` (e.g. ``"umap"``, ``"pca"``).
            Defaults to ``"umap"``.
        figsize: ``(width, height)`` of each individual subplot in inches.
            Defaults to ``(4, 4)``.
        hspace: Vertical spacing between subplots.  Passed to
            :func:`matplotlib.pyplot.subplots_adjust`.
        wspace: Horizontal spacing between subplots.  Passed to
            :func:`matplotlib.pyplot.subplots_adjust`.
        groups: Subset of category values to plot.  When ``None``, all
            categories are plotted.
        sort_order: Plot the highest values on top for continuous annotations.
            Defaults to ``True``.
        **kwargs: Additional keyword arguments forwarded to
            :func:`scanpy.pl.embedding`.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: When *column* is not found in ``adata.obs.columns``.
    """
    
    
    # Check if `column` is in adata.obs.columns
    if column not in adata.obs.columns:
        raise ValueError("Input category not found in adata.obs columns. Choose a valid column.")
        
    # If column is not categorical, change the dtype
    if not adata.obs[column].dtype.name == 'category':
        adata.obs[column] = adata.obs[column].astype("category")
       
    # Save original colors to restore after plotting (avoid mutating adata.uns)
    _original_colors = adata.uns.get(f"{column}_colors", None)

    # Set palette in adata.uns[f"{column}_colors"]
    if palette is not None and matplotlib.colors.is_color_like(palette):
        # Single color name - all categories plotted with the same color
        adata.uns[f"{column}_colors"] = [matplotlib.colors.to_hex(palette)] * len(adata.obs[column].cat.categories)
    
    elif isinstance(palette, str):
        # Matplotlib palette/colormap name
        set_colors_for_categorical_obs(adata, value_to_plot = column,  palette=palette)
    
    elif is_sequence(palette):
        # Extend palette if needed
        adata.uns[f"{column}_colors"] = flatten_list_of_lists(
            [palette for _ in range(math.ceil(len(adata.obs[column].cat.categories) / len(palette)))])[:len(adata.obs[column].cat.categories)]
    
    else:
        if f"{column}_colors" not in adata.uns: 
            set_default_colors_for_categorical_obs(adata, value_to_plot = column)
    
    
    
    # Determine categories to plot
    all_cats = list(adata.obs[column].cat.categories)
    if groups is not None:
        cats_to_plot = [c for c in all_cats if c in groups]
    else:
        cats_to_plot = all_cats
    
    # Compute ncols for square grid if not specified
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(len(cats_to_plot))))

    # Make subplots
    nrows = int(np.ceil(len(cats_to_plot) / ncols ))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    if nrows * ncols == 1:
        axs = np.array([axs])
    
    # Parse legend_loc (in case legend wanted on top of the plot)
    if use_legend == True:
        legend_loc = "on data"
    else:
        legend_loc = None
    
    
    # Fill subplots
    for category, ax in zip(cats_to_plot,
                                   axs.flat[:len(cats_to_plot)]):

        
        if "show" in kwargs:
            del kwargs["show"]
            
        sc.pl.embedding(adata,
                        basis=basis,
                        color=[column],
                        groups=[category],
                        ax=ax,
                        show=False,
                        legend_loc=legend_loc,
                        title=category,
                        na_in_legend=False,
                        sort_order=sort_order,
                        **kwargs)
 
    # Restore original colors to avoid mutating adata.uns
    if _original_colors is not None:
        adata.uns[f"{column}_colors"] = _original_colors
    elif f"{column}_colors" in adata.uns:
        del adata.uns[f"{column}_colors"]

    # Remove empty axes
    for ax in axs.flat[len(cats_to_plot):]:
        ax.axis('off')
        
    # Tight layout so plots don't overlap
    plt.tight_layout()
    if hspace is not None or wspace is not None:
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
    
    # Return figure object
    return fig


def embedding_gene_expression_multiplot(
    adata: sc.AnnData,
    column: str,
    feature: str,
    ncols: int = None,
    return_fig: bool = False,
    basis: str = "umap",
    vmin: Union[str, float] = 0.0,
    vmax: Union[str, float] = "p95",
    layer: str = None,
    size: float = None,
    figsize: tuple[float, float] = (4, 4),
    hspace: float = None,
    wspace: float = None,
    groups: Sequence = None,
    shared_colorscale: bool = True,
    sort_order: bool = True,
    **kwargs,  # Keyword arguments forwarded to sc.pl.embedding()
) -> Figure:
    """Plot embedding panels split by a categorical column, coloured by a gene or obs feature.

    Creates a matplotlib grid and fills each panel with an embedding plot using
    :func:`scanpy.pl.embedding`.  Each panel shows only cells from one category
    and colours them by *feature*.

    Args:
        adata: Annotated data matrix.
        column: ``adata.obs`` column used to split panels.
        feature: Gene name or ``adata.obs`` column to use as the colour
            variable.  Resolved against ``adata.obs.columns`` first, then
            ``adata.var_names`` (or ``adata.var[gene_symbols]`` when
            *gene_symbols* is supplied via ``**kwargs``).
        ncols: Number of columns in the grid.  Defaults to
            ``ceil(sqrt(n_groups))``.
        return_fig: When ``True`` the figure is returned.  Deprecated — the
            figure is always returned regardless of this flag.
        basis: Embedding key in ``adata.obsm``.  Defaults to ``"umap"``.
        vmin: Lower colour-scale limit.  Plain float or percentile string
            such as ``"p5"``.  Defaults to ``0.0``.
        vmax: Upper colour-scale limit.  Plain float or percentile string
            such as ``"p95"``.  Defaults to ``"p95"``.
        layer: AnnData layer to use for gene expression.  Ignored when
            *feature* is an obs column.  Defaults to ``None`` (use ``adata.X``).
        size: Scatter point size.  Defaults to ``120000 / n_cells``.
        figsize: ``(width, height)`` of each individual subplot in inches.
            Defaults to ``(4, 4)``.
        hspace: Vertical spacing between subplots.
        wspace: Horizontal spacing between subplots.
        groups: Subset of category values to plot.  When ``None``, all
            categories are plotted.
        shared_colorscale: When ``True``, percentile-based ``vmin``/``vmax``
            are computed globally across all cells.  When ``False``, each
            panel gets its own colour scale. Defaults to ``True``.
        sort_order: Plot the highest values on top.  Defaults to ``True``.
        **kwargs: Additional keyword arguments forwarded to
            :func:`scanpy.pl.embedding`.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: When *column* is not found in ``adata.obs.columns``.
        ValueError: When *feature* is not found in ``adata.obs.columns`` or
            ``adata.var_names``.
    """
    
    
    # Check if `column` is in adata.obs.columns
    if column not in adata.obs.columns:
        raise ValueError("Input category not found in adata.obs columns. Choose a valid column.")
        
    gene_symbols = kwargs.get("gene_symbols", None)

    # Resolve feature: obs column takes priority, then var
    _is_obs_col = feature in adata.obs.columns
    if not _is_obs_col:
        if gene_symbols is not None:
            if feature not in adata.var[gene_symbols].tolist():
                raise ValueError(
                    f"Feature '{feature}' not found in adata.obs.columns or "
                    f"adata.var['{gene_symbols}']. Choose a valid obs column or gene."
                )
        else:
            if feature not in adata.var_names.tolist():
                raise ValueError(
                    f"Feature '{feature}' not found in adata.obs.columns or "
                    f"adata.var_names. Choose a valid obs column or gene."
                )

    # layer and gene_symbols are only meaningful for var (gene) features
    _layer = None if _is_obs_col else layer
    _gene_symbols = None if _is_obs_col else gene_symbols
        
        
    # If column is not categorical, change the dtype
    if not adata.obs[column].dtype.name == 'category':
        adata.obs[column] = adata.obs[column].astype("category")
    
    # Determine categories to plot
    all_cats = list(adata.obs[column].cat.categories)
    if groups is not None:
        cats_to_plot = [c for c in all_cats if c in groups]
    else:
        cats_to_plot = all_cats
    
    # Compute ncols for square grid if not specified
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(len(cats_to_plot))))

    # Make subplots
    nrows = int(np.ceil(len(cats_to_plot) / ncols ))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize = (figsize[0] * ncols, figsize[1] * nrows))
    if nrows * ncols == 1:
        axs = np.array([axs])
        
    # Set vmin and vmax - resolve globally when using a shared colorscale
    if shared_colorscale:
        if vmax is not None:
            values = sc.get.obs_df(adata, keys=[feature], use_raw=False, layer=_layer, gene_symbols=_gene_symbols)[feature].values
            vmax = np.quantile(values, float(vmax.replace("p", "")) / 100) if str(vmax).startswith("p") else float(vmax)

        if vmin is not None:
            values = sc.get.obs_df(adata, keys=[feature], use_raw=False, layer=_layer, gene_symbols=_gene_symbols)[feature].values
            vmin = np.quantile(values, float(vmin.replace("p", "")) / 100) if str(vmin).startswith("p") else float(vmin)
    
    # Set dot size of scatter
    if size is None:
        size = 120000 / adata.shape[0]
        
        
    # Deal with `**kwargs`
    extra_kwargs_dict = dict(
         legend_fontoutline = 2,
         legend_fontsize = 11,
         add_outline=True,
         outline_width=(0.2, 0.1),
         outline_color=('black', 'lightgray'),

    )
    
    combined_kwargs_dict = dict(kwargs)
    for k, v in extra_kwargs_dict.items():
        if k not in combined_kwargs_dict:
            combined_kwargs_dict[k] = v
            
    ## kwargs to remove (computed in function)
    for k in ["vmin", "vmax", "size", 'layer', 'sort_order']:
         combined_kwargs_dict.pop(k, None)
    if _is_obs_col:
        # gene_symbols is irrelevant when colouring by an obs column
        combined_kwargs_dict.pop("gene_symbols", None)
    
    # Fill subplots
    for category, ax in zip(cats_to_plot,
                                   axs.flat[:len(cats_to_plot)]):

        if not shared_colorscale:
            cat_data = sc.get.obs_df(adata[adata.obs[column] == category], keys=[feature], use_raw=False, layer=_layer, gene_symbols=_gene_symbols)[feature].values
            _vmax = np.quantile(cat_data, float(vmax.replace("p", "")) / 100) if vmax is not None and str(vmax).startswith("p") else (float(vmax) if vmax is not None else None)
            _vmin = np.quantile(cat_data, float(vmin.replace("p", "")) / 100) if vmin is not None and str(vmin).startswith("p") else (float(vmin) if vmin is not None else None)
        else:
            _vmax = vmax
            _vmin = vmin

        sc.pl.embedding(adata,
                        basis=basis,
                        ax=ax,
                        show=False,
                        size = size,
                        sort_order=sort_order,
                        **combined_kwargs_dict)
        
        sc.pl.embedding(adata[adata.obs[column] == category],
                        basis=basis,
                        color=[feature],
                        ax=ax,
                        show=False,
                        title=category,
                        vmin = _vmin,
                        vmax = _vmax,
                        size = size,
                        layer = _layer,
                        sort_order=sort_order,
                        **combined_kwargs_dict)
 
    # Remove empty axes
    for ax in axs.flat[len(cats_to_plot):]:
        ax.axis('off')
        
    # Tight layout so plots don't overlap
    plt.tight_layout()
    if hspace is not None or wspace is not None:
        plt.subplots_adjust(hspace=hspace, wspace=wspace)

    # Return figure object
    return fig