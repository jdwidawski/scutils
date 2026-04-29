from __future__ import annotations

import itertools
import math
from typing import Tuple, Any, Dict, Optional, Sequence, Union

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.figure import Figure
from anndata import AnnData

from scutils.plotting._utils import _resolve_palette

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


# ---------------------------------------------------------------------------
# 3-D interactive UMAP (Plotly)
# ---------------------------------------------------------------------------


def _resolve_obsm_key(adata: sc.AnnData, basis: str) -> str:
    """Resolve an ``adata.obsm`` key, prepending ``"X_"`` when needed.

    Args:
        adata: Annotated data matrix.
        basis: Either the exact obsm key (e.g. ``"X_umap"``) or a short alias
            without the ``"X_"`` prefix (e.g. ``"umap"``).

    Returns:
        The matching key present in ``adata.obsm``.

    Raises:
        KeyError: If neither *basis* nor ``"X_{basis}"`` is found.
    """
    if basis in adata.obsm:
        return basis
    prefixed = f"X_{basis}"
    if prefixed in adata.obsm:
        return prefixed
    raise KeyError(
        f"Embedding key '{basis}' (or 'X_{basis}') not found in adata.obsm. "
        f"Available keys: {list(adata.obsm.keys())}"
    )


def _resolve_vbound(
    values: np.ndarray,
    bound: Optional[Union[float, str]],
) -> Optional[float]:
    """Resolve a colour-scale bound, supporting percentile strings like ``'p5'``.

    Args:
        values: 1-D numeric array (NaN-safe) used for percentile computation.
        bound: Either a plain float or a string of the form ``"pXX"``
            (e.g. ``"p95"``).  ``None`` is returned unchanged.

    Returns:
        The resolved float bound, or ``None``.
    """
    if bound is None:
        return None
    if isinstance(bound, str) and bound.startswith("p"):
        pct = float(bound[1:])
        return float(np.nanpercentile(values, pct))
    return float(bound)


def _categorical_color_map(
    adata: sc.AnnData,
    key: str,
    palette: Optional[Union[str, Sequence[Any], Dict[str, str]]],
) -> Dict[str, str]:
    """Build a ``{category: hex_colour}`` mapping for a categorical obs column.

    Resolution order:

    1. *palette* (forwarded to
       :func:`~scutils.plotting._utils._resolve_palette`).
    2. ``adata.uns["{key}_colors"]``.
    3. Scanpy's default colour cycle (``adata.uns`` state is restored
       afterwards to avoid mutation).

    Args:
        adata: Annotated data matrix.
        key: Column in ``adata.obs`` with a categorical dtype.
        palette: Colour specification — see
            :func:`~scutils.plotting._utils._resolve_palette`.

    Returns:
        Mapping from category label (str) to CSS hex colour.
    """
    categories = list(adata.obs[key].cat.categories)

    if palette is not None:
        resolved = _resolve_palette(palette, [str(c) for c in categories])
        if resolved is not None:
            return {str(c): mcolors.to_hex(v) for c, v in resolved.items()}

    if f"{key}_colors" in adata.uns:
        uns_colors = list(adata.uns[f"{key}_colors"])
        cycled = list(
            itertools.islice(itertools.cycle(uns_colors), len(categories))
        )
        return {
            str(cat): mcolors.to_hex(col)
            for cat, col in zip(categories, cycled)
        }

    # Fall back: use scanpy defaults temporarily, then restore uns state.
    _saved = adata.uns.get(f"{key}_colors", None)
    set_default_colors_for_categorical_obs(adata, value_to_plot=key)
    uns_colors = list(adata.uns[f"{key}_colors"])
    cycled = list(itertools.islice(itertools.cycle(uns_colors), len(categories)))
    color_map = {
        str(cat): mcolors.to_hex(col) for cat, col in zip(categories, cycled)
    }
    if _saved is None:
        adata.uns.pop(f"{key}_colors", None)
    else:
        adata.uns[f"{key}_colors"] = _saved
    return color_map


def _get_obs_or_var_values(
    adata: sc.AnnData,
    key: str,
    layer: Optional[str],
) -> np.ndarray:
    """Retrieve numeric values for *key* from obs or the expression matrix.

    Args:
        adata: Annotated data matrix.
        key: Column in ``adata.obs`` (returned as-is) or a gene name in
            ``adata.var_names``.
        layer: AnnData layer used for gene retrieval.  ``None`` means
            ``adata.X``.

    Returns:
        1-D float32 NumPy array of length ``n_obs``.

    Raises:
        ValueError: If *key* is not found in ``adata.obs.columns`` or
            ``adata.var_names``.
    """
    from scipy.sparse import issparse

    if key in adata.obs.columns:
        return np.asarray(adata.obs[key].values, dtype=np.float32)
    if key not in adata.var_names:
        raise ValueError(
            f"Key '{key}' not found in adata.obs.columns or adata.var_names."
        )
    gene_idx = adata.var_names.get_loc(key)
    X = adata.layers[layer] if layer is not None else adata.X
    col = X[:, gene_idx]
    if issparse(col):
        vals = np.asarray(col.todense()).ravel()
    else:
        vals = np.asarray(col).ravel()
    return vals.astype(np.float32)


def umap_3d(
    adata: sc.AnnData,
    color: Optional[Union[str, Sequence[str]]] = None,
    *,
    groups: Optional[Sequence[str]] = None,
    palette: Optional[Union[str, Sequence[Any], Dict[str, str]]] = None,
    layer: Optional[str] = None,
    vmin: Optional[Union[float, str]] = None,
    vmax: Optional[Union[float, str]] = None,
    na_color: str = "lightgray",
    basis: str = "X_umap",
    size: float = 3.0,
    alpha: float = 0.8,
    title: Optional[Union[str, Sequence[str]]] = None,
    width: int = 900,
    height: int = 700,
    colorscale: str = "Viridis",
    colorbar_title: Optional[str] = None,
    hover_keys: Optional[Sequence[str]] = None,
    sort_order: bool = True,
    ncols: Optional[int] = None,
    merge_traces: bool = True,
    max_cells: Optional[int] = None,
    random_state: int = 0,
) -> "go.Figure":  # noqa: F821 – plotly imported lazily
    """Interactive 3-D UMAP scatter plot powered by Plotly WebGL.

    Visualises a three-component UMAP embedding stored in ``adata.obsm`` as
    an interactive Plotly figure.  Colour handling mirrors
    :func:`scanpy.pl.umap`: categorical variables are coloured from
    ``adata.uns``, continuous variables use a sequential colour scale, and
    gene-expression values are fetched from the expression matrix or a named
    layer.

    The function is optimised for large datasets (≥10⁶ cells) by:

    * Extracting coordinates as ``float32`` arrays to minimise memory and
      serialisation overhead.
    * When *merge_traces* is ``True`` (default), all categories are collapsed
      into a **single** WebGL trace with a flat colour array.  This is the
      most effective optimisation: rotation and zoom are ~10× faster because
      the browser issues only one WebGL draw call per panel.  Invisible
      zero-size legend traces are added so the legend remains interactive.
    * When *merge_traces* is ``False``, one trace per category is added,
      allowing per-category visibility toggling at the cost of slower
      interaction.
    * Defaulting to ``hoverinfo="skip"`` on background traces so the browser
      does not hit-test hidden geometry.
    * Accepting an optional ``hover_keys`` list; extra columns are only
      serialised as ``customdata`` when explicitly requested.
    * Accepting *max_cells* for random downsampling before any rendering,
      which linearly reduces GPU memory and serialisation time.

    When *color* is a list, a subplot grid is created (up to *ncols* columns).

    Args:
        adata: Annotated data matrix.  The embedding must have been computed
            with ``scanpy.tl.umap(adata, n_components=3)``.
        color: ``adata.obs`` column name(s) or gene name(s) used to colour
            cells.  Accepts a single string or a list of strings.  When
            ``None``, all cells are plotted in a uniform grey.
        groups: Subset of category values to highlight (categorical *color*
            only).  Cells outside the selection are rendered with *na_color*
            at reduced opacity.  Has no effect for continuous variables.
        palette: Colour specification for categorical variables.  Accepts a
            single colour string (all categories share that colour), a list
            of colours, a ``{category: colour}`` dict, or a
            matplotlib / seaborn palette name.  When ``None``, colours are
            taken from ``adata.uns["{color}_colors"]`` if available, otherwise
            Scanpy defaults are used.
        layer: AnnData layer to use when *color* is a gene name.  ``None``
            means ``adata.X``.  Ignored for ``adata.obs`` columns.
        vmin: Lower colour-scale limit for continuous variables.  Plain float
            or percentile string such as ``"p5"``.
        vmax: Upper colour-scale limit for continuous variables.  Plain float
            or percentile string such as ``"p95"``.
        na_color: Colour for cells outside *groups* (when *groups* is set)
            or for NaN values in continuous variables.  Defaults to
            ``"lightgray"``.
        basis: Embedding key in ``adata.obsm``.  Accepts the full key
            (``"X_umap"``) or the short alias (``"umap"``).  The embedding
            must have at least 3 components.  Defaults to ``"X_umap"``.
        size: Marker diameter in pixels.  Smaller values improve rendering
            speed for large datasets.  Defaults to ``3.0``.
        alpha: Marker opacity in ``[0, 1]``.  Defaults to ``0.8``.
        title: Panel title(s).  Defaults to the *color* key name(s).
        width: Figure width in pixels per column.  Total figure width is
            ``width × ncols``.  Defaults to ``900``.
        height: Figure height in pixels per row.  Total figure height is
            ``height × nrows``.  Defaults to ``700``.
        colorscale: Plotly named colour scale used for continuous variables
            (e.g. ``"Viridis"``, ``"RdBu"``, ``"Plasma"``).  Defaults to
            ``"Viridis"``.
        colorbar_title: Title text for the colour bar of continuous variables.
            Defaults to the *color* key name.
        hover_keys: Additional ``adata.obs`` column names to include in the
            hover tooltip.  When ``None`` (default), only coordinates and the
            trace / category name are shown — the most performant option for
            very large datasets.
        sort_order: When ``True``, cells with higher values are drawn last
            (on top) for continuous variables.  Defaults to ``True``.
        ncols: Number of subplot columns when *color* is a list.  Defaults
            to ``min(n_colors, 3)``.
        merge_traces: When ``True`` (default), collapse all categories into a
            single trace with a flat colour array.  Dramatically improves
            rotation/zoom responsiveness — recommended for datasets with many
            cells or many categories.  Legend entries remain visible but
            clicking them no longer toggles individual categories.
        max_cells: If set, randomly subsample to at most this many cells
            before building the figure.  Useful for exploratory views of very
            large datasets.  Defaults to ``None`` (no downsampling).
        random_state: Random seed for *max_cells* downsampling.  Defaults to
            ``0``.

    Returns:
        A :class:`plotly.graph_objects.Figure` containing one 3-D scene per
        colour variable.  Display it with
        :func:`scutils.pl.show_plotly` in a notebook or call ``.show()``
        in a script.

    Raises:
        KeyError: If *basis* (or ``"X_{basis}"``) is not found in
            ``adata.obsm``.
        ValueError: If the resolved embedding has fewer than 3 components.
        ValueError: If a *color* key is not found in ``adata.obs.columns``
            or ``adata.var_names``.

    Example:
        >>> sc.tl.umap(adata, n_components=3)
        >>> fig = scutils.pl.umap_3d(adata, color="leiden")
        >>> scutils.pl.show_plotly(fig)

        Colour by gene expression with a custom palette and subplots::

        >>> fig = scutils.pl.umap_3d(
        ...     adata,
        ...     color=["leiden", "CD3E", "CD19"],
        ...     palette="tab20",
        ...     vmax="p99",
        ...     ncols=2,
        ... )
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── 1. Resolve embedding ───────────────────────────────────────────────
    obsm_key = _resolve_obsm_key(adata, basis)
    coords = adata.obsm[obsm_key]
    if coords.shape[1] < 3:
        raise ValueError(
            f"Embedding '{obsm_key}' has only {coords.shape[1]} component(s); "
            "umap_3d requires at least 3.  Re-run sc.tl.umap with n_components=3."
        )
    x = np.asarray(coords[:, 0], dtype=np.float32)
    y = np.asarray(coords[:, 1], dtype=np.float32)
    z = np.asarray(coords[:, 2], dtype=np.float32)

    # ── 1b. Optional random downsampling ──────────────────────────────────
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        idx.sort()  # keep original cell order
        adata = adata[idx]
        x, y, z = x[idx], y[idx], z[idx]

    # ── 2. Normalise color / title to lists ────────────────────────────────
    if color is None:
        color_list: list[Optional[str]] = [None]
    elif isinstance(color, str):
        color_list = [color]
    else:
        color_list = list(color)

    n_panels = len(color_list)

    if title is None:
        title_list = [str(c) if c is not None else "" for c in color_list]
    elif isinstance(title, str):
        title_list = [title] + [""] * (n_panels - 1)
    else:
        title_list = list(title)
        title_list += [""] * (n_panels - len(title_list))

    # ── 3. Subplot grid ────────────────────────────────────────────────────
    _ncols = ncols if ncols is not None else min(n_panels, 3)
    _nrows = math.ceil(n_panels / _ncols)
    specs = [[{"type": "scene"} for _ in range(_ncols)] for _ in range(_nrows)]

    # Pad title list to fill the grid (empty panels have no title)
    padded_titles = title_list[:n_panels] + [""] * (_nrows * _ncols - n_panels)
    fig = make_subplots(
        rows=_nrows,
        cols=_ncols,
        specs=specs,
        subplot_titles=padded_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    # ── 4. Build and add traces ────────────────────────────────────────────
    for panel_idx, col_key in enumerate(color_list):
        row = panel_idx // _ncols + 1
        col = panel_idx % _ncols + 1

        # Colorbar x position: placed at the right edge of this panel's column
        # in paper coordinates to prevent overlap in multi-panel figures.
        _cbar_x: float = (col / _ncols) - 0.01 if _ncols > 1 else 1.02

        # ── None: uniform colour ──────────────────────────────────────────
        if col_key is None:
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=size,
                        color="steelblue",
                        opacity=alpha,
                        line=dict(width=0),
                    ),
                    name="",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            continue

        # ── Detect variable type ──────────────────────────────────────────
        is_obs = col_key in adata.obs.columns
        if is_obs:
            series = adata.obs[col_key]
            is_categorical = hasattr(series, "cat") or series.dtype == object
        else:
            is_categorical = False

        if is_categorical:
            # ── Categorical obs column ────────────────────────────────────
            if not hasattr(adata.obs[col_key], "cat"):
                adata.obs[col_key] = adata.obs[col_key].astype("category")

            categories = list(adata.obs[col_key].cat.categories)
            color_map = _categorical_color_map(adata, col_key, palette)
            cat_values = adata.obs[col_key].values

            active_cats = (
                [c for c in categories if str(c) in {str(g) for g in groups}]
                if groups is not None
                else categories
            )

            # Background trace for cells outside the selected groups
            if groups is not None:
                active_str = {str(g) for g in groups}
                inactive_mask = np.array(
                    [str(v) not in active_str for v in cat_values], dtype=bool
                )
                if inactive_mask.any():
                    fig.add_trace(
                        go.Scatter3d(
                            x=x[inactive_mask],
                            y=y[inactive_mask],
                            z=z[inactive_mask],
                            mode="markers",
                            marker=dict(
                                size=size,
                                color=na_color,
                                opacity=alpha * 0.3,
                                line=dict(width=0),
                            ),
                            name="other",
                            legendgroup=f"{col_key}_other_{panel_idx}",
                            showlegend=True,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

            # ── Per-cell colour array (used by both paths) ──────────────
            cat_str = np.asarray(cat_values, dtype=str)
            cell_colors = np.array(
                [color_map.get(v, na_color) for v in cat_str]
            )
            # Mask out inactive cells (groups filter)
            active_str_set = {str(c) for c in active_cats}
            active_mask = np.array(
                [v in active_str_set for v in cat_str], dtype=bool
            )

            if merge_traces:
                # ── Single merged trace: one WebGL draw call per panel ────
                # Inactive cells already rendered by the background trace;
                # here we render only active cells with their colours.
                hover_kw: dict[str, Any] = {}
                if hover_keys:
                    valid_hk = [k for k in hover_keys if k in adata.obs.columns]
                    if valid_hk:
                        cd = np.column_stack(
                            [
                                adata.obs[k].values[active_mask].astype(str)
                                for k in valid_hk
                            ]
                        )
                        extra = "<br>".join(
                            f"{k}: %{{customdata[{i}]}}"
                            for i, k in enumerate(valid_hk)
                        )
                        hover_kw["customdata"] = cd
                        hover_kw["hovertemplate"] = (
                            f"%{{text}}<br>{extra}<extra></extra>"
                        )
                        hover_kw["text"] = cat_str[active_mask].tolist()

                fig.add_trace(
                    go.Scatter3d(
                        x=x[active_mask],
                        y=y[active_mask],
                        z=z[active_mask],
                        mode="markers",
                        marker=dict(
                            size=size,
                            color=cell_colors[active_mask].tolist(),
                            opacity=alpha,
                            line=dict(width=0),
                        ),
                        name=col_key,
                        showlegend=False,
                        hoverinfo="skip" if not hover_keys else None,
                        **hover_kw,
                    ),
                    row=row,
                    col=col,
                )

                # Add zero-size legend-only traces (no geometry, no GPU cost)
                for cat in active_cats:
                    cat_color = color_map.get(str(cat), na_color)
                    fig.add_trace(
                        go.Scatter3d(
                            x=[None],
                            y=[None],
                            z=[None],
                            mode="markers",
                            marker=dict(
                                size=8,
                                color=cat_color,
                                opacity=1.0,
                                line=dict(width=0),
                            ),
                            name=str(cat),
                            legendgroup=f"{col_key}_{cat}",
                            showlegend=True,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

            else:
                # ── One trace per category (allows legend toggling) ───────
                for cat in active_cats:
                    mask = cat_str == str(cat)
                    if not mask.any():
                        continue

                    cat_color = color_map.get(str(cat), na_color)

                    hover_kw: dict[str, Any] = {}
                    if hover_keys:
                        valid_hk = [
                            k for k in hover_keys if k in adata.obs.columns
                        ]
                        if valid_hk:
                            cd = np.column_stack(
                                [
                                    adata.obs[k].values[mask].astype(str)
                                    for k in valid_hk
                                ]
                            )
                            extra = "<br>".join(
                                f"{k}: %{{customdata[{i}]}}"
                                for i, k in enumerate(valid_hk)
                            )
                            hover_kw["customdata"] = cd
                            hover_kw["hovertemplate"] = (
                                f"<b>{cat}</b><br>{extra}<extra></extra>"
                            )

                    fig.add_trace(
                        go.Scatter3d(
                            x=x[mask],
                            y=y[mask],
                            z=z[mask],
                            mode="markers",
                            marker=dict(
                                size=size,
                                color=cat_color,
                                opacity=alpha,
                                line=dict(width=0),
                            ),
                            name=str(cat),
                            legendgroup=f"{col_key}_{cat}",
                            showlegend=True,
                            **hover_kw,
                        ),
                        row=row,
                        col=col,
                    )

        else:
            # ── Continuous: numeric obs column or gene expression ─────────
            values = _get_obs_or_var_values(adata, col_key, layer)
            valid_mask = ~np.isnan(values)

            # NaN cells plotted with na_color in the background
            if not valid_mask.all():
                nan_mask = ~valid_mask
                fig.add_trace(
                    go.Scatter3d(
                        x=x[nan_mask],
                        y=y[nan_mask],
                        z=z[nan_mask],
                        mode="markers",
                        marker=dict(
                            size=size,
                            color=na_color,
                            opacity=alpha * 0.5,
                            line=dict(width=0),
                        ),
                        name="NA",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            # Resolve colour bounds on valid values only
            valid_values = values[valid_mask]
            _vmin = _resolve_vbound(valid_values, vmin)
            _vmax = _resolve_vbound(valid_values, vmax)

            _x = x[valid_mask]
            _y = y[valid_mask]
            _z = z[valid_mask]
            _v = valid_values.copy()

            if sort_order:
                order = np.argsort(_v)
                _x, _y, _z, _v = _x[order], _y[order], _z[order], _v[order]

            hover_kw = {}
            if hover_keys:
                idx_valid = np.where(valid_mask)[0]
                if sort_order:
                    idx_valid = idx_valid[order]
                valid_hk = [k for k in hover_keys if k in adata.obs.columns]
                if valid_hk:
                    cd = np.column_stack(
                        [
                            adata.obs[k].values[idx_valid].astype(str)
                            for k in valid_hk
                        ]
                    )
                    extra = "<br>".join(
                        f"{k}: %{{customdata[{i}]}}"
                        for i, k in enumerate(valid_hk)
                    )
                    hover_kw["customdata"] = cd
                    hover_kw["hovertemplate"] = (
                        f"<b>{col_key}</b>: %{{marker.color:.3g}}"
                        f"<br>{extra}<extra></extra>"
                    )

            fig.add_trace(
                go.Scatter3d(
                    x=_x,
                    y=_y,
                    z=_z,
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=_v,
                        colorscale=colorscale,
                        cmin=_vmin,
                        cmax=_vmax,
                        opacity=alpha,
                        colorbar=dict(
                            title=dict(
                                text=colorbar_title or col_key,
                                side="right",
                            ),
                            thickness=16,
                            len=0.6,
                            x=_cbar_x,
                        ),
                        line=dict(width=0),
                    ),
                    name=col_key,
                    showlegend=False,
                    **hover_kw,
                ),
                row=row,
                col=col,
            )

    # ── 5. Scene layout: visible axes, white background ───────────────────
    _axis_style = dict(
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
        zeroline=False,
        showticklabels=True,
        tickfont=dict(size=10),
        showspikes=False,
        showline=True,
        linecolor="black",
        linewidth=1,
        backgroundcolor="white",
    )
    _scene_layout = dict(
        bgcolor="white",
        xaxis=dict(title="UMAP 1", **_axis_style),
        yaxis=dict(title="UMAP 2", **_axis_style),
        zaxis=dict(title="UMAP 3", **_axis_style),
    )

    scene_updates: dict[str, Any] = {}
    for idx in range(n_panels):
        scene_key = "scene" if idx == 0 else f"scene{idx + 1}"
        scene_updates[scene_key] = _scene_layout

    fig.update_layout(
        **scene_updates,
        paper_bgcolor="white",
        width=width * _ncols,
        height=height * _nrows,
        legend=dict(
            itemsizing="constant",
            tracegroupgap=4,
        ),
    )

    return fig



def annotate_umap_clusters(
    adata: AnnData,
    cluster_key: str,
    groups: Optional[Sequence[str]] = None,
    labels: Optional[dict[str, str]] = None,
    basis: str = "X_umap",
    ax: Optional[plt.Axes] = None,
    text_offset: Tuple[float, float] = (1.5, 1.5),
    fontsize: int = 10,
    arrowprops: Optional[dict] = None,
    **kwargs,
) -> plt.Axes:
    """Annotate UMAP cluster centroids with arrows and text labels.

    Computes the 2-D centroid of each cluster (or a subset of clusters)
    and draws an annotated arrow from an offset text label to the centroid.
    Designed to be called *after* ``sc.pl.umap`` so the annotation is
    overlaid on the existing figure.

    Args:
        adata: Annotated data matrix with ``basis`` in ``adata.obsm``.
        cluster_key: Column in ``adata.obs`` containing cluster labels.
        groups: Subset of cluster labels to annotate. Defaults to all
            clusters found in ``adata.obs[cluster_key]``.
        labels: Optional mapping of cluster label → custom annotation text.
            For example ``{"0": "Monocytes", "3": "NK cells"}``.
            Clusters not present in this dict fall back to the raw cluster
            label as the annotation text.
        basis: Key in ``adata.obsm`` for the 2-D embedding coordinates.
            Defaults to ``"X_umap"``.
        ax: Matplotlib axes to draw on. If ``None``, uses ``plt.gca()``.
        text_offset: ``(dx, dy)`` offset (in embedding units) from the
            centroid to the text anchor. Defaults to ``(1.5, 1.5)``.
        fontsize: Font size for annotation labels. Defaults to ``10``.
        arrowprops: Dict passed to ``ax.annotate`` ``arrowprops``.
            Defaults to a curved arrow with a dark grey colour.
        **kwargs: Additional keyword arguments forwarded to
            ``ax.annotate`` (e.g. ``color``, ``fontweight``).

    Returns:
        The ``Axes`` object with annotations added.

    Raises:
        ValueError: If ``cluster_key`` is not found in ``adata.obs``.
        KeyError: If ``basis`` is not found in ``adata.obsm``.

    Example:
        >>> sc.pl.umap(adata, color="leiden", show=False)
        >>> annotate_umap_clusters(
        ...     adata,
        ...     cluster_key="leiden",
        ...     groups=["0", "3"],
        ...     labels={"0": "Monocytes", "3": "NK cells"},
        ...     text_offset=(2.0, 2.0),
        ... )
        >>> plt.show()
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"Key '{cluster_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if basis not in adata.obsm:
        raise KeyError(
            f"Embedding '{basis}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    coords = adata.obsm[basis][:, :2]
    cluster_labels = adata.obs[cluster_key].astype(str)

    if groups is None:
        groups = cluster_labels.unique().tolist()

    if labels is None:
        labels = {}

    if ax is None:
        ax = plt.gca()

    _arrowprops = dict(
        arrowstyle="->",
        color="#333333",
        connectionstyle="arc3,rad=0.2",
        lw=1.2,
    )
    if arrowprops is not None:
        _arrowprops.update(arrowprops)

    dx, dy = text_offset

    for group in groups:
        mask = cluster_labels == str(group)
        if not mask.any():
            continue
        cx, cy = coords[mask].mean(axis=0)
        annotation_text = labels.get(str(group), str(group))
        ax.annotate(
            annotation_text,
            xy=(cx, cy),
            xytext=(cx + dx, cy + dy),
            fontsize=fontsize,
            arrowprops=_arrowprops,
            ha="center",
            va="center",
            **kwargs,
        )

    return ax

