from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
from anndata import AnnData
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Module-level constants for dot-size scaling
# ---------------------------------------------------------------------------

_LARGEST_DOT: float = 300.0  # scatter ``s`` value for a fully expressed group
_SIZE_EXPONENT: float = 1.5  # power applied to the normalised fraction


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fraction_to_size(fraction: float, dot_min: float, dot_max: float) -> float:
    """Map a fraction value to a scatter marker area using a power transform.

    Uses the same ``_SIZE_EXPONENT`` and ``_LARGEST_DOT`` constants as
    :func:`_plot_size_legend`, ensuring that the dots drawn in the scatter
    plot always match those shown in the size legend.

    Args:
        fraction: Fraction of cells expressing the gene (e.g. ``0.42``).
        dot_min: Lower end of the displayed fraction range.
        dot_max: Upper end of the displayed fraction range.

    Returns:
        Marker area ``s`` value for :func:`matplotlib.axes.Axes.scatter`.
    """
    if dot_max <= dot_min:
        return _LARGEST_DOT
    normalised = np.clip((fraction - dot_min) / (dot_max - dot_min), 0.0, 1.0)
    return float(normalised**_SIZE_EXPONENT * _LARGEST_DOT)


def _size_legend_ticks(dot_min: float, dot_max: float) -> np.ndarray:
    """Choose evenly spaced, human-readable tick fractions for the size legend.

    Tries progressively coarser step sizes until between 3 and 6 ticks are
    found.  The last tick is always set to *dot_max* so the legend faithfully
    represents the maximum dot size.

    Args:
        dot_min: Lower end of the displayed fraction range.
        dot_max: Upper end of the displayed fraction range.

    Returns:
        1-D array of fraction values in ascending order.
    """
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5):
        ticks = np.arange(0.0, dot_max + step / 2.0, step)
        # Exclude 0: a fraction of 0 means no dot is drawn at all, so it
        # should not appear in the size legend.
        ticks = ticks[(ticks > 1e-9) & (ticks <= dot_max + 1e-9)]
        if 3 <= len(ticks) <= 6:
            ticks[-1] = dot_max  # guarantee exact upper bound
            return ticks
    # Fallback: evenly spaced, skipping 0
    ticks = np.linspace(dot_min, dot_max, 6)[1:]
    return ticks if len(ticks) > 0 else np.array([dot_max])


def _format_pct_labels(pct_values: list) -> list:
    """Format percentage values as strings with the fewest decimal places needed
    to make every label unique.

    Tries integer labels first (e.g. ``"40"``), then one decimal place
    (``"0.4"``), then two (``"0.04"``), falling back to full ``repr`` only if
    even two decimals produce duplicates.

    Args:
        pct_values: Percentage values (0–100 scale) to format.

    Returns:
        List of strings the same length as *pct_values* with no duplicates.
    """
    for decimals in (0, 1, 2):
        if decimals == 0:
            labels = [str(int(round(v))) for v in pct_values]
        else:
            labels = [f"{v:.{decimals}f}" for v in pct_values]
        if len(set(labels)) == len(labels):
            return labels
    return [repr(v) for v in pct_values]


def _make_size_legend_handles(
    dot_min: float,
    dot_max: float,
) -> tuple:
    """Build matplotlib legend handles for the dot-size legend.

    Creates :class:`matplotlib.lines.Line2D` handles with circular markers
    whose sizes approximate those in the scatter plot, suitable for use with
    :meth:`matplotlib.axes.Axes.legend`.

    Args:
        dot_min: Lower end of the displayed fraction range.
        dot_max: Upper end of the displayed fraction range.

    Returns:
        A ``(handles, labels)`` tuple where *handles* is a list of
        :class:`~matplotlib.lines.Line2D` artists and *labels* is a list
        of percentage strings (e.g. ``["0", "20", "40", "60"]``).
    """
    ticks = _size_legend_ticks(dot_min, dot_max)
    handles = []
    for t in ticks:
        s = _fraction_to_size(t, dot_min, dot_max)
        # Line2D markersize is diameter in points; scatter ``s`` is area in
        # points².  Approximate visual match: diameter ≈ sqrt(area) * factor.
        ms = max(np.sqrt(s) * 0.55, 2.0)
        h = matplotlib.lines.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color="darkgray",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=ms,
        )
        handles.append(h)
    labels = _format_pct_labels([t * 100.0 for t in ticks])
    return handles, labels


def _resolve_vmin_vmax(
    values: pd.Series,
    v: Optional[Union[str, float]],
) -> Optional[float]:
    """Resolve a *vmin* / *vmax* argument to a concrete float.

    Args:
        values: Numeric series used when *v* is a percentile string.
        v: Raw value: ``None``, a plain ``float``, or a percentile string
            such as ``"p95"``.

    Returns:
        Resolved float, or ``None`` when *v* is ``None``.

    Raises:
        ValueError: If *v* is a string that does not start with ``"p"``.
    """
    if v is None:
        return None
    if isinstance(v, str):
        if v.startswith("p"):
            return float(np.quantile(values.dropna(), float(v[1:]) / 100))
        raise ValueError(
            f"Invalid vmin/vmax string '{v}'. "
            "Use a percentile string like 'p95' or a plain float."
        )
    return float(v)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def dotplot_expression_two_categories(
    adata: AnnData,
    feature: str,
    category_x: str,
    category_y: str,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    cmap: str = "Reds",
    expression_cutoff: float = 0.0,
    dot_max: Optional[float] = None,
    vmin: Optional[Union[str, float]] = None,
    vmax: Optional[Union[str, float]] = None,
    size_title: str = "% cells expressing",
    color_title: str = "Mean expression",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    use_zscores: bool = False,
    return_dataframe: bool = False,
) -> Union[Figure, Tuple[Figure, pd.DataFrame]]:
    """Dotplot with gene expression split by two categorical variables.

    An alternative to ``sc.pl.dotplot`` that visualises a *single* gene
    across two independent ``adata.obs`` categorical columns.  Each dot's
    **colour** encodes the mean expression within the
    ``(category_x, category_y)`` group, and its **size** encodes the fraction
    of cells in the group expressing the gene above *expression_cutoff*.

    When *ax* is ``None`` the figure size is computed automatically from the
    number of categories on each axis.  Pass *figsize* to override this, or
    supply an existing *ax* to embed the plot in a larger figure (in which
    case auto-sizing is skipped and the colourbar / size legend are added as
    inset axes).

    Args:
        adata: Annotated data matrix.
        feature: Gene name or ``adata.obs`` column to visualise.  Resolved
            with obs-first priority: if *feature* is present in
            ``adata.obs.columns`` it is used directly (and *layer* /
            *gene_symbols* are ignored).  Otherwise it is matched against
            ``adata.var_names`` or ``adata.var[gene_symbols]`` when
            *gene_symbols* is provided.
        category_x: Column in ``adata.obs`` to use on the x-axis.
        category_y: Column in ``adata.obs`` to use on the y-axis.
        layer: Expression layer to use.  ``None`` uses ``adata.X``.
            Defaults to ``None``.
        gene_symbols: Column in ``adata.var`` that stores alternative gene
            identifiers.  When set, *gene* is matched against that column
            instead of ``adata.var_names``.  Defaults to ``None``.
        cmap: Matplotlib colormap name for the expression colour scale.
            Defaults to ``"Reds"``.
        expression_cutoff: A cell is considered to express the gene only
            when its value is strictly greater than this threshold.  Used to
            compute the dot-size fraction.  Defaults to ``0.0``.
        dot_max: Maximum fraction value that maps to the largest dot.
            Fractions above *dot_max* are clipped.  When ``None`` the value
            is the observed maximum rounded up to the nearest 10 %.
            Defaults to ``None``.
        vmin: Lower colour-scale limit.  Accepts a plain ``float`` or a
            percentile string (e.g. ``"p5"``).  ``None`` uses the data
            minimum.  Defaults to ``None``.
        vmax: Upper colour-scale limit.  Accepts a plain ``float`` or a
            percentile string (e.g. ``"p95"``).  ``None`` uses the data
            maximum.  Defaults to ``None``.
        size_title: Title text shown above the dot-size legend.
            Defaults to ``"% cells expressing"``.
        color_title: Label on the colourbar.
            Defaults to ``"Mean expression"``.
        figsize: Figure size as ``(width, height)`` in inches.  When
            ``None`` and *ax* is ``None``, the size is derived from the
            number of categories.  Defaults to ``None``.
        ax: Existing :class:`matplotlib.axes.Axes` to plot into.  When
            supplied, *figsize* has no effect and the colourbar / size
            legend are attached as inset axes.  Defaults to ``None``.
        use_zscores: When ``True``, z-score the per-group mean expression
            values before plotting.  Defaults to ``False``.
        return_dataframe: When ``True``, also return the aggregated
            ``pd.DataFrame`` with columns *category_x*, *category_y*,
            ``"mean"``, and ``"size"``.  Defaults to ``False``.

    Returns:
        The matplotlib ``Figure``, or a ``(Figure, DataFrame)`` tuple when
        *return_dataframe* is ``True``.

    Raises:
        ValueError: If *category_x* or *category_y* is not found in
            ``adata.obs.columns``.
        KeyError: If *feature* is not found in ``adata.obs.columns``,
            ``adata.var_names``, or ``adata.var[gene_symbols]`` (when
            supplied).
        ValueError: If *vmin* or *vmax* is a string not starting with
            ``"p"``.

    Example:
        >>> fig = dotplot_expression_two_categories(
        ...     adata,
        ...     feature="CD3E",
        ...     category_x="leiden",
        ...     category_y="condition",
        ...     cmap="Blues",
        ...     expression_cutoff=0.5,
        ...     vmax="p95",
        ... )
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if category_x not in adata.obs.columns:
        raise ValueError(
            f"category_x='{category_x}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if category_y not in adata.obs.columns:
        raise ValueError(
            f"category_y='{category_y}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    # Resolve feature: obs column takes priority over var (gene)
    _is_obs_col = feature in adata.obs.columns
    if not _is_obs_col:
        if gene_symbols is not None:
            if feature not in adata.var[gene_symbols].tolist():
                raise KeyError(
                    f"Feature '{feature}' not found in adata.obs.columns or "
                    f"adata.var['{gene_symbols}']. "
                    "Choose a valid obs column or gene."
                )
        else:
            if feature not in adata.var_names.tolist():
                raise KeyError(
                    f"Feature '{feature}' not found in adata.obs.columns or "
                    "adata.var_names. Choose a valid obs column or gene."
                )
    # layer and gene_symbols only apply to var (gene) features
    _layer = None if _is_obs_col else layer
    _gene_symbols = None if _is_obs_col else gene_symbols

    # ------------------------------------------------------------------
    # Build per-group aggregates
    # ------------------------------------------------------------------
    obs_tidy = sc.get.obs_df(
        adata,
        keys=[feature, category_x, category_y],
        use_raw=False,
        layer=_layer,
        gene_symbols=_gene_symbols,
    )

    # Ensure categorical dtype so that category order is stable and
    # unused categories from a previously-subsetted AnnData are dropped.
    for col in (category_x, category_y):
        if obs_tidy[col].dtype.name != "category":
            obs_tidy[col] = obs_tidy[col].astype("category")
        obs_tidy[col] = obs_tidy[col].cat.remove_unused_categories()

    cats_x: list = obs_tidy[category_x].cat.categories.tolist()
    cats_y: list = obs_tidy[category_y].cat.categories.tolist()
    n_x, n_y = len(cats_x), len(cats_y)

    obs_tidy["_expressed"] = (obs_tidy[feature] > expression_cutoff).astype(float)
    grouped = obs_tidy.groupby([category_x, category_y], observed=True)
    mean_series = grouped[feature].mean()
    size_series = grouped["_expressed"].mean()

    gene_df = (
        pd.DataFrame({"mean": mean_series, "size": size_series})
        .reset_index()
    )

    if use_zscores:
        gene_df["mean"] = scipy.stats.zscore(gene_df["mean"].values)

    # ------------------------------------------------------------------
    # Resolve dot_max, vmin, vmax
    # ------------------------------------------------------------------
    dot_min = 0.0
    if dot_max is None:
        raw_max = float(gene_df["size"].max())
        # Round up to the nearest 10 % so the legend shows a clean upper value
        dot_max = min(1.0, float(np.ceil(raw_max * 10.0) / 10.0))
        # Never clip actual data points
        dot_max = max(dot_max, raw_max)

    _vmin = _resolve_vmin_vmax(gene_df["mean"], vmin)
    _vmax = _resolve_vmin_vmax(gene_df["mean"], vmax)

    # ------------------------------------------------------------------
    # Compute scatter sizes (must use same formula as _plot_size_legend)
    # ------------------------------------------------------------------
    gene_df["_scatter_size"] = gene_df["size"].apply(
        lambda s: _fraction_to_size(float(s), dot_min, dot_max)
    )

    # Integer x/y positions for stable, predictable scatter placement
    x_to_pos = {cat: i for i, cat in enumerate(cats_x)}
    y_to_pos = {cat: i for i, cat in enumerate(cats_y)}
    gene_df["_x"] = gene_df[category_x].map(x_to_pos)
    gene_df["_y"] = gene_df[category_y].map(y_to_pos)

    # ------------------------------------------------------------------
    # Create figure / axes layout
    # Sizing constants (inches per category tick + fixed padding)
    # ------------------------------------------------------------------
    _CELL: float = 0.5    # inches per category tick
    _CBAR_W: float = 0.275  # colourbar axes width
    _GAP_W: float = 0.8   # gap between colorbar and legend box
    _LEGEND_W: float = 2.0  # width reserved for the size-legend box

    if ax is None:
        if figsize is None:
            main_w = max(n_x * _CELL, 2.0)
            main_h = max(n_y * _CELL, 2.0)
            figsize = (
                main_w + _CBAR_W + _GAP_W + _LEGEND_W + 1.5,  # + left/right padding
                main_h + 1.0,                                    # + top/bottom padding
            )
        else:
            main_w = max(figsize[0] - _CBAR_W - _GAP_W - _LEGEND_W - 1.5, 1.0)
            main_h = max(figsize[1] - 1.0, 1.0)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            1, 4,
            width_ratios=[main_w, _CBAR_W, _GAP_W, _LEGEND_W],
            left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.05,
        )
        scatter_ax = fig.add_subplot(gs[0, 0])
        cbar_ax: Optional[plt.Axes] = fig.add_subplot(gs[0, 1])
        _gap_ax = fig.add_subplot(gs[0, 2])
        _gap_ax.set_axis_off()
        _legend_space_ax: Optional[plt.Axes] = fig.add_subplot(gs[0, 3])
        _legend_space_ax.set_axis_off()
        _use_dedicated_cax = True
    else:
        scatter_ax = ax
        fig = ax.get_figure()
        cbar_ax = None
        _legend_space_ax = None
        _use_dedicated_cax = False

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    norm = matplotlib.colors.Normalize(vmin=_vmin, vmax=_vmax)
    scatter = scatter_ax.scatter(
        x=gene_df["_x"],
        y=gene_df["_y"],
        c=gene_df["mean"],
        s=gene_df["_scatter_size"],
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=1.0,
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Colourbar
    # ------------------------------------------------------------------
    if _use_dedicated_cax:
        cbar = fig.colorbar(scatter, cax=cbar_ax)
    else:
        cbar = fig.colorbar(scatter, ax=scatter_ax, pad=0.02, fraction=0.046)
    cbar.set_label(color_title, fontsize="small")
    cbar.ax.tick_params(labelsize="small")
    cbar.ax.grid(False)

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    scatter_ax.set_title(feature)
    scatter_ax.set_xlabel(category_x)
    scatter_ax.set_ylabel(category_y)
    scatter_ax.set_xticks(range(n_x))
    scatter_ax.set_xticklabels(cats_x, rotation=90)
    scatter_ax.set_yticks(range(n_y))
    scatter_ax.set_yticklabels(cats_y)
    scatter_ax.set_xlim(-0.5, n_x - 0.5)
    scatter_ax.set_ylim(n_y - 0.5, -0.5)
    scatter_ax.grid(False)

    # ------------------------------------------------------------------
    # Size legend — boxed matplotlib Legend, top-right outside the plot
    # ------------------------------------------------------------------
    size_handles, size_labels = _make_size_legend_handles(dot_min, dot_max)
    _legend_kwargs: dict = dict(
        title=size_title,
        title_fontsize="small",
        fontsize="small",
        loc="upper left",
        frameon=True,
        borderpad=0.7,
        labelspacing=0.8,
        handlelength=1.5,
        handleheight=2.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    if _legend_space_ax is not None:
        # Anchor the legend box to the top-left of the dedicated legend-space
        # column (to the right of the colourbar) using its own transform so
        # the position is independent of the scatter axes width.
        scatter_ax.legend(
            size_handles, size_labels,
            bbox_to_anchor=(0.0, 1.0),
            bbox_transform=_legend_space_ax.transAxes,
            **_legend_kwargs,
        )
    else:
        scatter_ax.legend(
            size_handles, size_labels,
            bbox_to_anchor=(1.0, 1.0),
            **_legend_kwargs,
        )

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------
    if return_dataframe:
        out_df = gene_df[[category_x, category_y, "mean", "size"]].copy()
        return fig, out_df
    return fig


def dotplot_expression_two_categories_multiplot(
    adata: AnnData,
    features: list,
    category_x: str,
    category_y: str,
    ncols: int = 2,
    shared_colorscale: bool = True,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    cmap: str = "Reds",
    expression_cutoff: float = 0.0,
    dot_max: Optional[float] = None,
    vmin: Optional[Union[str, float]] = None,
    vmax: Optional[Union[str, float]] = None,
    size_title: str = "% cells expressing",
    color_title: str = "Mean expression",
    figsize: Optional[Tuple[float, float]] = None,
    hspace: float = 0.2,
    wspace: float = 1.2,
    border_ticks_only: bool = True,
    use_zscores: bool = False,
) -> Figure:
    """Grid of dotplots across multiple features.

    Creates a multi-panel figure with one :func:`dotplot_expression_two_categories`
    panel per feature, arranged in a ``nrows × ncols`` grid.  Dot sizes are
    always on a shared scale (same *dot_max* for all subplots).  A single
    dot-size legend is placed to the right of the last panel in the first row.

    Args:
        adata: Annotated data matrix.
        features: Ordered list of gene names or ``adata.obs`` column names to
            plot.  Each entry resolves with obs-first priority: if present in
            ``adata.obs.columns`` it is used directly (and *layer* /
            *gene_symbols* are ignored for that entry).  Otherwise it is
            matched against ``adata.var_names`` or ``adata.var[gene_symbols]``.
        category_x: Column in ``adata.obs`` to use on the x-axis of every
            subplot.
        category_y: Column in ``adata.obs`` to use on the y-axis of every
            subplot.
        ncols: Number of columns in the grid.  Defaults to ``2``.
        shared_colorscale: When ``True``, a single *vmin* / *vmax* is computed
            across all features so colours are comparable between subplots.
            When ``False``, each subplot uses its own colour scale.
            Defaults to ``True``.
        layer: Expression layer to use.  ``None`` uses ``adata.X``.
            Defaults to ``None``.
        gene_symbols: Column in ``adata.var`` that stores alternative gene
            identifiers.  Defaults to ``None``.
        cmap: Matplotlib colormap name.  Defaults to ``"Reds"``.
        expression_cutoff: Cells are considered expressing only when their
            value exceeds this threshold.  Defaults to ``0.0``.
        dot_max: Maximum fraction that maps to the largest dot.  Shared
            across all subplots.  ``None`` uses the global observed maximum
            rounded up to the nearest 10 %.  Defaults to ``None``.
        vmin: Lower colour-scale limit (plain ``float`` or percentile string
            such as ``"p5"``).  Applied globally when *shared_colorscale* is
            ``True``, or per-subplot otherwise.  Defaults to ``None``.
        vmax: Upper colour-scale limit.  Defaults to ``None``.
        size_title: Title text of the dot-size legend.
            Defaults to ``"% cells expressing"``.
        color_title: Colorbar label applied to every subplot.
            Defaults to ``"Mean expression"``.
        figsize: Size of a **single** dotplot panel ``(width, height)`` in
            inches.  The overall figure size is computed automatically from
            this value, the number of rows/columns, and the fixed widths of
            the colourbar and legend.  When ``None``, the panel size is
            derived from the number of categories on each axis.
            Defaults to ``None``.
        hspace: Vertical space between subplot rows, as a fraction of the
            average axes height.  Defaults to ``0.2``.
        wspace: Horizontal gap in inches between adjacent plot groups
            (i.e. between the colourbar of one group and the scatter axes of
            the next).  The scatter axes and its colourbar are always flush
            with each other regardless of this setting.  Increase if the
            colourbar label of one group overlaps the left spine of the next.
            Defaults to ``1.2``.
        border_ticks_only: When ``True``, x-axis tick labels and the x-axis
            label are shown only on the bottom row of subplots, and y-axis
            tick labels and the y-axis label are shown only on the leftmost
            column.  This reduces clutter in multi-row, multi-column grids.
            Set to ``False`` to display ticks and labels on every subplot.
            Defaults to ``True``.
        use_zscores: When ``True``, z-score per-group means before plotting.
            Applied independently per feature.  Defaults to ``False``.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If *category_x* or *category_y* is not found in
            ``adata.obs.columns``.
        ValueError: If *features* is empty.
        KeyError: If any entry in *features* cannot be resolved.
        ValueError: If *vmin* or *vmax* is a string not starting with ``"p"``.

    Example:
        >>> fig = dotplot_expression_two_categories_multiplot(
        ...     adata,
        ...     features=["CD3E", "CD8A", "CD19", "MS4A1"],
        ...     category_x="leiden",
        ...     category_y="condition",
        ...     ncols=2,
        ...     shared_colorscale=True,
        ... )
    """
    # ------------------------------------------------------------------
    # Validate global inputs
    # ------------------------------------------------------------------
    if category_x not in adata.obs.columns:
        raise ValueError(
            f"category_x='{category_x}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if category_y not in adata.obs.columns:
        raise ValueError(
            f"category_y='{category_y}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if len(features) == 0:
        raise ValueError("features must not be empty.")

    # ------------------------------------------------------------------
    # Resolve, validate, and aggregate each feature
    # ------------------------------------------------------------------
    all_dfs: list = []
    cats_x: Optional[list] = None
    cats_y: Optional[list] = None

    for feat in features:
        _is_obs_col = feat in adata.obs.columns
        if not _is_obs_col:
            if gene_symbols is not None:
                if feat not in adata.var[gene_symbols].tolist():
                    raise KeyError(
                        f"Feature '{feat}' not found in adata.obs.columns or "
                        f"adata.var['{gene_symbols}']. "
                        "Choose a valid obs column or gene."
                    )
            else:
                if feat not in adata.var_names.tolist():
                    raise KeyError(
                        f"Feature '{feat}' not found in adata.obs.columns or "
                        "adata.var_names. Choose a valid obs column or gene."
                    )
        _layer = None if _is_obs_col else layer
        _gene_symbols = None if _is_obs_col else gene_symbols

        obs_tidy = sc.get.obs_df(
            adata,
            keys=[feat, category_x, category_y],
            use_raw=False,
            layer=_layer,
            gene_symbols=_gene_symbols,
        )
        for col in (category_x, category_y):
            if obs_tidy[col].dtype.name != "category":
                obs_tidy[col] = obs_tidy[col].astype("category")
            obs_tidy[col] = obs_tidy[col].cat.remove_unused_categories()

        if cats_x is None:
            cats_x = obs_tidy[category_x].cat.categories.tolist()
            cats_y = obs_tidy[category_y].cat.categories.tolist()

        obs_tidy["_expressed"] = (obs_tidy[feat] > expression_cutoff).astype(float)
        grouped = obs_tidy.groupby([category_x, category_y], observed=True)
        gene_df = (
            pd.DataFrame({
                "mean": grouped[feat].mean(),
                "size": grouped["_expressed"].mean(),
            })
            .reset_index()
        )
        if use_zscores:
            gene_df["mean"] = scipy.stats.zscore(gene_df["mean"].values)
        all_dfs.append(gene_df)

    n_x = len(cats_x)
    n_y = len(cats_y)

    # ------------------------------------------------------------------
    # Shared dot_max (always shared so dot sizes are comparable)
    # ------------------------------------------------------------------
    dot_min = 0.0
    if dot_max is None:
        raw_max = max(float(df["size"].max()) for df in all_dfs)
        dot_max = min(1.0, float(np.ceil(raw_max * 10.0) / 10.0))
        dot_max = max(dot_max, raw_max)

    # ------------------------------------------------------------------
    # Shared or per-feature vmin / vmax
    # ------------------------------------------------------------------
    if shared_colorscale:
        _all_means = pd.concat([df["mean"] for df in all_dfs], ignore_index=True)
        _global_vmin = _resolve_vmin_vmax(_all_means, vmin)
        _global_vmax = _resolve_vmin_vmax(_all_means, vmax)

    # ------------------------------------------------------------------
    # Pre-compute scatter sizes and integer axis positions for every df
    # ------------------------------------------------------------------
    x_to_pos = {cat: i for i, cat in enumerate(cats_x)}
    y_to_pos = {cat: i for i, cat in enumerate(cats_y)}
    for gene_df in all_dfs:
        gene_df["_scatter_size"] = gene_df["size"].apply(
            lambda s: _fraction_to_size(float(s), dot_min, dot_max)
        )
        gene_df["_x"] = gene_df[category_x].map(x_to_pos)
        gene_df["_y"] = gene_df[category_y].map(y_to_pos)

    # ------------------------------------------------------------------
    # Figure layout
    #
    # Column structure (repeated ncols times):
    #   [scatter_w, cbar_w, between_gap*]   (* omitted after last plot col)
    # Then appended at the right edge:
    #   [legend_gap_w, legend_w]
    #
    # Scatter column for plot i : 3*i
    # Cbar    column for plot i : 3*i + 1
    # Legend-gap column         : 3*ncols - 1
    # Legend column             : 3*ncols
    # ------------------------------------------------------------------
    nrows = int(np.ceil(len(features) / ncols))

    _CELL: float = 0.6           # inches per category tick for auto-sizing
    _CBAR_W: float = 0.275
    _BETWEEN_GAP: float = wspace  # gap between adjacent scatter+cbar groups
    _LEGEND_GAP_W: float = 1.5  # gap between last cbar and size legend
    _LEGEND_W: float = 2.0

    if figsize is not None:
        # figsize is the per-panel scatter size; derive total from it
        main_w, main_h = float(figsize[0]), float(figsize[1])
    else:
        main_w = max(n_x * _CELL, 3.0)
        main_h = max(n_y * _CELL, 3.0)

    total_w = (
        ncols * (main_w + _CBAR_W)
        + (ncols - 1) * _BETWEEN_GAP
        + _LEGEND_GAP_W + _LEGEND_W
        + 1.5
    )
    total_h = nrows * main_h + (nrows - 1) * 0.5 + 1.2

    width_ratios: list = []
    for i in range(ncols):
        width_ratios.append(main_w)
        width_ratios.append(_CBAR_W)
        if i < ncols - 1:
            width_ratios.append(_BETWEEN_GAP)
    width_ratios.append(_LEGEND_GAP_W)
    width_ratios.append(_LEGEND_W)

    fig = plt.figure(figsize=(total_w, total_h))
    gs = fig.add_gridspec(
        nrows,
        len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=[main_h] * nrows,
        left=0.1, right=0.95,
        top=0.92, bottom=0.12,
        wspace=0,
        hspace=hspace,
    )

    # Invisible axes whose transform anchors the size legend
    _legend_space_ax = fig.add_subplot(gs[0, 3 * ncols])
    _legend_space_ax.set_axis_off()

    # ------------------------------------------------------------------
    # Draw each subplot
    # ------------------------------------------------------------------
    first_scatter_ax = None

    for idx, (feat, gene_df) in enumerate(zip(features, all_dfs)):
        row = idx // ncols
        col = idx % ncols

        scatter_ax = fig.add_subplot(gs[row, 3 * col])
        cbar_ax = fig.add_subplot(gs[row, 3 * col + 1])

        if first_scatter_ax is None:
            first_scatter_ax = scatter_ax

        if shared_colorscale:
            _vmin, _vmax = _global_vmin, _global_vmax
        else:
            _vmin = _resolve_vmin_vmax(gene_df["mean"], vmin)
            _vmax = _resolve_vmin_vmax(gene_df["mean"], vmax)

        norm = matplotlib.colors.Normalize(vmin=_vmin, vmax=_vmax)
        scatter = scatter_ax.scatter(
            x=gene_df["_x"],
            y=gene_df["_y"],
            c=gene_df["mean"],
            s=gene_df["_scatter_size"],
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )

        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label(color_title, fontsize="small")
        cbar.ax.tick_params(labelsize="small")
        cbar.ax.grid(False)

        scatter_ax.set_title(feat)
        scatter_ax.set_xticks(range(n_x))
        scatter_ax.set_yticks(range(n_y))

        # x-axis: labels and xlabel only on the bottom row
        _is_bottom_row = (row == nrows - 1)
        if border_ticks_only and not _is_bottom_row:
            scatter_ax.set_xticklabels([])
            scatter_ax.set_xlabel("")
        else:
            scatter_ax.set_xticklabels(cats_x, rotation=90)
            scatter_ax.set_xlabel(category_x)

        # y-axis: labels and ylabel only on the first column
        if border_ticks_only and col > 0:
            scatter_ax.set_yticklabels([])
            scatter_ax.set_ylabel("")
        else:
            scatter_ax.set_yticklabels(cats_y)
            scatter_ax.set_ylabel(category_y)

        scatter_ax.set_xlim(-0.5, n_x - 0.5)
        scatter_ax.set_ylim(n_y - 0.5, -0.5)
        scatter_ax.grid(False)

    # ------------------------------------------------------------------
    # Size legend — once, anchored to the legend-space column in row 0
    # ------------------------------------------------------------------
    size_handles, size_labels = _make_size_legend_handles(dot_min, dot_max)
    first_scatter_ax.legend(
        size_handles,
        size_labels,
        title=size_title,
        title_fontsize="small",
        fontsize="small",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=_legend_space_ax.transAxes,
        frameon=True,
        borderpad=0.7,
        labelspacing=0.8,
        handlelength=1.5,
        handleheight=2.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    return fig
