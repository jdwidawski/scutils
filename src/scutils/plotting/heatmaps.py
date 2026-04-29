from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
from anndata import AnnData
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


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


def _pvalue_to_stars(p: float) -> str:
    """Convert a p-value to an asterisk significance string.

    Args:
        p: The p-value.

    Returns:
        ``"ns"`` for p ≥ 0.05, otherwise one to four asterisks.
    """
    if p >= 0.05:
        return "ns"
    if p >= 0.01:
        return "*"
    if p >= 0.001:
        return "**"
    if p >= 0.0001:
        return "***"
    return "****"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def heatmap_expression_two_categories(
    adata: AnnData,
    feature: str,
    category_x: str,
    category_y: str,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    cmap: str = "Reds",
    vmin: Optional[Union[str, float]] = None,
    vmax: Optional[Union[str, float]] = None,
    color_title: str = "Mean expression",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    use_zscores: bool = False,
    return_dataframe: bool = False,
) -> Union[Figure, Tuple[Figure, pd.DataFrame]]:
    """Heatmap with gene expression split by two categorical variables.

    An alternative to :func:`dotplot_expression_two_categories` that
    visualises a *single* feature across two independent ``adata.obs``
    categorical columns using a plain colour heatmap.  Each cell's colour
    encodes the mean expression within the ``(category_x, category_y)``
    group.  Unlike the dotplot variant, no size encoding is used.

    When *ax* is ``None`` the figure size is computed automatically from the
    number of categories on each axis.  Pass *figsize* to override this, or
    supply an existing *ax* to embed the plot in a larger figure (in which
    case auto-sizing is skipped and the colourbar is added as an inset axis).

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
            identifiers.  When set, *feature* is matched against that column
            instead of ``adata.var_names``.  Defaults to ``None``.
        cmap: Matplotlib colormap name for the expression colour scale.
            Defaults to ``"Reds"``.
        vmin: Lower colour-scale limit.  Accepts a plain ``float`` or a
            percentile string (e.g. ``"p5"``).  ``None`` uses the data
            minimum.  Defaults to ``None``.
        vmax: Upper colour-scale limit.  Accepts a plain ``float`` or a
            percentile string (e.g. ``"p95"``).  ``None`` uses the data
            maximum.  Defaults to ``None``.
        color_title: Label on the colourbar.
            Defaults to ``"Mean expression"``.
        figsize: Figure size as ``(width, height)`` in inches.  When
            ``None`` and *ax* is ``None``, the size is derived from the
            number of categories.  Defaults to ``None``.
        ax: Existing :class:`matplotlib.axes.Axes` to plot into.  When
            supplied, *figsize* has no effect and the colourbar is attached
            as an inset axis.  Defaults to ``None``.
        use_zscores: When ``True``, z-score the per-group mean expression
            values before plotting.  Defaults to ``False``.
        return_dataframe: When ``True``, also return the aggregated
            ``pd.DataFrame`` with columns *category_x*, *category_y*, and
            ``"mean"``.  Defaults to ``False``.

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
        >>> fig = heatmap_expression_two_categories(
        ...     adata,
        ...     feature="CD3E",
        ...     category_x="leiden",
        ...     category_y="condition",
        ...     cmap="Blues",
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
    for col in (category_x, category_y):
        if obs_tidy[col].dtype.name != "category":
            obs_tidy[col] = obs_tidy[col].astype("category")
        obs_tidy[col] = obs_tidy[col].cat.remove_unused_categories()

    cats_x: list = obs_tidy[category_x].cat.categories.tolist()
    cats_y: list = obs_tidy[category_y].cat.categories.tolist()
    n_x, n_y = len(cats_x), len(cats_y)

    grouped = obs_tidy.groupby([category_x, category_y], observed=True)
    gene_df = pd.DataFrame({"mean": grouped[feature].mean()}).reset_index()

    if use_zscores:
        gene_df["mean"] = scipy.stats.zscore(gene_df["mean"].values)

    # ------------------------------------------------------------------
    # Resolve vmin / vmax
    # ------------------------------------------------------------------
    _vmin = _resolve_vmin_vmax(gene_df["mean"], vmin)
    _vmax = _resolve_vmin_vmax(gene_df["mean"], vmax)

    # ------------------------------------------------------------------
    # Build integer index maps and 2-D colour matrix
    # rows = cats_y index (top → bottom), cols = cats_x index (left → right)
    # ------------------------------------------------------------------
    x_to_pos = {cat: i for i, cat in enumerate(cats_x)}
    y_to_pos = {cat: i for i, cat in enumerate(cats_y)}

    mat = np.full((n_y, n_x), np.nan)
    mat[
        gene_df[category_y].map(y_to_pos).values,
        gene_df[category_x].map(x_to_pos).values,
    ] = gene_df["mean"].values

    # ------------------------------------------------------------------
    # Figure / axes layout
    # ------------------------------------------------------------------
    _CELL: float = 0.5      # inches per category tick
    _CBAR_W: float = 0.275  # colourbar axes width

    if ax is None:
        if figsize is None:
            main_w = max(n_x * _CELL, 2.0)
            main_h = max(n_y * _CELL, 2.0)
            figsize = (
                main_w + _CBAR_W + 1.5,  # + left/right padding
                main_h + 1.0,            # + top/bottom padding
            )
        else:
            main_w = max(figsize[0] - _CBAR_W - 1.5, 1.0)
            main_h = max(figsize[1] - 1.0, 1.0)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[main_w, _CBAR_W],
            left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.05,
        )
        heatmap_ax = fig.add_subplot(gs[0, 0])
        cbar_ax: Optional[plt.Axes] = fig.add_subplot(gs[0, 1])
        _use_dedicated_cax = True
    else:
        heatmap_ax = ax
        fig = ax.get_figure()
        cbar_ax = None
        _use_dedicated_cax = False

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    norm = matplotlib.colors.Normalize(vmin=_vmin, vmax=_vmax)
    im = heatmap_ax.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )

    # ------------------------------------------------------------------
    # Colourbar
    # ------------------------------------------------------------------
    if _use_dedicated_cax:
        cbar = fig.colorbar(im, cax=cbar_ax)
    else:
        cbar = fig.colorbar(im, ax=heatmap_ax, pad=0.02, fraction=0.046)
    cbar.set_label(color_title, fontsize="small")
    cbar.ax.tick_params(labelsize="small")
    cbar.ax.grid(False)

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    heatmap_ax.set_title(feature)
    heatmap_ax.set_xlabel(category_x)
    heatmap_ax.set_ylabel(category_y)
    heatmap_ax.set_xticks(range(n_x))
    heatmap_ax.set_xticklabels(cats_x, rotation=90)
    heatmap_ax.set_yticks(range(n_y))
    heatmap_ax.set_yticklabels(cats_y)
    heatmap_ax.grid(False)

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------
    if return_dataframe:
        out_df = gene_df[[category_x, category_y, "mean"]].copy()
        return fig, out_df
    return fig


def heatmap_expression_two_categories_multiplot(
    adata: AnnData,
    features: list,
    category_x: str,
    category_y: str,
    ncols: int = 2,
    shared_colorscale: bool = True,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    cmap: str = "Reds",
    vmin: Optional[Union[str, float]] = None,
    vmax: Optional[Union[str, float]] = None,
    color_title: str = "Mean expression",
    figsize: Optional[Tuple[float, float]] = None,
    hspace: float = 0.2,
    wspace: float = 1.2,
    border_ticks_only: bool = True,
    use_zscores: bool = False,
) -> Figure:
    """Grid of heatmaps across multiple features.

    Creates a multi-panel figure with one
    :func:`heatmap_expression_two_categories` panel per feature, arranged in
    a ``nrows × ncols`` grid.  A single colour scale can optionally be shared
    across all subplots for direct comparability.

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
        vmin: Lower colour-scale limit (plain ``float`` or percentile string
            such as ``"p5"``).  Applied globally when *shared_colorscale* is
            ``True``, or per-subplot otherwise.  Defaults to ``None``.
        vmax: Upper colour-scale limit.  Defaults to ``None``.
        color_title: Colorbar label applied to every subplot.
            Defaults to ``"Mean expression"``.
        figsize: Size of a **single** heatmap panel ``(width, height)`` in
            inches.  The overall figure size is computed automatically from
            this value and the number of rows/columns.  When ``None``, the
            panel size is derived from the number of categories on each axis.
            Defaults to ``None``.
        hspace: Vertical space between subplot rows, as a fraction of the
            average axes height.  Defaults to ``0.2``.
        wspace: Horizontal gap in inches between adjacent plot groups
            (i.e. between the colourbar of one group and the heatmap of the
            next).  The heatmap and its colourbar are always flush with each
            other regardless of this setting.  Increase if the colourbar label
            of one group overlaps the left spine of the next.
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
        ValueError: If *vmin* or *vmax* is a string not starting with
            ``"p"``.

    Example:
        >>> fig = heatmap_expression_two_categories_multiplot(
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

        grouped = obs_tidy.groupby([category_x, category_y], observed=True)
        gene_df = pd.DataFrame({"mean": grouped[feat].mean()}).reset_index()

        if use_zscores:
            gene_df["mean"] = scipy.stats.zscore(gene_df["mean"].values)
        all_dfs.append(gene_df)

    n_x = len(cats_x)
    n_y = len(cats_y)

    # ------------------------------------------------------------------
    # Shared or per-feature vmin / vmax
    # ------------------------------------------------------------------
    if shared_colorscale:
        _all_means = pd.concat([df["mean"] for df in all_dfs], ignore_index=True)
        _global_vmin = _resolve_vmin_vmax(_all_means, vmin)
        _global_vmax = _resolve_vmin_vmax(_all_means, vmax)

    # ------------------------------------------------------------------
    # Build integer index maps and 2-D colour matrices for every feature
    # rows = cats_y index (top → bottom), cols = cats_x index (left → right)
    # ------------------------------------------------------------------
    x_to_pos = {cat: i for i, cat in enumerate(cats_x)}
    y_to_pos = {cat: i for i, cat in enumerate(cats_y)}

    all_mats: list = []
    for gene_df in all_dfs:
        mat = np.full((n_y, n_x), np.nan)
        mat[
            gene_df[category_y].map(y_to_pos).values,
            gene_df[category_x].map(x_to_pos).values,
        ] = gene_df["mean"].values
        all_mats.append(mat)

    # ------------------------------------------------------------------
    # Figure layout
    #
    # Column structure (repeated ncols times):
    #   [heatmap_w, cbar_w, between_gap*]   (* omitted after last column)
    #
    # Heatmap column for plot i : 3*i
    # Cbar    column for plot i : 3*i + 1
    # Between-gap after group i  : 3*i + 2  (only when i < ncols - 1)
    # ------------------------------------------------------------------
    nrows = int(np.ceil(len(features) / ncols))

    _CELL: float = 0.6           # inches per category tick for auto-sizing
    _CBAR_W: float = 0.275
    _BETWEEN_GAP: float = wspace  # gap between adjacent heatmap+cbar groups

    if figsize is not None:
        main_w, main_h = float(figsize[0]), float(figsize[1])
    else:
        main_w = max(n_x * _CELL, 3.0)
        main_h = max(n_y * _CELL, 3.0)

    total_w = (
        ncols * (main_w + _CBAR_W)
        + (ncols - 1) * _BETWEEN_GAP
        + 1.5
    )
    total_h = nrows * main_h + (nrows - 1) * 0.5 + 1.2

    width_ratios: list = []
    for i in range(ncols):
        width_ratios.append(main_w)
        width_ratios.append(_CBAR_W)
        if i < ncols - 1:
            width_ratios.append(_BETWEEN_GAP)

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

    # ------------------------------------------------------------------
    # Draw each subplot
    # ------------------------------------------------------------------
    for idx, (feat, mat) in enumerate(zip(features, all_mats)):
        gene_df = all_dfs[idx]
        row = idx // ncols
        col = idx % ncols

        heatmap_ax = fig.add_subplot(gs[row, 3 * col])
        cbar_ax = fig.add_subplot(gs[row, 3 * col + 1])

        if shared_colorscale:
            _vmin, _vmax = _global_vmin, _global_vmax
        else:
            _vmin = _resolve_vmin_vmax(gene_df["mean"], vmin)
            _vmax = _resolve_vmin_vmax(gene_df["mean"], vmax)

        norm = matplotlib.colors.Normalize(vmin=_vmin, vmax=_vmax)
        im = heatmap_ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )

        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(color_title, fontsize="small")
        cbar.ax.tick_params(labelsize="small")
        cbar.ax.grid(False)

        heatmap_ax.set_title(feat)
        heatmap_ax.set_xticks(range(n_x))
        heatmap_ax.set_yticks(range(n_y))

        # x-axis: labels and xlabel only on the bottom row
        _is_bottom_row = (row == nrows - 1)
        if border_ticks_only and not _is_bottom_row:
            heatmap_ax.set_xticklabels([])
            heatmap_ax.set_xlabel("")
        else:
            heatmap_ax.set_xticklabels(cats_x, rotation=90)
            heatmap_ax.set_xlabel(category_x)

        # y-axis: labels and ylabel only on the first column
        if border_ticks_only and col > 0:
            heatmap_ax.set_yticklabels([])
            heatmap_ax.set_ylabel("")
        else:
            heatmap_ax.set_yticklabels(cats_y)
            heatmap_ax.set_ylabel(category_y)

        heatmap_ax.grid(False)

    return fig


def heatmap_feature_aggregated_three_categories(
    adata: AnnData,
    feature: str,
    x: str,
    y: str,
    panel: str,
    sample_col: str,
    x_ref: str,
    groups_x: Optional[List[str]] = None,
    groups_y: Optional[List[str]] = None,
    groups_panel: Optional[List[str]] = None,
    agg_fn: Literal["mean", "median", "sum"] = "mean",
    min_cells: int = 10,
    min_samples: Optional[int] = None,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    ncols: int = 3,
    shared_colorscale: bool = False,
    zero_center: bool = True,
    cmap: str = "Reds",
    vmin: Optional[Union[str, float]] = None,
    vmax: Optional[Union[str, float]] = None,
    color_title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hspace: float = 0.2,
    wspace: float = 1.2,
    border_ticks_only: bool = True,
    use_zscores: bool = False,
    stat_test: Literal["mann-whitney", "t-test"] = "mann-whitney",
    show_stats: bool = True,
    show_ns: bool = False,
    pvalue_fontsize: Union[str, float] = "x-small",
) -> Figure:
    """Heatmap of a single feature across three categorical variables with per-sample aggregation.

    Creates a multi-panel figure where each panel corresponds to a unique value
    of *panel*.  Within each panel, *x* categories appear on the columns and
    *y* categories on the rows; cell colour encodes the mean of the per-sample
    aggregated feature values.  Optionally, p-value stars are overlaid on each
    non-reference cell by comparing each *x* level against a reference level
    (*x_ref*) using the sample-level aggregated values within the same
    ``(panel, y)`` group.

    Each panel is processed independently following the same aggregation
    pipeline as :func:`plot_feature_boxplot_aggregated`.  Within a panel,
    cell-level values are grouped by ``(sample_col, x, y)`` — matching the
    boxplot's ``groupby([sample_col, x, hue])`` — and aggregated with
    *agg_fn*.  Samples contributing fewer than *min_cells* cells to a group
    are silently dropped.  Statistical tests are then computed on the
    resulting per-sample aggregates using the same ``.dropna()`` and
    ``scipy.stats`` calls as the boxplot's ``_annotate_pvalues``, so
    subsetting *adata* to a single *panel* value and calling
    :func:`plot_feature_boxplot_aggregated` with ``x=x`` and ``hue=y``
    produces identical p-value annotations.

    Args:
        adata: Annotated data matrix.
        feature: Gene name or ``adata.obs`` column to visualise.  Resolved
            with obs-first priority: if *feature* is present in
            ``adata.obs.columns`` it is used directly (and *layer* /
            *gene_symbols* are ignored).  Otherwise it is matched against
            ``adata.var_names`` or ``adata.var[gene_symbols]``.
        x: Column in ``adata.obs`` for the x-axis categories (e.g.
            condition).
        y: Column in ``adata.obs`` for the y-axis categories (e.g. cell
            type).
        panel: Column in ``adata.obs`` whose unique values each become a
            separate heatmap panel (e.g. tissue).
        sample_col: Column in ``adata.obs`` identifying biological samples.
            Values are first aggregated per ``(sample_col, x, y, panel)``
            group before further averaging for the colour scale.
        x_ref: Reference level of *x* used as the baseline for p-value
            computations.  Must be a value present in the *x* column after
            any *groups_x* filtering.
        groups_x: Ordered subset of *x* categories to include.  The list
            order defines the display order on the x-axis.  *x_ref* is
            always retained; if absent from the list it is prepended as the
            first column.  When ``None``, all categories are kept in their
            original order.  Defaults to ``None``.
        groups_y: Ordered subset of *y* categories to include.  The list
            order defines the display order on the y-axis.  When ``None``,
            all categories are kept in their original order.
            Defaults to ``None``.
        groups_panel: Ordered subset of *panel* categories to include.  The
            list order defines the panel sequence in the figure grid.  When
            ``None``, all categories are kept in their original order.
            Defaults to ``None``.
        agg_fn: Aggregation function applied per
            ``(sample_col, x, y, panel)`` group.  One of ``"mean"``,
            ``"median"``, or ``"sum"``.  Defaults to ``"mean"``.
        min_cells: Minimum number of cells a sample must contribute to a
            group to be retained.  Groups with fewer cells are silently
            dropped.  Defaults to ``10``.
        min_samples: Minimum number of per-sample aggregates required for a
            ``(x, y, panel)`` combination to be displayed in colour.
            Combinations below this threshold are overlaid with a grey
            rectangle and receive no annotation, regardless of
            *show_stats*.  When ``None``, no masking is applied.
            Defaults to ``None``.
        layer: Expression layer to use for gene features.  ``None`` uses
            ``adata.X``.  Defaults to ``None``.
        gene_symbols: Column in ``adata.var`` holding alternative gene
            identifiers.  When set, *feature* is matched against that column
            instead of ``adata.var_names``.  Defaults to ``None``.
        ncols: Number of panel columns in the figure grid.  Defaults to
            ``3``.
        shared_colorscale: When ``True``, a single colour scale is used
            across all panels, enabling direct visual comparison.  Defaults
            to ``False``.
        zero_center: When ``True``, the colour scale is made symmetric
            around zero.  If *vmax* is not provided, it is set to the
            maximum absolute value of the data (per-panel when
            *shared_colorscale* is ``False``, or global when ``True``);
            *vmin* is then set to ``-vmax``.  If *vmax* is explicitly
            provided, *vmin* is set to ``-vmax`` (any explicit *vmin* is
            overridden).  Defaults to ``True``.
        cmap: Matplotlib colormap name.  Defaults to ``"Reds"``.
        vmin: Lower colour-scale limit.  Accepts a plain ``float`` or a
            percentile string (e.g. ``"p5"``).  Defaults to ``None``.
        vmax: Upper colour-scale limit.  Defaults to ``None``.
        color_title: Colourbar label.  When ``None``, defaults to
            ``"{feature} ({agg_fn} per sample)"``.  Defaults to ``None``.
        figsize: Size of a **single** heatmap panel ``(width, height)`` in
            inches.  When ``None``, the panel size is derived automatically
            from the number of categories on each axis.  Defaults to
            ``None``.
        hspace: Vertical space between subplot rows, as a fraction of the
            average axes height.  Defaults to ``0.2``.
        wspace: Horizontal gap in inches between adjacent panel groups
            (i.e. between the colourbar of one group and the heatmap of the
            next).  Defaults to ``1.2``.
        border_ticks_only: When ``True``, x-axis tick labels and the x-axis
            label are shown only on the bottom row of subplots, and y-axis
            tick labels and the y-axis label are shown only on the leftmost
            column.  Defaults to ``True``.
        use_zscores: When ``True``, z-score the per-group mean values before
            plotting.  P-values are always computed on the un-z-scored
            per-sample aggregated values.  Defaults to ``False``.
        stat_test: Statistical test for comparisons against *x_ref*.  Either
            ``"mann-whitney"`` (Mann–Whitney U, two-sided) or ``"t-test"``
            (Welch's independent t-test).  Defaults to ``"mann-whitney"``.
        show_stats: Whether to draw p-value annotations on the heatmap
            cells.  Defaults to ``True``.
        show_ns: When ``True`` and *show_stats* is ``True``, annotate
            non-significant cells with ``"ns"``.  When ``False``, only
            significant cells are annotated.  Defaults to ``False``.
        pvalue_fontsize: Font size for p-value annotation text.  Accepts
            any value recognised by matplotlib (e.g. ``"x-small"``,
            ``8``).  Defaults to ``"x-small"``.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If *x*, *y*, *panel*, or *sample_col* is not found in
            ``adata.obs.columns``.
        KeyError: If *feature* cannot be resolved against ``adata.obs``,
            ``adata.var_names``, or ``adata.var[gene_symbols]``.
        ValueError: If *x_ref* is not present in the *x* column after
            filtering.
        ValueError: If *agg_fn* is not one of ``"mean"``, ``"median"``,
            ``"sum"``.
        ValueError: If no samples remain after applying *min_cells*.
        ValueError: If *vmin* or *vmax* is a string not starting with
            ``"p"``.
        ValueError: If any value in *groups_x* / *groups_y* / *groups_panel*
            is not a valid category of the respective column.

    Example:
        >>> fig = heatmap_feature_aggregated_three_categories(
        ...     adata,
        ...     feature="CD3E",
        ...     x="condition",
        ...     y="cell_type",
        ...     panel="tissue",
        ...     sample_col="donor_id",
        ...     x_ref="control",
        ...     cmap="Blues",
        ...     show_stats=True,
        ...     show_ns=False,
        ... )
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for _arg_name, _col_val in [
        ("x", x), ("y", y), ("panel", panel), ("sample_col", sample_col)
    ]:
        if _col_val not in adata.obs.columns:
            raise ValueError(
                f"{_arg_name}='{_col_val}' not found in adata.obs.columns. "
                f"Available columns: {list(adata.obs.columns)}"
            )
    if agg_fn not in ("mean", "median", "sum"):
        raise ValueError(
            f"agg_fn='{agg_fn}' is not supported. "
            "Choose from 'mean', 'median', 'sum'."
        )

    # ------------------------------------------------------------------
    # Resolve feature: obs column takes priority over var (gene)
    # ------------------------------------------------------------------
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
    _layer = None if _is_obs_col else layer
    _gene_symbols = None if _is_obs_col else gene_symbols

    # ------------------------------------------------------------------
    # Build per-cell long-form DataFrame
    # ------------------------------------------------------------------
    obs_tidy = sc.get.obs_df(
        adata,
        keys=[feature, x, y, panel, sample_col],
        use_raw=False,
        layer=_layer,
        gene_symbols=_gene_symbols,
    )

    # Ensure categorical dtypes on grouping columns
    for col in (x, y, panel):
        if obs_tidy[col].dtype.name != "category":
            obs_tidy[col] = obs_tidy[col].astype("category")
        obs_tidy[col] = obs_tidy[col].cat.remove_unused_categories()

    # ------------------------------------------------------------------
    # Apply groups filtering
    # ------------------------------------------------------------------
    if groups_x is not None:
        all_x_cats = obs_tidy[x].cat.categories.tolist()
        invalid_x = [g for g in groups_x if g not in all_x_cats]
        if invalid_x:
            raise ValueError(
                f"groups_x values {invalid_x} not found in x='{x}' "
                f"categories. Available: {all_x_cats}"
            )
        # Always retain the reference level; respect the user-supplied order
        if x_ref not in groups_x:
            keep_x = [x_ref] + list(groups_x)
        else:
            keep_x = list(groups_x)
        obs_tidy = obs_tidy[obs_tidy[x].isin(keep_x)].copy()
        obs_tidy[x] = pd.Categorical(obs_tidy[x], categories=keep_x, ordered=False)

    if groups_y is not None:
        all_y_cats = obs_tidy[y].cat.categories.tolist()
        invalid_y = [g for g in groups_y if g not in all_y_cats]
        if invalid_y:
            raise ValueError(
                f"groups_y values {invalid_y} not found in y='{y}' "
                f"categories. Available: {all_y_cats}"
            )
        obs_tidy = obs_tidy[obs_tidy[y].isin(groups_y)].copy()
        obs_tidy[y] = pd.Categorical(obs_tidy[y], categories=list(groups_y), ordered=False)

    if groups_panel is not None:
        all_panel_cats = obs_tidy[panel].cat.categories.tolist()
        invalid_panel = [g for g in groups_panel if g not in all_panel_cats]
        if invalid_panel:
            raise ValueError(
                f"groups_panel values {invalid_panel} not found in "
                f"panel='{panel}' categories. Available: {all_panel_cats}"
            )
        obs_tidy = obs_tidy[obs_tidy[panel].isin(groups_panel)].copy()
        obs_tidy[panel] = pd.Categorical(obs_tidy[panel], categories=list(groups_panel), ordered=False)

    # ------------------------------------------------------------------
    # Validate x_ref and freeze category lists
    # ------------------------------------------------------------------
    cats_x: List = obs_tidy[x].cat.categories.tolist()
    cats_y: List = obs_tidy[y].cat.categories.tolist()
    cats_panel: List = obs_tidy[panel].cat.categories.tolist()

    if x_ref not in cats_x:
        raise ValueError(
            f"x_ref='{x_ref}' is not a category in x='{x}' after "
            f"filtering. Available categories: {cats_x}"
        )

    n_x = len(cats_x)
    n_y = len(cats_y)
    n_panels = len(cats_panel)

    # ------------------------------------------------------------------
    # Per-panel aggregation and statistical testing
    #
    # For each panel value, data is processed independently using the
    # same pipeline as plot_feature_boxplot_aggregated:
    #   1. Subset obs_tidy to the panel
    #   2. Group by (sample_col, x, y) — matching the boxplot's
    #      groupby([sample_col, x, hue], observed=True)
    #   3. Apply min_cells filter on the per-group cell count
    #   4. Compute statistical tests following the same approach as
    #      _annotate_pvalues in boxplots.py (including .dropna())
    #
    # This ensures that subsetting adata to a single panel value and
    # calling plot_feature_boxplot_aggregated with x=x, hue=y produces
    # identical p-value annotations.
    # ------------------------------------------------------------------

    # Force float64 — matches boxplot's values.astype(float) step
    obs_tidy[feature] = obs_tidy[feature].astype(float)

    all_panel_agg: list = []
    sample_count_lookup: dict = {}
    pvalue_stars: dict = {}

    for p_val in cats_panel:
        panel_df = obs_tidy[obs_tidy[panel] == p_val]

        # Aggregate per (sample_col, x, y) — same groupby as the
        # boxplot's groupby([sample_col, x, hue], observed=True)
        group_cols = [sample_col, x, y]
        grouped = panel_df.groupby(group_cols, observed=True)
        cell_counts = grouped[feature].count()
        agg_values = getattr(grouped[feature], agg_fn)()

        # Apply min_cells filter — same as boxplot
        panel_agg_df = agg_values[cell_counts >= min_cells].reset_index()

        if panel_agg_df.empty:
            continue

        # Tag with panel value and collect
        panel_agg_df[panel] = p_val
        all_panel_agg.append(panel_agg_df)

        # Sample-count lookup for this panel
        _sc = (
            panel_agg_df.groupby([x, y], observed=True)[feature]
            .count()
            .reset_index()
            .rename(columns={feature: "n_samples"})
        )
        for _, _row in _sc.iterrows():
            sample_count_lookup[
                (p_val, _row[y], _row[x])
            ] = int(_row["n_samples"])

        # --------------------------------------------------------------
        # Statistical test — mirrors _annotate_pvalues from boxplots.py
        #
        # For each y category (= boxplot hue), compare each x_val
        # against x_ref using the per-sample aggregated values, with
        # .dropna() applied exactly as in _annotate_pvalues.
        # --------------------------------------------------------------
        if show_stats:
            for y_val in cats_y:
                subset = panel_agg_df[panel_agg_df[y] == y_val]
                ref_vals = (
                    subset[subset[x] == x_ref][feature].dropna().values
                )
                for x_val in cats_x:
                    if x_val == x_ref:
                        continue
                    test_vals = (
                        subset[subset[x] == x_val][feature]
                        .dropna()
                        .values
                    )
                    if len(ref_vals) < 2 or len(test_vals) < 2:
                        if show_ns:
                            pvalue_stars[(p_val, y_val, x_val)] = "ns"
                        continue
                    if stat_test == "mann-whitney":
                        _, p = scipy.stats.mannwhitneyu(
                            test_vals,
                            ref_vals,
                            alternative="two-sided",
                        )
                    else:
                        _, p = scipy.stats.ttest_ind(
                            test_vals,
                            ref_vals,
                            equal_var=False,
                        )
                    stars = _pvalue_to_stars(p)
                    if stars != "ns" or show_ns:
                        pvalue_stars[(p_val, y_val, x_val)] = stars

    # Combine all panel aggregates
    if not all_panel_agg:
        raise ValueError(
            f"No samples remain after applying min_cells={min_cells}. "
            "Lower the threshold or check your grouping columns."
        )
    agg_df = pd.concat(all_panel_agg, ignore_index=True)

    # Restore categorical ordering on agg_df
    agg_df[x] = pd.Categorical(agg_df[x], categories=cats_x)
    agg_df[y] = pd.Categorical(agg_df[y], categories=cats_y)
    agg_df[panel] = pd.Categorical(agg_df[panel], categories=cats_panel)

    # ------------------------------------------------------------------
    # Compute mean per (x, y, panel) for colour matrix
    # ------------------------------------------------------------------
    mean_df = (
        agg_df.groupby([x, y, panel], observed=True)[feature]
        .mean()
        .reset_index()
        .rename(columns={feature: "mean"})
    )

    if use_zscores:
        mean_df["mean"] = scipy.stats.zscore(mean_df["mean"].values)

    # ------------------------------------------------------------------
    # Resolve global vmin / vmax
    # ------------------------------------------------------------------
    if shared_colorscale:
        _global_vmin = _resolve_vmin_vmax(mean_df["mean"], vmin)
        _global_vmax = _resolve_vmin_vmax(mean_df["mean"], vmax)
        if zero_center:
            if _global_vmax is None:
                _global_vmax = float(mean_df["mean"].abs().max())
            _global_vmin = -_global_vmax

    # ------------------------------------------------------------------
    # Build index maps and per-panel colour matrices
    # rows = cats_y index (top → bottom), cols = cats_x index (left → right)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Figure layout (mirrors heatmap_expression_two_categories_multiplot)
    #
    # Column structure repeated ncols times:
    #   [heatmap_w, cbar_w, between_gap*]  (* omitted after last column)
    #
    # Heatmap column for panel i : 3*col
    # Cbar    column for panel i : 3*col + 1
    # Between-gap after group i  : 3*col + 2  (only when col < ncols - 1)
    # ------------------------------------------------------------------
    _ncols = min(ncols, n_panels)  # avoid empty gridspec columns
    nrows = int(np.ceil(n_panels / _ncols))

    _CELL: float = 0.6
    _CBAR_W: float = 0.275
    _BETWEEN_GAP: float = wspace

    if figsize is not None:
        main_w, main_h = float(figsize[0]), float(figsize[1])
    else:
        main_w = max(n_x * _CELL, 3.0)
        main_h = max(n_y * _CELL, 3.0)

    total_w = (
        _ncols * (main_w + _CBAR_W)
        + (_ncols - 1) * _BETWEEN_GAP
        + 1.5
    )
    total_h = nrows * main_h + (nrows - 1) * 0.5 + 1.2

    width_ratios: List = []
    for i in range(_ncols):
        width_ratios.append(main_w)
        width_ratios.append(_CBAR_W)
        if i < _ncols - 1:
            width_ratios.append(_BETWEEN_GAP)

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

    _color_title = (
        color_title
        if color_title is not None
        else f"{feature} ({agg_fn} per sample)"
    )

    # ------------------------------------------------------------------
    # Draw each panel
    # ------------------------------------------------------------------
    for idx, p_val in enumerate(cats_panel):
        row = idx // _ncols
        col = idx % _ncols

        heatmap_ax = fig.add_subplot(gs[row, 3 * col])
        cbar_ax = fig.add_subplot(gs[row, 3 * col + 1])

        # Per-panel: keep only x/y categories that have at least one value
        p_df = mean_df[mean_df[panel] == p_val].dropna(subset=["mean"])
        active_x_set = set(p_df[x].unique())
        active_y_set = set(p_df[y].unique())
        active_x = [c for c in cats_x if c in active_x_set]
        active_y = [c for c in cats_y if c in active_y_set]
        n_ax = len(active_x)
        n_ay = len(active_y)
        x_pos = {cat: i for i, cat in enumerate(active_x)}
        y_pos = {cat: i for i, cat in enumerate(active_y)}

        mat = np.full((n_ay, n_ax), np.nan)
        for _, r in p_df.iterrows():
            xi = x_pos.get(r[x])
            yi = y_pos.get(r[y])
            if xi is not None and yi is not None:
                mat[yi, xi] = r["mean"]

        if shared_colorscale:
            _vmin, _vmax = _global_vmin, _global_vmax
        else:
            _vmin = _resolve_vmin_vmax(p_df["mean"], vmin)
            _vmax = _resolve_vmin_vmax(p_df["mean"], vmax)
            if zero_center:
                if _vmax is None:
                    _vmax = float(p_df["mean"].abs().max())
                _vmin = -_vmax

        norm = matplotlib.colors.Normalize(vmin=_vmin, vmax=_vmax)
        im = heatmap_ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )

        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(_color_title, fontsize="small")
        cbar.ax.tick_params(labelsize="small")
        cbar.ax.grid(False)

        # ---- Grey masking for low-sample cells --------------------------
        if min_samples is not None:
            for x_val in active_x:
                for y_val in active_y:
                    if sample_count_lookup.get((p_val, y_val, x_val), 0) < min_samples:
                        heatmap_ax.add_patch(
                            matplotlib.patches.Rectangle(
                                (x_pos[x_val] - 0.5, y_pos[y_val] - 0.5),
                                1, 1,
                                linewidth=0,
                                facecolor="lightgray",
                                zorder=2,
                            )
                        )

        # ---- P-value annotations ----------------------------------------
        if show_stats:
            for x_val in active_x:
                if x_val == x_ref:
                    continue
                col_pos = x_pos[x_val]
                for y_val in active_y:
                    row_pos = y_pos[y_val]
                    # Skip annotation for grey-masked cells
                    if min_samples is not None and (
                        sample_count_lookup.get((p_val, y_val, x_val), 0) < min_samples
                    ):
                        continue
                    stars = pvalue_stars.get((p_val, y_val, x_val))
                    if stars is not None:
                        heatmap_ax.text(
                            col_pos,
                            row_pos,
                            stars,
                            ha="center",
                            va="center",
                            fontsize=pvalue_fontsize,
                            color="black",
                        )

        # ---- Axes formatting --------------------------------------------
        heatmap_ax.set_title(str(p_val))
        heatmap_ax.set_xticks(range(n_ax))
        heatmap_ax.set_yticks(range(n_ay))

        _is_bottom_row = row == nrows - 1
        if border_ticks_only and not _is_bottom_row:
            heatmap_ax.set_xticklabels([])
            heatmap_ax.set_xlabel("")
        else:
            heatmap_ax.set_xticklabels(active_x, rotation=90)
            heatmap_ax.set_xlabel(x)

        if border_ticks_only and col > 0:
            heatmap_ax.set_yticklabels([])
            heatmap_ax.set_ylabel("")
        else:
            heatmap_ax.set_yticklabels(active_y)
            heatmap_ax.set_ylabel(y)

        heatmap_ax.grid(False)

    return fig


def plot_correlation_distance_heatmap(
    adata: AnnData,
    groupby: str,
    linkage_method: Literal["complete", "average", "ward", "single"] = "complete",
    dendrogram_key: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 7.0),
    cmap: str = "RdYlGn_r",
    annot_fmt: str = ".2f",
    annot_fontsize: float = 7.0,
    dendrogram_ratio: float = 0.15,
    show_top_dendrogram: bool = True,
    colorbar_label: str = "Distance (1 \u2212 r)",
) -> matplotlib.figure.Figure:
    """Plot a pairwise-distance heatmap with dendrogram from a Scanpy correlation matrix.

    Reads the correlation matrix stored in ``adata.uns`` by
    ``sc.tl.dendrogram`` and converts it to a distance matrix
    (``1 \u2212 correlation``).  The groups are reordered to match the
    dendrogram leaf order used by Scanpy, and the distances are
    annotated inside each cell of the heatmap.

    Args:
        adata: Annotated data matrix.  Must contain a dendrogram entry
            in ``adata.uns`` computed with ``sc.tl.dendrogram``.
        groupby: The ``groupby`` key passed to ``sc.tl.dendrogram``
            (e.g. ``"cell_type"``).  Used to look up
            ``adata.uns["dendrogram_{groupby}"]``.
        linkage_method: Hierarchical linkage method used to recompute the
            dendrogram for the distance matrix.  Defaults to
            ``"complete"`` (matching the Scanpy default).
        dendrogram_key: Full key in ``adata.uns`` for the dendrogram dict.
            When ``None``, defaults to ``"dendrogram_{groupby}"``.
        figsize: Overall figure size in inches.  Defaults to ``(8.0, 7.0)``.
        cmap: Matplotlib colormap name for the heatmap.  Defaults to
            ``"RdYlGn_r"`` (low distance = green, high = red).
        annot_fmt: Format string for distance annotations.
            Defaults to ``".2f"``.
        annot_fontsize: Font size for in-cell annotations.
            Defaults to ``7.0``.
        dendrogram_ratio: Fraction of the figure width/height allocated to
            the left dendrogram panel (and the top dendrogram panel when
            ``show_top_dendrogram=True``).  Defaults to ``0.15``.
        show_top_dendrogram: When ``True`` (default), draw the top
            dendrogram panel above the heatmap.  Set to ``False`` to omit
            it and give the full vertical space to the heatmap.
        colorbar_label: Label for the colour bar.
            Defaults to ``"Distance (1 \u2212 r)"``.

    Returns:
        The matplotlib figure.

    Raises:
        KeyError: If the dendrogram key is not found in ``adata.uns``.
        KeyError: If ``"correlation_matrix"`` is absent from the dendrogram
            entry.

    Example:
        >>> import scanpy as sc
        >>> sc.tl.dendrogram(adata, groupby="cell_type")
        >>> fig = plot_correlation_distance_heatmap(adata, groupby="cell_type")
        >>> fig.savefig("distance_heatmap.png", dpi=150)
    """
    # ------------------------------------------------------------------ #
    # 1. Retrieve stored dendrogram data                                  #
    # ------------------------------------------------------------------ #
    if dendrogram_key is None:
        dendrogram_key = f"dendrogram_{groupby}"

    if dendrogram_key not in adata.uns:
        raise KeyError(
            f"Dendrogram key '{dendrogram_key}' not found in adata.uns. "
            f"Run sc.tl.dendrogram(adata, groupby='{groupby}') first."
        )

    dendro_data = adata.uns[dendrogram_key]

    if "correlation_matrix" not in dendro_data:
        raise KeyError(
            f"'correlation_matrix' not found in adata.uns['{dendrogram_key}']. "
            "Re-run sc.tl.dendrogram() to populate it."
        )

    cor_matrix = np.array(dendro_data["correlation_matrix"])
    # ``categories_ordered`` is the LEAF-ORDER list.  The correlation_matrix
    # rows/columns follow the ORIGINAL adata.obs[groupby].cat.categories order.
    categories_ordered: list = list(dendro_data["categories_ordered"])

    # ------------------------------------------------------------------ #
    # 2. Resolve original category order (= row/col order of cor_matrix)  #
    # ------------------------------------------------------------------ #
    # Scanpy stores the correlation_matrix in the order of the original
    # categorical dtype, NOT in leaf order.  We must label the DataFrame
    # with the original order so that values and labels stay aligned.
    if hasattr(adata.obs[groupby], "cat"):
        original_categories: list = list(adata.obs[groupby].cat.categories)
    else:
        # Recover original order by inverting the stored permutation.
        idx_ordered = list(dendro_data["categories_idx_ordered"])
        original_categories = [""] * len(idx_ordered)
        for new_i, orig_i in enumerate(idx_ordered):
            original_categories[orig_i] = categories_ordered[new_i]

    # ------------------------------------------------------------------ #
    # 3. Build distance matrix with original category labels              #
    # ------------------------------------------------------------------ #
    distance_matrix = 1.0 - cor_matrix
    distance_df = pd.DataFrame(
        distance_matrix,
        index=original_categories,
        columns=original_categories,
    )

    # ------------------------------------------------------------------ #
    # 4. Linkage and leaf order                                            #
    # ------------------------------------------------------------------ #
    # Use the linkage stored by sc.tl.dendrogram when available so that
    # the rendered dendrogram is identical to the Scanpy one.
    if "linkage" in dendro_data:
        Z = np.array(dendro_data["linkage"])
    else:
        condensed = squareform(distance_df.values, checks=False)
        Z = linkage(condensed, method=linkage_method)

    ddata_ref = dendrogram(Z, labels=original_categories, no_plot=True)
    leaf_order: list[int] = ddata_ref["leaves"]
    ordered_labels = [original_categories[i] for i in leaf_order]

    # ------------------------------------------------------------------ #
    # 5. Reorder distance matrix to match dendrogram leaves               #
    # ------------------------------------------------------------------ #
    dist_ordered = distance_df.loc[ordered_labels, ordered_labels].values
    n = len(ordered_labels)

    # ------------------------------------------------------------------ #
    # 6. Build figure layout                                              #
    # ------------------------------------------------------------------ #
    # layout="constrained" lets matplotlib automatically reserve space for
    # tick labels so they are never clipped by adjacent axes.
    #
    #   show_top_dendrogram=True     show_top_dendrogram=False
    #   +----------+-----------+    +----------+-----------+
    #   |  corner  | top dendro|    |  left    |  heatmap  |
    #   +----------+-----------+    |  dendro  |           |
    #   |  left    |  heatmap  |    |          |           |
    #   |  dendro  |           |    +----------+-----------+
    #   +----------+-----------+

    dr = dendrogram_ratio

    if show_top_dendrogram:
        n_rows = 2
        height_ratios = [dr, 1.0 - dr]
    else:
        n_rows = 1
        height_ratios = [1.0]

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(
        n_rows, 2,
        width_ratios=[dr, 1.0 - dr],
        height_ratios=height_ratios,
    )

    if show_top_dendrogram:
        ax_corner = fig.add_subplot(gs[0, 0])
        ax_dendro_top = fig.add_subplot(gs[0, 1])
        ax_dendro_left = fig.add_subplot(gs[1, 0])
        ax_heatmap = fig.add_subplot(gs[1, 1])
        ax_corner.set_visible(False)
    else:
        ax_dendro_left = fig.add_subplot(gs[0, 0])
        ax_heatmap = fig.add_subplot(gs[0, 1])

    # ------------------------------------------------------------------ #
    # 7. Left dendrogram                                                  #
    # ------------------------------------------------------------------ #
    dendrogram(
        Z,
        labels=ordered_labels,
        ax=ax_dendro_left,
        orientation="left",
        link_color_func=lambda _: "black",
        no_labels=True,
    )
    ax_dendro_left.invert_yaxis()
    ax_dendro_left.axis("off")

    # ------------------------------------------------------------------ #
    # 8. Top dendrogram — aligned to heatmap columns                      #
    # ------------------------------------------------------------------ #
    if show_top_dendrogram:
        # scipy places leaf i at position 10*(i+1) - 5  ->  5, 15, 25, ...
        # Setting xlim to [0, 10n] maps each leaf at 10k-5 exactly above
        # heatmap column k-1, since imshow x-range is [-0.5, n-0.5].
        dendrogram(
            Z,
            labels=ordered_labels,
            ax=ax_dendro_top,
            orientation="top",
            link_color_func=lambda _: "black",
            no_labels=True,
        )
        ax_dendro_top.set_xlim(0, 10.0 * n)
        ax_dendro_top.axis("off")

    # ------------------------------------------------------------------ #
    # 9. Heatmap                                                           #
    # ------------------------------------------------------------------ #
    vmax_val = float(np.nanmax(dist_ordered[~np.eye(n, dtype=bool)]))

    im = ax_heatmap.imshow(
        dist_ordered,
        cmap=cmap,
        aspect="auto",
        vmin=0.0,
        vmax=vmax_val,
    )

    ax_heatmap.set_facecolor("white")
    ax_heatmap.grid(False)
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)

    # X ticks at the bottom
    ax_heatmap.set_xticks(range(n))
    ax_heatmap.set_xticklabels(ordered_labels, rotation=90, fontsize=8)
    ax_heatmap.xaxis.set_ticks_position("bottom")
    ax_heatmap.xaxis.set_label_position("bottom")

    # Y ticks on the right
    ax_heatmap.set_yticks(range(n))
    ax_heatmap.set_yticklabels(ordered_labels, fontsize=8)
    ax_heatmap.yaxis.set_ticks_position("right")
    ax_heatmap.yaxis.set_label_position("right")

    # Annotations
    for i in range(n):
        for j in range(n):
            val = dist_ordered[i, j]
            text_color = "white" if val > vmax_val * 0.65 else "black"
            ax_heatmap.text(
                j, i,
                format(val, annot_fmt),
                ha="center", va="center",
                fontsize=annot_fontsize,
                color=text_color,
            )

    # ------------------------------------------------------------------ #
    # 10. Colorbar: horizontal, below heatmap      #
    # ------------------------------------------------------------------ #
    # Using ax=ax_heatmap (not cax) lets constrained_layout handle the
    # spacing between the heatmap tick labels and the colorbar.
    cbar = fig.colorbar(
        im,
        ax=ax_heatmap,
        location="bottom",
        shrink=1.0,
        pad=0.05,
        aspect=40,
    )
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    return fig
