from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
from anndata import AnnData
from matplotlib.figure import Figure


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
