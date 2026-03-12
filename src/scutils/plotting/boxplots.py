from __future__ import annotations

import itertools
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure
from scipy import stats

from scutils.plotting._utils import _resolve_palette


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_feature(
    adata: AnnData,
    feature: str,
    layer: Optional[str],
    gene_symbols: Optional[str],
) -> Tuple[pd.Series, bool]:
    """Extract a feature vector from *adata*, preferring obs columns over var.

    Args:
        adata: Annotated data matrix.
        feature: Name of an ``adata.obs`` column or a gene in ``adata.var``.
        layer: Layer to pull gene expression from. Ignored for obs features.
        gene_symbols: Column in ``adata.var`` holding alternative gene
            identifiers. Ignored for obs features.

    Returns:
        A tuple ``(values, is_obs)`` where *values* is a ``pd.Series`` indexed
        by cell barcode and *is_obs* indicates whether the feature came from
        ``adata.obs``.

    Raises:
        ValueError: If *feature* cannot be found in either ``adata.obs`` or
            ``adata.var``.
    """
    if feature in adata.obs.columns:
        return adata.obs[feature].copy(), True

    # Fall back to var / gene expression
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

    series = sc.get.obs_df(
        adata,
        keys=[feature],
        use_raw=False,
        layer=layer,
        gene_symbols=gene_symbols,
    )[feature]
    return series, False


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
    """Convert a p-value to an asterisk string.

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


def _annotate_pvalues(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    hue: str,
    value_col: str,
    comparisons: Optional[List[Tuple[str, str]]],
    orient: Literal["v", "h"],
    test: Literal["mann-whitney", "t-test"],
) -> None:
    """Draw significance brackets between hue groups on *ax*.

    Brackets are drawn for every (x_category, hue_pair) combination. Only
    comparisons that are significant (p < 0.05) are annotated unless
    *comparisons* filters to specific hue pairs. The annotation is skipped
    silently when a group has fewer than 2 observations.

    Args:
        ax: The axes to annotate.
        data: Long-form DataFrame with columns *x*, *hue*, and *value_col*.
        x: Column used for the x-axis grouping.
        hue: Column used for the hue grouping.
        value_col: Column holding the numeric values.
        comparisons: Pairs of hue categories to compare. If ``None``, all
            pairwise combinations are tested.
        orient: ``"v"`` for vertical boxplots, ``"h"`` for horizontal.
        test: Statistical test to use. ``"mann-whitney"`` runs a two-sided
            Mann–Whitney U test; ``"t-test"`` runs Welch's t-test.
    """
    x_cats = data[x].cat.categories.tolist() if hasattr(data[x], "cat") else sorted(data[x].unique())
    hue_cats = data[hue].cat.categories.tolist() if hasattr(data[hue], "cat") else sorted(data[hue].unique())
    n_hue = len(hue_cats)
    n_x = len(x_cats)

    all_pairs = list(itertools.combinations(hue_cats, 2))
    pairs_to_test = comparisons if comparisons is not None else all_pairs

    # seaborn places hue groups symmetrically around each x tick
    # group_width ≈ 0.8, each hue bar width = group_width / n_hue
    group_width = 0.8
    bar_width = group_width / n_hue

    # Build a map: (x_cat, hue_cat) → centre x-position
    tick_positions = {cat: i for i, cat in enumerate(x_cats)}

    def _bar_centre(x_cat: str, hue_cat: str) -> float:
        hue_idx = hue_cats.index(hue_cat)
        offset = (hue_idx - (n_hue - 1) / 2) * bar_width
        return tick_positions[x_cat] + offset

    # Determine current top of plot for stacking brackets
    if orient == "v":
        y_max_global = ax.get_ylim()[1]
    else:
        y_max_global = ax.get_xlim()[1]

    bracket_step = (y_max_global - ax.get_ylim()[0]) * 0.08 if orient == "v" else (y_max_global - ax.get_xlim()[0]) * 0.08
    current_top: Dict[str, float] = {}  # key: x_cat, tracks stacking height

    for x_cat in x_cats:
        subset = data[data[x] == x_cat]
        level = 0
        for h1, h2 in pairs_to_test:
            g1 = subset[subset[hue] == h1][value_col].dropna().values
            g2 = subset[subset[hue] == h2][value_col].dropna().values
            if len(g1) < 2 or len(g2) < 2:
                continue

            if test == "mann-whitney":
                _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            else:
                _, p = stats.ttest_ind(g1, g2, equal_var=False)

            label = _pvalue_to_stars(p)
            if label == "ns":
                continue

            x1 = _bar_centre(x_cat, h1)
            x2 = _bar_centre(x_cat, h2)

            # Stack brackets per x_cat
            key = str(x_cat)
            base = current_top.get(key, y_max_global * 0.97 if orient == "v" else y_max_global * 0.97)
            bracket_y = base + level * bracket_step

            if orient == "v":
                ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + bracket_step * 0.3,
                                             bracket_y + bracket_step * 0.3, bracket_y],
                        lw=1.0, color="black")
                ax.text((x1 + x2) / 2, bracket_y + bracket_step * 0.35, label,
                        ha="center", va="bottom", fontsize=9)
                if level == 0:
                    current_top[key] = bracket_y
            else:
                ax.plot([bracket_y, bracket_y + bracket_step * 0.3,
                         bracket_y + bracket_step * 0.3, bracket_y],
                        [x1, x1, x2, x2],
                        lw=1.0, color="black")
                ax.text(bracket_y + bracket_step * 0.35, (x1 + x2) / 2, label,
                        ha="left", va="center", fontsize=9)
                if level == 0:
                    current_top[key] = bracket_y

            level += 1


def _annotate_pvalues_single_group(
    ax: plt.Axes,
    data: pd.DataFrame,
    hue: str,
    value_col: str,
    comparisons: Optional[List[Tuple[str, str]]],
    orient: Literal["v", "h"],
    test: Literal["mann-whitney", "t-test"],
) -> None:
    """Draw significance brackets for a single-group panel where *hue* is the x-axis.

    Used by :func:`plot_feature_boxplot_multiplot` where each panel shows a
    single *x* category with *hue* groups drawn as separate boxes along the
    categorical axis (integer positions 0, 1, 2 …).

    Significant pairs are collected in a first pass, the axis limit is then
    expanded upward to give each bracket its own space, and brackets are drawn
    in a second pass so that none are clipped or overlapping.

    Args:
        ax: The axes to annotate.
        data: Long-form DataFrame with columns *hue* and *value_col*.
        hue: Column drawn as the categorical axis of the panel.
        value_col: Column holding the numeric values.
        comparisons: Pairs of hue categories to compare.  ``None`` tests all
            pairwise combinations.
        orient: ``"v"`` for vertical boxplots, ``"h"`` for horizontal.
        test: Statistical test to use.
    """
    hue_cats = (
        data[hue].cat.categories.tolist()
        if hasattr(data[hue], "cat")
        else sorted(data[hue].unique())
    )
    all_pairs = list(itertools.combinations(hue_cats, 2))
    pairs_to_test = comparisons if comparisons is not None else all_pairs

    # Each hue category occupies its own integer tick position
    hue_positions = {cat: i for i, cat in enumerate(hue_cats)}

    # --- First pass: collect significant comparisons ---
    sig_pairs: list = []
    for h1, h2 in pairs_to_test:
        g1 = data[data[hue] == h1][value_col].dropna().values
        g2 = data[data[hue] == h2][value_col].dropna().values
        if len(g1) < 2 or len(g2) < 2:
            continue
        if test == "mann-whitney":
            _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        else:
            _, p = stats.ttest_ind(g1, g2, equal_var=False)
        label = _pvalue_to_stars(p)
        if label != "ns":
            sig_pairs.append((h1, h2, label))

    if not sig_pairs:
        return

    # Current axis data range
    if orient == "v":
        y_lo, y_hi = ax.get_ylim()
    else:
        y_lo, y_hi = ax.get_xlim()

    data_range = y_hi - y_lo
    # Each bracket occupies one slot: arm height + text + inter-bracket gap
    bracket_slot = data_range * 0.15
    gap_above = data_range * 0.03  # small gap between data and first bracket

    # Expand axis so all brackets are fully visible.
    # extra_top gives clearance above the topmost annotation text.
    extra_top = bracket_slot * 0.5
    new_hi = y_hi + gap_above + len(sig_pairs) * bracket_slot + extra_top
    if orient == "v":
        ax.set_ylim(y_lo, new_hi)
    else:
        ax.set_xlim(y_lo, new_hi)

    # --- Second pass: draw brackets ---
    arm_h = bracket_slot * 0.4  # height of the vertical arms
    for level, (h1, h2, label) in enumerate(sig_pairs):
        x1 = hue_positions[h1]
        x2 = hue_positions[h2]
        bracket_y = y_hi + gap_above + level * bracket_slot

        if orient == "v":
            ax.plot(
                [x1, x1, x2, x2],
                [bracket_y, bracket_y + arm_h,
                 bracket_y + arm_h, bracket_y],
                lw=1.0, color="black",
            )
            ax.text(
                (x1 + x2) / 2, bracket_y + arm_h * 1.15, label,
                ha="center", va="bottom", fontsize=9,
            )
        else:
            ax.plot(
                [bracket_y, bracket_y + arm_h,
                 bracket_y + arm_h, bracket_y],
                [x1, x1, x2, x2],
                lw=1.0, color="black",
            )
            ax.text(
                bracket_y + arm_h * 1.15, (x1 + x2) / 2, label,
                ha="left", va="center", fontsize=9,
            )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def plot_feature_boxplot(
    adata: AnnData,
    feature: str,
    x: str,
    hue: Optional[str] = None,
    groups_x: Optional[List[str]] = None,
    groups_hue: Optional[List[str]] = None,
    x_order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    figsize: Tuple[float, float] = (6.0, 4.0),
    orient: Literal["v", "h"] = "v",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_points: bool = False,
    point_size: float = 2.0,
    point_alpha: float = 0.4,
    legend_loc: Literal["outside right", "outside top", "best"] = "outside right",
    comparisons: Optional[List[Tuple[str, str]]] = None,
    stat_test: Literal["mann-whitney", "t-test"] = "mann-whitney",
    show_stats: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> Figure:
    """Boxplot of a feature per cell, grouped by one or two ``adata.obs`` columns.

    Plots the raw per-cell values of *feature* (a gene or an ``adata.obs``
    column) as a seaborn boxplot split by *x* on the primary axis and
    optionally by *hue* as a secondary grouping. When *hue* is provided and
    *show_stats* is ``True``, Mann–Whitney U (or Welch's t-test) p-values are
    drawn as asterisk brackets between *hue* groups within each *x* category.

    Args:
        adata: Annotated data matrix.
        feature: Gene name or ``adata.obs`` column to visualise. Resolved
            against ``adata.obs.columns`` first, then ``adata.var_names`` (or
            ``adata.var[gene_symbols]`` when supplied).
        x: Column in ``adata.obs`` to use as the primary grouping axis.
        hue: Column in ``adata.obs`` to use as the secondary (colour)
            grouping. If ``None``, only *x* is used. Defaults to ``None``.
        groups_x: Subset of *x* category values to include. When ``None``,
            all categories are included. Defaults to ``None``.
        groups_hue: Subset of *hue* category values to include. Requires
            *hue* to be set. When ``None``, all hue categories are included.
            Defaults to ``None``.
        x_order: Explicit display order for *x* categories. Must contain
            every value that will be plotted (after applying *groups_x*).
            When ``None``, the original categorical order is used.
            Defaults to ``None``.
        hue_order: Explicit display order for *hue* categories. Must contain
            every value that will be plotted (after applying *groups_hue*).
            When ``None``, the original categorical order is used.
            Defaults to ``None``.
        layer: Expression layer to use for gene features. Ignored for obs
            columns. Defaults to ``None`` (uses ``adata.X``).
        gene_symbols: Column in ``adata.var`` holding alternative gene
            identifiers. Ignored for obs features. Defaults to ``None``.
        palette: Colour palette for *hue* (or *x* when *hue* is ``None``).
            Accepts a single colour string, a list of colours, a dict mapping
            category labels to colours, or a seaborn/matplotlib palette name.
            Defaults to ``None`` (seaborn default).
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``(6.0, 4.0)``.
        orient: ``"v"`` for vertical boxplots (feature on y-axis) or ``"h"``
            for horizontal (feature on x-axis). Defaults to ``"v"``.
        vmin: Lower clip limit for the feature values. Defaults to ``None``.
        vmax: Upper clip limit for the feature values. Defaults to ``None``.
        show_points: Overlay individual data points as a strip plot on top of
            each box. Defaults to ``False``.
        point_size: Marker size for the strip plot. Defaults to ``2.0``.
        point_alpha: Opacity of strip-plot points. Defaults to ``0.4``.
        legend_loc: Legend placement. ``"outside right"`` places the legend
            outside the axes on the right (default); ``"outside top"`` places
            it above the axes; ``"best"`` lets matplotlib choose inside the
            axes.
        comparisons: List of ``(hue_cat_a, hue_cat_b)`` pairs to annotate with
            p-value brackets. Requires *hue* and *show_stats* to be set.
            If ``None`` and *show_stats* is ``True``, all pairwise hue
            combinations are tested. Defaults to ``None``.
        stat_test: Statistical test for pairwise comparisons. Either
            ``"mann-whitney"`` (Mann–Whitney U, two-sided) or ``"t-test"``
            (Welch's independent t-test). Defaults to ``"mann-whitney"``.
        show_stats: Whether to draw significance brackets. Requires *hue*.
            Defaults to ``False``.
        title: Axes title. Defaults to *feature*.
        xlabel: x-axis label. Defaults to *x*.
        ylabel: y-axis label. Defaults to *feature*.
        **kwargs: Additional keyword arguments forwarded to
            ``seaborn.boxplot``.

    Returns:
        The matplotlib ``Figure`` object.

    Raises:
        ValueError: If *x* or *hue* is not found in ``adata.obs.columns``.
        ValueError: If *feature* cannot be resolved to an obs column or gene.
        ValueError: If *show_stats* is ``True`` but *hue* is not provided.
        ValueError: If any value in *groups_x* / *groups_hue* is not a valid
            category of the respective column.
        ValueError: If *x_order* / *hue_order* does not cover all plotted
            categories.

    Example:
        >>> fig = plot_feature_boxplot(
        ...     adata,
        ...     feature="CD3E",
        ...     x="cell_type",
        ...     hue="condition",
        ...     show_stats=True,
        ...     comparisons=[("ctrl", "stim")],
        ...     palette="Set2",
        ... )
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if x not in adata.obs.columns:
        raise ValueError(
            f"x='{x}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if hue is not None and hue not in adata.obs.columns:
        raise ValueError(
            f"hue='{hue}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if show_stats and hue is None:
        raise ValueError("show_stats=True requires a hue column to be specified.")

    # ------------------------------------------------------------------
    # Build long-form DataFrame
    # ------------------------------------------------------------------
    values, _ = _resolve_feature(adata, feature, layer, gene_symbols)
    values = values.astype(float)
    if vmin is not None:
        values = values.clip(lower=vmin)
    if vmax is not None:
        values = values.clip(upper=vmax)

    plot_df = adata.obs[[x] + ([hue] if hue else [])].copy()
    plot_df["_value"] = values.values

    # Ensure categoricals preserve a stable order
    if not hasattr(plot_df[x], "cat"):
        plot_df[x] = pd.Categorical(plot_df[x])
    if hue is not None and not hasattr(plot_df[hue], "cat"):
        plot_df[hue] = pd.Categorical(plot_df[hue])

    # --- Apply groups_x / groups_hue subsetting ---
    all_x_cats: List[str] = plot_df[x].cat.categories.tolist()
    if groups_x is not None:
        invalid_x = [g for g in groups_x if g not in all_x_cats]
        if invalid_x:
            raise ValueError(
                f"groups_x values {invalid_x} not found in x='{x}' categories. "
                f"Available: {all_x_cats}"
            )
        plot_df = plot_df[plot_df[x].isin(groups_x)].copy()
        plot_df[x] = plot_df[x].cat.remove_unused_categories()

    if hue is not None and groups_hue is not None:
        all_hue_cats: List[str] = plot_df[hue].cat.categories.tolist()
        invalid_hue = [g for g in groups_hue if g not in all_hue_cats]
        if invalid_hue:
            raise ValueError(
                f"groups_hue values {invalid_hue} not found in hue='{hue}' categories. "
                f"Available: {all_hue_cats}"
            )
        plot_df = plot_df[plot_df[hue].isin(groups_hue)].copy()
        plot_df[hue] = plot_df[hue].cat.remove_unused_categories()

    # --- Apply x_order / hue_order ---
    if x_order is not None:
        current_x_cats = plot_df[x].cat.categories.tolist()
        missing = [c for c in current_x_cats if c not in x_order]
        if missing:
            raise ValueError(
                f"x_order is missing categories {missing} that are present in the data."
            )
        plot_df[x] = pd.Categorical(plot_df[x], categories=x_order, ordered=False)
        plot_df = plot_df[plot_df[x].notna()].copy()

    if hue is not None and hue_order is not None:
        current_hue_cats = plot_df[hue].cat.categories.tolist()
        missing_hue = [c for c in current_hue_cats if c not in hue_order]
        if missing_hue:
            raise ValueError(
                f"hue_order is missing categories {missing_hue} that are present in the data."
            )
        plot_df[hue] = pd.Categorical(plot_df[hue], categories=hue_order, ordered=False)
        plot_df = plot_df[plot_df[hue].notna()].copy()

    hue_cats = plot_df[hue].cat.categories.tolist() if hue else plot_df[x].cat.categories.tolist()
    color_map = _resolve_palette(palette, hue_cats)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # When there is no hue, use x as hue with legend=False to avoid the
    # seaborn FutureWarning: "Passing palette without assigning hue".
    _box_hue = hue if hue is not None else (x if orient == "v" else "_value")
    _box_legend: Union[bool, str] = "auto" if hue is not None else False

    box_kwargs = dict(
        data=plot_df,
        x=x if orient == "v" else "_value",
        y="_value" if orient == "v" else x,
        hue=_box_hue,
        palette=color_map,
        ax=ax,
        legend=_box_legend,
        showfliers=False,
        linewidth=1.5,
    )
    box_kwargs.update(kwargs)
    sns.boxplot(**box_kwargs)

    if show_points:
        strip_kwargs = dict(
            data=plot_df,
            x=x if orient == "v" else "_value",
            y="_value" if orient == "v" else x,
            hue=_box_hue,
            palette=color_map,
            ax=ax,
            size=point_size,
            alpha=point_alpha,
            dodge=hue is not None,
            legend=False,
            linewidth=0,
        )
        sns.stripplot(**strip_kwargs)

    # ------------------------------------------------------------------
    # Statistical annotations
    # ------------------------------------------------------------------
    if show_stats and hue:
        _annotate_pvalues(
            ax=ax,
            data=plot_df,
            x=x,
            hue=hue,
            value_col="_value",
            comparisons=comparisons,
            orient=orient,
            test=stat_test,
        )

    # ------------------------------------------------------------------
    # Labels & legend
    # ------------------------------------------------------------------
    ax.set_title(title if title is not None else feature)
    if orient == "v":
        ax.set_xlabel(xlabel if xlabel is not None else x)
        ax.set_ylabel(ylabel if ylabel is not None else feature)
    else:
        ax.set_ylabel(xlabel if xlabel is not None else x)
        ax.set_xlabel(ylabel if ylabel is not None else feature)

    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        # Keep only hue legend entries (seaborn may duplicate if strip visible)
        handles = handles[:len(hue_cats)]
        labels = labels[:len(hue_cats)]
        if legend_loc == "outside right":
            ax.legend(handles, labels, title=hue,
                      bbox_to_anchor=(1.01, 1), loc="upper left",
                      borderaxespad=0, frameon=True)
        elif legend_loc == "outside top":
            ax.legend(handles, labels, title=hue,
                      bbox_to_anchor=(0.5, 1.01), loc="lower center",
                      ncol=len(hue_cats), borderaxespad=0, frameon=True)
        else:
            ax.legend(handles, labels, title=hue, loc=legend_loc)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.tight_layout()
    return fig


def plot_feature_boxplot_multiplot(
    adata: AnnData,
    feature: str,
    x: str,
    groups: Optional[List[str]] = None,
    groups_hue: Optional[List[str]] = None,
    x_order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    ncols: int = 3,
    shared_colorscale: bool = True,
    hue: Optional[str] = None,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hspace: float = 0.5,
    wspace: float = 0.3,
    orient: Literal["v", "h"] = "v",
    vmin: Optional[Union[str, float]] = None,
    vmax: Optional[Union[str, float]] = None,
    show_points: bool = False,
    point_size: float = 2.0,
    point_alpha: float = 0.4,
    border_ticks_only: bool = True,
    xtick_rotation: int = 90,
    comparisons: Optional[List[Tuple[str, str]]] = None,
    stat_test: Literal["mann-whitney", "t-test"] = "mann-whitney",
    show_stats: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> Figure:
    """Grid of boxplots — one panel per ``x`` category value.

    Splits the ``x`` grouping axis into individual panels so that each panel
    shows the distribution of *feature* for a single *x* category, optionally
    broken down further by *hue*.  This is useful for comparing within-group
    distributions side by side while keeping the overall figure compact.

    The value axis (y-axis for ``orient="v"``, x-axis for ``orient="h"``) is
    shared across all panels by default (``shared_colorscale=True``), enabling
    direct visual comparison.  When ``shared_colorscale=False``, each panel
    autoscales independently; if *vmin* / *vmax* percentile strings are
    supplied in that mode, each panel resolves them from its own data, so
    scales will differ between panels.

    Args:
        adata: Annotated data matrix.
        feature: Gene name or ``adata.obs`` column to visualise. Resolved
            against ``adata.obs.columns`` first, then ``adata.var_names`` (or
            ``adata.var[gene_symbols]`` when supplied).
        x: Column in ``adata.obs`` whose unique values each become a panel.
        groups: Subset of *x* category values to use as panels. When
            ``None``, all categories are plotted in their original order.
            Defaults to ``None``.
        groups_hue: Subset of *hue* category values to include within each
            panel. Requires *hue* to be set. When ``None``, all hue categories
            are included. Defaults to ``None``.
        x_order: Explicit display order for the panel sequence (i.e. the
            order in which panels appear in the grid). Must cover all values
            that will be plotted (after applying *groups*). When ``None``,
            the original categorical order is used. Defaults to ``None``.
        hue_order: Explicit display order for *hue* categories within each
            panel. Must cover all hue values that will be plotted (after
            applying *groups_hue*). When ``None``, the original categorical
            order is used. Defaults to ``None``.
        ncols: Number of columns in the panel grid. Defaults to ``3``.
        shared_colorscale: When ``True``, the value axis is set to the same
            range across all panels, enabling direct visual comparison.
            When ``False``, each panel autoscales (or uses its own
            percentile-resolved *vmin* / *vmax*).  Defaults to ``True``.
        hue: Column in ``adata.obs`` for secondary (colour) grouping within
            each panel.  When ``None``, a single box is drawn per panel
            coloured by the panel's *x* group.  Defaults to ``None``.
        layer: Expression layer to use for gene features. Ignored for obs
            columns. Defaults to ``None`` (uses ``adata.X``).
        gene_symbols: Column in ``adata.var`` holding alternative gene
            identifiers. Ignored for obs features. Defaults to ``None``.
        palette: Colour palette. When *hue* is provided, colours *hue*
            categories; otherwise colours each panel by its *x* group.
            Accepts a single colour string, a list of colours, a dict mapping
            category labels to colours, or a seaborn/matplotlib palette name.
            Defaults to ``None`` (auto-assigned distinct colours per group).
        figsize: Size of a **single** panel ``(width, height)`` in inches.
            The total figure size is ``(ncols × width, nrows × height)``.
            When ``None``, panel size is derived from the number of *hue*
            categories: ``(max(n_hue × 0.8, 4.0), 4.0)`` or ``(3.0, 4.0)``
            when *hue* is ``None``.  Defaults to ``None``.
        hspace: Vertical space between subplot rows, as a fraction of the
            average axes height. Defaults to ``0.5``.
        wspace: Horizontal space between subplot columns, as a fraction of
            the average axes width. Defaults to ``0.3``.
        orient: ``"v"`` for vertical boxplots (feature on y-axis) or ``"h"``
            for horizontal (feature on x-axis). Defaults to ``"v"``.
        vmin: Lower limit for the value axis. Accepts a plain ``float`` or a
            percentile string (e.g. ``"p5"``).  With ``shared_colorscale=True``
            the percentile is resolved from all displayed groups combined;
            with ``shared_colorscale=False`` it is resolved per panel.
            Defaults to ``None``.
        vmax: Upper limit for the value axis. Same semantics as *vmin*.
            Defaults to ``None``.
        show_points: Overlay individual data points as a strip plot on top of
            each box. Defaults to ``False``.
        point_size: Marker size for the strip plot. Defaults to ``2.0``.
        point_alpha: Opacity of strip-plot points. Defaults to ``0.4``.
        border_ticks_only: When ``True``, x-axis tick labels and the x-axis
            label are shown only on the bottom row of panels, reducing clutter
            in multi-row grids.  Set to ``False`` to display them on every
            panel.  Defaults to ``True``.
        xtick_rotation: Rotation angle in degrees applied to x-axis tick
            labels on panels where they are displayed.  Defaults to ``90``.
        comparisons: List of ``(hue_cat_a, hue_cat_b)`` pairs to annotate
            with significance brackets within each panel. Requires *hue* and
            *show_stats* to be set. ``None`` tests all pairwise *hue*
            combinations. Defaults to ``None``.
        stat_test: Statistical test for pairwise comparisons. Either
            ``"mann-whitney"`` or ``"t-test"``. Defaults to ``"mann-whitney"``.
        show_stats: Whether to draw significance brackets. Requires *hue*.
            Defaults to ``False``.
        title: Overall figure super-title. Defaults to ``None``.
        xlabel: Override for the categorical axis label (x-axis for
            ``orient="v"``, y-axis for ``orient="h"``). When ``None``,
            defaults to the *hue* column name (or *x* when *hue* is ``None``).
        ylabel: Override for the value axis label (y-axis for ``orient="v"``,
            x-axis for ``orient="h"``). When ``None``, defaults to *feature*.
        **kwargs: Additional keyword arguments forwarded to
            ``seaborn.boxplot``.

    Returns:
        The matplotlib ``Figure`` object.

    Raises:
        ValueError: If *x* or *hue* is not found in ``adata.obs.columns``.
        ValueError: If *feature* cannot be resolved to an obs column or gene.
        ValueError: If *show_stats* is ``True`` but *hue* is not provided.
        ValueError: If any value in *groups* is not a category of *x*.
        ValueError: If any value in *groups_hue* is not a category of *hue*.
        ValueError: If *x_order* / *hue_order* does not cover all plotted
            categories.
        ValueError: If *vmin* or *vmax* is a string not starting with ``"p"``.

    Example:
        >>> fig = plot_feature_boxplot_multiplot(
        ...     adata,
        ...     feature="CD3E",
        ...     x="cell_type",
        ...     hue="condition",
        ...     ncols=2,
        ...     shared_colorscale=True,
        ...     show_stats=True,
        ...     comparisons=[("ctrl", "stim")],
        ... )
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if x not in adata.obs.columns:
        raise ValueError(
            f"x='{x}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if hue is not None and hue not in adata.obs.columns:
        raise ValueError(
            f"hue='{hue}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if show_stats and hue is None:
        raise ValueError("show_stats=True requires a hue column to be specified.")

    # ------------------------------------------------------------------
    # Extract feature and build long-form DataFrame
    # ------------------------------------------------------------------
    values, _ = _resolve_feature(adata, feature, layer, gene_symbols)
    values = values.astype(float)

    cols = [x] + ([hue] if hue else [])
    plot_df = adata.obs[cols].copy()
    plot_df["_value"] = values.values

    # Ensure stable categorical ordering
    if not hasattr(plot_df[x], "cat"):
        plot_df[x] = pd.Categorical(plot_df[x])
    if hue is not None and not hasattr(plot_df[hue], "cat"):
        plot_df[hue] = pd.Categorical(plot_df[hue])

    # ------------------------------------------------------------------
    # Resolve groups (subset + order of x categories → panel sequence)
    # ------------------------------------------------------------------
    all_x_cats: List[str] = plot_df[x].cat.categories.tolist()
    if groups is None:
        groups_to_plot: List[str] = all_x_cats
    else:
        invalid = [g for g in groups if g not in all_x_cats]
        if invalid:
            raise ValueError(
                f"Groups {invalid} not found in x='{x}' categories. "
                f"Available: {all_x_cats}"
            )
        groups_to_plot = list(groups)

    # x_order controls the sequence of panels
    if x_order is not None:
        missing_x = [c for c in groups_to_plot if c not in x_order]
        if missing_x:
            raise ValueError(
                f"x_order is missing panel groups {missing_x} that are present in the data."
            )
        groups_to_plot = [c for c in x_order if c in groups_to_plot]

    # Apply groups_hue subsetting on the full DataFrame
    if hue is not None and groups_hue is not None:
        all_hue_cats_full: List[str] = plot_df[hue].cat.categories.tolist()
        invalid_hue = [g for g in groups_hue if g not in all_hue_cats_full]
        if invalid_hue:
            raise ValueError(
                f"groups_hue values {invalid_hue} not found in hue='{hue}' categories. "
                f"Available: {all_hue_cats_full}"
            )
        plot_df = plot_df[plot_df[hue].isin(groups_hue)].copy()
        plot_df[hue] = plot_df[hue].cat.remove_unused_categories()

    # Apply hue_order (reorder hue categories globally)
    if hue is not None and hue_order is not None:
        current_hue_cats = plot_df[hue].cat.categories.tolist()
        missing_hue = [c for c in current_hue_cats if c not in hue_order]
        if missing_hue:
            raise ValueError(
                f"hue_order is missing hue categories {missing_hue} that are present in the data."
            )
        plot_df[hue] = pd.Categorical(plot_df[hue], categories=hue_order, ordered=False)

    n_groups = len(groups_to_plot)
    nrows = int(np.ceil(n_groups / ncols))

    # ------------------------------------------------------------------
    # Palette setup
    # ------------------------------------------------------------------
    if hue is not None:
        hue_cats: List[str] = plot_df[hue].cat.categories.tolist()
        color_map = _resolve_palette(palette, hue_cats)
        if color_map is None:
            # Materialise a default palette keyed by ALL global hue categories
            # so colours stay consistent across panels even when some panels
            # are missing certain hue values.
            _default_colors = sns.color_palette(n_colors=len(hue_cats))
            color_map = {c: _default_colors[i] for i, c in enumerate(hue_cats)}
    else:
        # One colour per panel group; fall back to seaborn's categorical palette
        group_color_map = _resolve_palette(palette, groups_to_plot)
        if group_color_map is None:
            _default_colors = sns.color_palette(n_colors=len(groups_to_plot))
            group_color_map = {
                g: _default_colors[i] for i, g in enumerate(groups_to_plot)
            }

    # ------------------------------------------------------------------
    # Shared vmin / vmax pre-computation
    # ------------------------------------------------------------------
    if shared_colorscale:
        _mask = plot_df[x].isin(groups_to_plot)
        _all_values = plot_df.loc[_mask, "_value"]
        _shared_vmin = _resolve_vmin_vmax(_all_values, vmin)
        _shared_vmax = _resolve_vmin_vmax(_all_values, vmax)

    # ------------------------------------------------------------------
    # Figure size
    # ------------------------------------------------------------------
    if figsize is not None:
        panel_w, panel_h = float(figsize[0]), float(figsize[1])
    else:
        if hue is not None:
            panel_w = max(len(hue_cats) * 0.8, 4.0)
        else:
            panel_w = 3.0
        panel_h = 4.0

    total_w = ncols * panel_w
    total_h = nrows * panel_h

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(total_w, total_h),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.01)

    # ------------------------------------------------------------------
    # Draw panels
    # ------------------------------------------------------------------
    all_used_axes: list = []
    all_panel_dfs: list = []  # cached per-panel DataFrames for annotation second pass

    for idx, group in enumerate(groups_to_plot):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        all_used_axes.append(ax)

        panel_df = plot_df[plot_df[x] == group].copy()
        # Reset x categories to only the current group (needed by _annotate_pvalues)
        panel_df[x] = panel_df[x].cat.remove_unused_categories()
        # Drop absent hue categories so seaborn only renders observed ticks
        # and doesn't reserve dodge-slots for missing groups (which would make
        # boxes narrower than expected).
        if hue is not None:
            panel_df[hue] = panel_df[hue].cat.remove_unused_categories()
        all_panel_dfs.append(panel_df)

        if hue is not None:
            # Multiple boxes: one per hue value
            box_kwargs: dict = dict(
                data=panel_df,
                x=hue if orient == "v" else "_value",
                y="_value" if orient == "v" else hue,
                hue=hue,
                palette=color_map,
                ax=ax,
                legend=False,
                showfliers=False,
                linewidth=1.5,
                dodge=False,
            )
            box_kwargs.update(kwargs)
            sns.boxplot(**box_kwargs)
        else:
            # Single box coloured by the panel's group
            box_kwargs = dict(
                data=panel_df,
                x=x if orient == "v" else "_value",
                y="_value" if orient == "v" else x,
                hue=x,
                palette=group_color_map,
                ax=ax,
                legend=False,
                showfliers=False,
                linewidth=1.5,
            )
            box_kwargs.update(kwargs)
            sns.boxplot(**box_kwargs)

        # Optional strip plot
        if show_points:
            if hue is not None:
                strip_kwargs = dict(
                    data=panel_df,
                    x=hue if orient == "v" else "_value",
                    y="_value" if orient == "v" else hue,
                    hue=hue,
                    palette=color_map,
                    ax=ax,
                    size=point_size,
                    alpha=point_alpha,
                    dodge=False,
                    legend=False,
                    linewidth=0,
                )
            else:
                strip_kwargs = dict(
                    data=panel_df,
                    x=x if orient == "v" else "_value",
                    y="_value" if orient == "v" else x,
                    hue=x,
                    palette=group_color_map,
                    ax=ax,
                    size=point_size,
                    alpha=point_alpha,
                    dodge=False,
                    legend=False,
                    linewidth=0,
                )
            sns.stripplot(**strip_kwargs)

        # Per-panel axis limits when shared_colorscale=False
        if not shared_colorscale:
            _pv_min = _resolve_vmin_vmax(panel_df["_value"], vmin)
            _pv_max = _resolve_vmin_vmax(panel_df["_value"], vmax)
            if _pv_min is not None or _pv_max is not None:
                _cur = ax.get_ylim() if orient == "v" else ax.get_xlim()
                _lo = _pv_min if _pv_min is not None else _cur[0]
                _hi = _pv_max if _pv_max is not None else _cur[1]
                if orient == "v":
                    ax.set_ylim(_lo, _hi)
                else:
                    ax.set_xlim(_lo, _hi)

        # Title and axis labels
        ax.set_title(group)
        _cat_label = xlabel if xlabel is not None else (hue if hue else x)
        _val_label = ylabel if ylabel is not None else feature
        _is_bottom_row = (row == nrows - 1)
        if orient == "v":
            if border_ticks_only and not _is_bottom_row:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.tick_params(axis="x", labelrotation=xtick_rotation)
                ax.set_xlabel(_cat_label)
            ax.set_ylabel(_val_label)
        else:
            ax.set_ylabel(_cat_label)
            ax.tick_params(axis="x", labelrotation=xtick_rotation)
            ax.set_xlabel(_val_label)

    # ------------------------------------------------------------------
    # Hide unused panels
    # ------------------------------------------------------------------
    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # ------------------------------------------------------------------
    # Apply shared value-axis range
    # ------------------------------------------------------------------
    if shared_colorscale:
        if _shared_vmin is not None or _shared_vmax is not None:
            for ax in all_used_axes:
                _cur = ax.get_ylim() if orient == "v" else ax.get_xlim()
                _lo = _shared_vmin if _shared_vmin is not None else _cur[0]
                _hi = _shared_vmax if _shared_vmax is not None else _cur[1]
                if orient == "v":
                    ax.set_ylim(_lo, _hi)
                else:
                    ax.set_xlim(_lo, _hi)
        else:
            # Both None: broadcast the broadest auto-range across all panels
            _all_lims = [
                ax.get_ylim() if orient == "v" else ax.get_xlim()
                for ax in all_used_axes
            ]
            _lo = min(lim[0] for lim in _all_lims)
            _hi = max(lim[1] for lim in _all_lims)
            for ax in all_used_axes:
                if orient == "v":
                    ax.set_ylim(_lo, _hi)
                else:
                    ax.set_xlim(_lo, _hi)

    # ------------------------------------------------------------------
    # Statistical annotations — drawn after limits are finalised so that
    # bracket positions and axis expansions are consistent.
    # ------------------------------------------------------------------
    if show_stats and hue is not None:
        for ax, panel_df in zip(all_used_axes, all_panel_dfs):
            _annotate_pvalues_single_group(
                ax=ax,
                data=panel_df,
                hue=hue,
                value_col="_value",
                comparisons=comparisons,
                orient=orient,
                test=stat_test,
            )

    return fig


def plot_feature_boxplot_aggregated(
    adata: AnnData,
    feature: str,
    x: str,
    sample_col: str,
    hue: Optional[str] = None,
    groups_x: Optional[List[str]] = None,
    groups_hue: Optional[List[str]] = None,
    x_order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    agg_fn: Literal["mean", "median", "sum"] = "mean",
    min_cells: int = 10,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    figsize: Tuple[float, float] = (6.0, 4.0),
    orient: Literal["v", "h"] = "v",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_points: bool = False,
    point_size: float = 6.0,
    point_alpha: float = 0.8,
    legend_loc: Literal["outside right", "outside top", "best"] = "outside right",
    comparisons: Optional[List[Tuple[str, str]]] = None,
    stat_test: Literal["mann-whitney", "t-test"] = "mann-whitney",
    show_stats: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> Figure:
    """Boxplot of a per-sample aggregated feature, grouped by one or two ``adata.obs`` columns.

    Unlike :func:`plot_feature_boxplot`, each data point on this plot
    represents one **sample** (the unique value in *sample_col*), not one cell.
    Expression values (or obs values) are first aggregated per
    ``(sample, x[, hue])`` group using *agg_fn*, and samples with fewer than
    *min_cells* cells in a group are silently dropped before aggregation.

    This approach is appropriate for pseudo-bulk comparisons where
    distributional assumptions should hold at the sample level rather than the
    cell level.

    Args:
        adata: Annotated data matrix.
        feature: Gene name or ``adata.obs`` column to visualise. Resolved
            against ``adata.obs.columns`` first, then ``adata.var_names`` (or
            ``adata.var[gene_symbols]`` when supplied).
        x: Column in ``adata.obs`` to use as the primary grouping axis.
        sample_col: Column in ``adata.obs`` that identifies biological
            samples. One aggregate value is computed per unique sample.
        hue: Column in ``adata.obs`` for secondary (colour) grouping.
            Defaults to ``None``.
        groups_x: Subset of *x* category values to include. When ``None``,
            all categories are included. Defaults to ``None``.
        groups_hue: Subset of *hue* category values to include. Requires
            *hue* to be set. When ``None``, all hue categories are included.
            Defaults to ``None``.
        x_order: Explicit display order for *x* categories. Must contain
            every value that will be plotted (after applying *groups_x*).
            When ``None``, the original categorical order is used.
            Defaults to ``None``.
        hue_order: Explicit display order for *hue* categories. Must contain
            every value that will be plotted (after applying *groups_hue*).
            When ``None``, the original categorical order is used.
            Defaults to ``None``.
        agg_fn: Aggregation function to apply per ``(sample, x[, hue])``
            group. One of ``"mean"``, ``"median"``, or ``"sum"``.
            Defaults to ``"mean"``.
        min_cells: Minimum number of cells a sample must contribute to a
            ``(sample, x[, hue])`` group to be included. Groups with fewer
            cells are dropped. Defaults to ``10``.
        layer: Expression layer to use for gene features. Ignored for obs
            columns. Defaults to ``None`` (uses ``adata.X``).
        gene_symbols: Column in ``adata.var`` holding alternative gene
            identifiers. Ignored for obs features. Defaults to ``None``.
        palette: Colour palette for *hue* (or *x* when *hue* is ``None``).
            Accepts a single colour string, a list of colours, a dict mapping
            category labels to colours, or a seaborn/matplotlib palette name.
            Defaults to ``None`` (seaborn default).
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``(6.0, 4.0)``.
        orient: ``"v"`` for vertical boxplots (feature on y-axis) or ``"h"``
            for horizontal (feature on x-axis). Defaults to ``"v"``.
        vmin: Lower clip limit applied **after** aggregation. Defaults to
            ``None``.
        vmax: Upper clip limit applied **after** aggregation. Defaults to
            ``None``.
        show_points: Overlay each sample's aggregate value as an individual
            point. Defaults to ``False``.
        point_size: Marker size for sample points. Defaults to ``6.0``.
        point_alpha: Opacity of sample points. Defaults to ``0.8``.
        legend_loc: Legend placement. ``"outside right"`` places the legend
            outside the axes on the right (default); ``"outside top"`` places
            it above the axes; ``"best"`` lets matplotlib choose inside the
            axes.
        comparisons: List of ``(hue_cat_a, hue_cat_b)`` pairs to annotate
            with p-value brackets. Requires *hue* and *show_stats* to be set.
            If ``None`` and *show_stats* is ``True``, all pairwise hue
            combinations are tested. Defaults to ``None``.
        stat_test: Statistical test for pairwise comparisons. Either
            ``"mann-whitney"`` or ``"t-test"``. Defaults to ``"mann-whitney"``.
        show_stats: Whether to draw significance brackets. Requires *hue*.
            Defaults to ``False``.
        title: Axes title. Defaults to ``"{feature} (per sample)"``.
        xlabel: x-axis label. Defaults to *x*.
        ylabel: y-axis label. Defaults to *feature*.
        **kwargs: Additional keyword arguments forwarded to
            ``seaborn.boxplot``.

    Returns:
        The matplotlib ``Figure`` object.

    Raises:
        ValueError: If *x*, *hue*, or *sample_col* is not found in
            ``adata.obs.columns``.
        ValueError: If *feature* cannot be resolved to an obs column or gene.
        ValueError: If *show_stats* is ``True`` but *hue* is not provided.
        ValueError: If any value in *groups_x* / *groups_hue* is not a valid
            category of the respective column.
        ValueError: If *x_order* / *hue_order* does not cover all plotted
            categories.
        ValueError: If *agg_fn* is not one of ``"mean"``, ``"median"``,
            ``"sum"``.
        ValueError: If no samples remain after applying *min_cells* filter.

    Example:
        >>> fig = plot_feature_boxplot_aggregated(
        ...     adata,
        ...     feature="CD3E",
        ...     x="cell_type",
        ...     sample_col="donor_id",
        ...     hue="condition",
        ...     min_cells=20,
        ...     agg_fn="mean",
        ...     show_stats=True,
        ...     comparisons=[("ctrl", "stim")],
        ... )
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if x not in adata.obs.columns:
        raise ValueError(
            f"x='{x}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if sample_col not in adata.obs.columns:
        raise ValueError(
            f"sample_col='{sample_col}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if hue is not None and hue not in adata.obs.columns:
        raise ValueError(
            f"hue='{hue}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if show_stats and hue is None:
        raise ValueError("show_stats=True requires a hue column to be specified.")
    if agg_fn not in ("mean", "median", "sum"):
        raise ValueError(
            f"agg_fn='{agg_fn}' is not supported. Choose from 'mean', 'median', 'sum'."
        )

    # ------------------------------------------------------------------
    # Build long-form DataFrame with raw (cell-level) values
    # ------------------------------------------------------------------
    values, _ = _resolve_feature(adata, feature, layer, gene_symbols)
    values = values.astype(float)

    _all_group_cols = [sample_col, x] + ([hue] if hue else [])
    plot_df = adata.obs[_all_group_cols].copy()
    plot_df["_value"] = values.values

    # ------------------------------------------------------------------
    # Apply groups_x / groups_hue subsetting and x_order / hue_order
    # ------------------------------------------------------------------
    if not hasattr(plot_df[x], "cat"):
        plot_df[x] = pd.Categorical(plot_df[x])
    if hue is not None and not hasattr(plot_df[hue], "cat"):
        plot_df[hue] = pd.Categorical(plot_df[hue])

    if groups_x is not None:
        all_x_cats_agg: List[str] = plot_df[x].cat.categories.tolist()
        invalid_x = [g for g in groups_x if g not in all_x_cats_agg]
        if invalid_x:
            raise ValueError(
                f"groups_x values {invalid_x} not found in x='{x}' categories. "
                f"Available: {all_x_cats_agg}"
            )
        plot_df = plot_df[plot_df[x].isin(groups_x)].copy()
        plot_df[x] = plot_df[x].cat.remove_unused_categories()

    if hue is not None and groups_hue is not None:
        all_hue_cats_agg: List[str] = plot_df[hue].cat.categories.tolist()
        invalid_hue = [g for g in groups_hue if g not in all_hue_cats_agg]
        if invalid_hue:
            raise ValueError(
                f"groups_hue values {invalid_hue} not found in hue='{hue}' categories. "
                f"Available: {all_hue_cats_agg}"
            )
        plot_df = plot_df[plot_df[hue].isin(groups_hue)].copy()
        plot_df[hue] = plot_df[hue].cat.remove_unused_categories()

    if x_order is not None:
        current_x_cats_agg = plot_df[x].cat.categories.tolist()
        missing_x = [c for c in current_x_cats_agg if c not in x_order]
        if missing_x:
            raise ValueError(
                f"x_order is missing categories {missing_x} that are present in the data."
            )
        plot_df[x] = pd.Categorical(plot_df[x], categories=x_order, ordered=False)
        plot_df = plot_df[plot_df[x].notna()].copy()

    if hue is not None and hue_order is not None:
        current_hue_cats_agg = plot_df[hue].cat.categories.tolist()
        missing_hue = [c for c in current_hue_cats_agg if c not in hue_order]
        if missing_hue:
            raise ValueError(
                f"hue_order is missing categories {missing_hue} that are present in the data."
            )
        plot_df[hue] = pd.Categorical(plot_df[hue], categories=hue_order, ordered=False)
        plot_df = plot_df[plot_df[hue].notna()].copy()

    # Rebuild group_cols list from (potentially filtered) plot_df
    group_cols = [sample_col, x] + ([hue] if hue else [])

    # ------------------------------------------------------------------
    # Aggregate per (sample, x[, hue]) — drop groups below min_cells
    # ------------------------------------------------------------------
    grouped = plot_df.groupby(group_cols, observed=True)
    cell_counts = grouped["_value"].count()
    agg_values = getattr(grouped["_value"], agg_fn)()

    # Apply min_cells filter
    agg_df = agg_values[cell_counts >= min_cells].reset_index()

    if agg_df.empty:
        raise ValueError(
            f"No samples remain after applying min_cells={min_cells}. "
            "Lower the threshold or check your grouping columns."
        )

    # Clip after aggregation
    if vmin is not None:
        agg_df["_value"] = agg_df["_value"].clip(lower=vmin)
    if vmax is not None:
        agg_df["_value"] = agg_df["_value"].clip(upper=vmax)

    # Restore categorical dtype for stable plot ordering — use the
    # (possibly filtered / reordered) categories from plot_df, not adata.obs.
    _final_x_cats = plot_df[x].cat.categories.tolist()
    agg_df[x] = pd.Categorical(agg_df[x], categories=_final_x_cats)
    if hue is not None:
        _final_hue_cats = plot_df[hue].cat.categories.tolist()
        agg_df[hue] = pd.Categorical(agg_df[hue], categories=_final_hue_cats)

    hue_cats = agg_df[hue].cat.categories.tolist() if hue else agg_df[x].cat.categories.tolist()
    color_map = _resolve_palette(palette, hue_cats)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # When there is no hue, use x as hue with legend=False to avoid the
    # seaborn FutureWarning: "Passing palette without assigning hue".
    _box_hue = hue if hue is not None else (x if orient == "v" else "_value")
    _box_legend: Union[bool, str] = "auto" if hue is not None else False

    box_kwargs = dict(
        data=agg_df,
        x=x if orient == "v" else "_value",
        y="_value" if orient == "v" else x,
        hue=_box_hue,
        palette=color_map,
        ax=ax,
        legend=_box_legend,
        showfliers=False,
        linewidth=1.5,
    )
    box_kwargs.update(kwargs)
    sns.boxplot(**box_kwargs)

    if show_points:
        strip_kwargs = dict(
            data=agg_df,
            x=x if orient == "v" else "_value",
            y="_value" if orient == "v" else x,
            hue=_box_hue,
            palette=color_map,
            ax=ax,
            size=point_size,
            alpha=point_alpha,
            dodge=hue is not None,
            legend=False,
            linewidth=0.5,
            edgecolor="white",
        )
        sns.stripplot(**strip_kwargs)

    # ------------------------------------------------------------------
    # Statistical annotations
    # ------------------------------------------------------------------
    if show_stats and hue:
        _annotate_pvalues(
            ax=ax,
            data=agg_df,
            x=x,
            hue=hue,
            value_col="_value",
            comparisons=comparisons,
            orient=orient,
            test=stat_test,
        )

    # ------------------------------------------------------------------
    # Labels & legend
    # ------------------------------------------------------------------
    default_title = f"{feature} (per sample, {agg_fn})"
    ax.set_title(title if title is not None else default_title)
    if orient == "v":
        ax.set_xlabel(xlabel if xlabel is not None else x)
        ax.set_ylabel(ylabel if ylabel is not None else feature)
    else:
        ax.set_ylabel(xlabel if xlabel is not None else x)
        ax.set_xlabel(ylabel if ylabel is not None else feature)

    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:len(hue_cats)]
        labels = labels[:len(hue_cats)]
        if legend_loc == "outside right":
            ax.legend(handles, labels, title=hue,
                      bbox_to_anchor=(1.01, 1), loc="upper left",
                      borderaxespad=0, frameon=True)
        elif legend_loc == "outside top":
            ax.legend(handles, labels, title=hue,
                      bbox_to_anchor=(0.5, 1.01), loc="lower center",
                      ncol=len(hue_cats), borderaxespad=0, frameon=True)
        else:
            ax.legend(handles, labels, title=hue, loc=legend_loc)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.tight_layout()
    return fig


def plot_feature_boxplot_aggregated_multiplot(
    adata: AnnData,
    feature: str,
    x: str,
    sample_col: str,
    hue: Optional[str] = None,
    groups: Optional[List[str]] = None,
    groups_hue: Optional[List[str]] = None,
    x_order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    agg_fn: Literal["mean", "median", "sum"] = "mean",
    min_cells: int = 10,
    ncols: int = 3,
    shared_colorscale: bool = True,
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hspace: float = 0.5,
    wspace: float = 0.3,
    orient: Literal["v", "h"] = "v",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_points: bool = True,
    point_size: float = 6.0,
    point_alpha: float = 0.8,
    border_ticks_only: bool = True,
    xtick_rotation: int = 90,
    comparisons: Optional[List[Tuple[str, str]]] = None,
    stat_test: Literal["mann-whitney", "t-test"] = "mann-whitney",
    show_stats: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> Figure:
    """Grid of pseudo-bulk boxplots — one panel per ``x`` category value.

    Combines the per-sample aggregation of :func:`plot_feature_boxplot_aggregated`
    with the multi-panel layout of :func:`plot_feature_boxplot_multiplot`.
    Each panel shows the distribution of per-sample aggregated values of
    *feature* for a single *x* category, optionally broken down further by
    *hue*.

    The value axis is shared across all panels by default
    (``shared_colorscale=True``), enabling direct visual comparison. Data
    points (one per sample) are shown by default (``show_points=True``) since
    pseudo-bulk data typically has few observations per group.

    Args:
        adata: Annotated data matrix.
        feature: Gene name or ``adata.obs`` column to visualise. Resolved
            against ``adata.obs.columns`` first, then ``adata.var_names`` (or
            ``adata.var[gene_symbols]`` when supplied).
        x: Column in ``adata.obs`` whose unique values each become a panel.
        sample_col: Column in ``adata.obs`` that identifies biological
            samples. One aggregate value is computed per unique sample.
        hue: Column in ``adata.obs`` for secondary (colour) grouping within
            each panel. When ``None``, a single box is drawn per panel
            coloured by the panel's *x* group. Defaults to ``None``.
        groups: Subset of *x* category values to use as panels. When
            ``None``, all categories are plotted in their original order.
            Defaults to ``None``.
        groups_hue: Subset of *hue* category values to include within each
            panel. Requires *hue* to be set. When ``None``, all hue
            categories are included. Defaults to ``None``.
        x_order: Explicit display order for the panel sequence. Must cover
            all values that will be plotted (after applying *groups*). When
            ``None``, the original categorical order is used.
            Defaults to ``None``.
        hue_order: Explicit display order for *hue* categories within each
            panel. Must cover all hue values that will be plotted (after
            applying *groups_hue*). When ``None``, the original categorical
            order is used. Defaults to ``None``.
        agg_fn: Aggregation function to apply per ``(sample, x[, hue])``
            group. One of ``"mean"``, ``"median"``, or ``"sum"``.
            Defaults to ``"mean"``.
        min_cells: Minimum number of cells a sample must contribute to a
            ``(sample, x[, hue])`` group to be included. Groups with fewer
            cells are dropped. Defaults to ``10``.
        ncols: Number of columns in the panel grid. Defaults to ``3``.
        shared_colorscale: When ``True``, the value axis is set to the same
            range across all panels. When ``False``, each panel autoscales.
            Defaults to ``True``.
        layer: Expression layer to use for gene features. Ignored for obs
            columns. Defaults to ``None`` (uses ``adata.X``).
        gene_symbols: Column in ``adata.var`` holding alternative gene
            identifiers. Ignored for obs features. Defaults to ``None``.
        palette: Colour palette. When *hue* is provided, colours *hue*
            categories; otherwise colours each panel by its *x* group.
            Accepts a single colour string, a list of colours, a dict mapping
            category labels to colours, or a seaborn/matplotlib palette name.
            Defaults to ``None`` (auto-assigned distinct colours per group).
        figsize: Size of a **single** panel ``(width, height)`` in inches.
            The total figure size is ``(ncols × width, nrows × height)``.
            When ``None``, panel size is derived from the number of *hue*
            categories: ``(max(n_hue × 0.8, 4.0), 4.0)`` or ``(3.0, 4.0)``
            when *hue* is ``None``. Defaults to ``None``.
        hspace: Vertical space between subplot rows, as a fraction of the
            average axes height. Defaults to ``0.5``.
        wspace: Horizontal space between subplot columns, as a fraction of
            the average axes width. Defaults to ``0.3``.
        orient: ``"v"`` for vertical boxplots (feature on y-axis) or ``"h"``
            for horizontal (feature on x-axis). Defaults to ``"v"``.
        vmin: Lower limit for the value axis. Accepts a plain ``float``.
            With ``shared_colorscale=True`` the limit is applied globally.
            Defaults to ``None``.
        vmax: Upper limit for the value axis. Same semantics as *vmin*.
            Defaults to ``None``.
        show_points: Overlay each sample's aggregate value as an individual
            point. Defaults to ``True``.
        point_size: Marker size for sample points. Defaults to ``6.0``.
        point_alpha: Opacity of sample points. Defaults to ``0.8``.
        border_ticks_only: When ``True``, x-axis tick labels and the x-axis
            label are shown only on the bottom row of panels.
            Defaults to ``True``.
        xtick_rotation: Rotation angle in degrees applied to x-axis tick
            labels. Defaults to ``90``.
        comparisons: List of ``(hue_cat_a, hue_cat_b)`` pairs to annotate
            with significance brackets within each panel. Requires *hue* and
            *show_stats* to be set. ``None`` tests all pairwise *hue*
            combinations. Defaults to ``None``.
        stat_test: Statistical test for pairwise comparisons. Either
            ``"mann-whitney"`` or ``"t-test"``. Defaults to ``"mann-whitney"``.
        show_stats: Whether to draw significance brackets. Requires *hue*.
            Defaults to ``False``.
        title: Overall figure super-title. Defaults to ``None``.
        xlabel: Override for the categorical axis label. When ``None``,
            defaults to the *hue* column name (or *x* when *hue* is
            ``None``). Defaults to ``None``.
        ylabel: Override for the value axis label. When ``None``, defaults
            to *feature*. Defaults to ``None``.
        **kwargs: Additional keyword arguments forwarded to
            ``seaborn.boxplot``.

    Returns:
        The matplotlib ``Figure`` object.

    Raises:
        ValueError: If *x*, *hue*, or *sample_col* is not found in
            ``adata.obs.columns``.
        ValueError: If *feature* cannot be resolved to an obs column or gene.
        ValueError: If *show_stats* is ``True`` but *hue* is not provided.
        ValueError: If any value in *groups* / *groups_hue* is not a valid
            category of the respective column.
        ValueError: If *x_order* / *hue_order* does not cover all plotted
            categories.
        ValueError: If *agg_fn* is not one of ``"mean"``, ``"median"``,
            ``"sum"``.
        ValueError: If no samples remain after applying *min_cells* filter.

    Example:
        >>> fig = plot_feature_boxplot_aggregated_multiplot(
        ...     adata,
        ...     feature="CD3E",
        ...     x="cell_type",
        ...     sample_col="donor_id",
        ...     hue="condition",
        ...     ncols=2,
        ...     show_stats=True,
        ...     comparisons=[("ctrl", "stim")],
        ... )
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if x not in adata.obs.columns:
        raise ValueError(
            f"x='{x}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if sample_col not in adata.obs.columns:
        raise ValueError(
            f"sample_col='{sample_col}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if hue is not None and hue not in adata.obs.columns:
        raise ValueError(
            f"hue='{hue}' not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if show_stats and hue is None:
        raise ValueError("show_stats=True requires a hue column to be specified.")
    if agg_fn not in ("mean", "median", "sum"):
        raise ValueError(
            f"agg_fn='{agg_fn}' is not supported. Choose from 'mean', 'median', 'sum'."
        )

    # ------------------------------------------------------------------
    # Build long-form DataFrame with raw (cell-level) values
    # ------------------------------------------------------------------
    values, _ = _resolve_feature(adata, feature, layer, gene_symbols)
    values = values.astype(float)

    _all_group_cols = [sample_col, x] + ([hue] if hue else [])
    plot_df = adata.obs[_all_group_cols].copy()
    plot_df["_value"] = values.values

    # Ensure stable categorical ordering
    if not hasattr(plot_df[x], "cat"):
        plot_df[x] = pd.Categorical(plot_df[x])
    if hue is not None and not hasattr(plot_df[hue], "cat"):
        plot_df[hue] = pd.Categorical(plot_df[hue])

    # ------------------------------------------------------------------
    # Resolve groups (subset + order of x categories → panel sequence)
    # ------------------------------------------------------------------
    all_x_cats: List[str] = plot_df[x].cat.categories.tolist()
    if groups is None:
        groups_to_plot: List[str] = all_x_cats
    else:
        invalid = [g for g in groups if g not in all_x_cats]
        if invalid:
            raise ValueError(
                f"Groups {invalid} not found in x='{x}' categories. "
                f"Available: {all_x_cats}"
            )
        groups_to_plot = list(groups)

    if x_order is not None:
        missing_x = [c for c in groups_to_plot if c not in x_order]
        if missing_x:
            raise ValueError(
                f"x_order is missing panel groups {missing_x} that are present in the data."
            )
        groups_to_plot = [c for c in x_order if c in groups_to_plot]

    # Apply groups_hue subsetting on the full DataFrame
    if hue is not None and groups_hue is not None:
        all_hue_cats_full: List[str] = plot_df[hue].cat.categories.tolist()
        invalid_hue = [g for g in groups_hue if g not in all_hue_cats_full]
        if invalid_hue:
            raise ValueError(
                f"groups_hue values {invalid_hue} not found in hue='{hue}' categories. "
                f"Available: {all_hue_cats_full}"
            )
        plot_df = plot_df[plot_df[hue].isin(groups_hue)].copy()
        plot_df[hue] = plot_df[hue].cat.remove_unused_categories()

    # Apply hue_order
    if hue is not None and hue_order is not None:
        current_hue_cats = plot_df[hue].cat.categories.tolist()
        missing_hue = [c for c in current_hue_cats if c not in hue_order]
        if missing_hue:
            raise ValueError(
                f"hue_order is missing hue categories {missing_hue} that are present in the data."
            )
        plot_df[hue] = pd.Categorical(plot_df[hue], categories=hue_order, ordered=False)

    # ------------------------------------------------------------------
    # Aggregate per (sample, x[, hue]) — drop groups below min_cells
    # ------------------------------------------------------------------
    group_cols = [sample_col, x] + ([hue] if hue else [])
    grouped = plot_df.groupby(group_cols, observed=True)
    cell_counts = grouped["_value"].count()
    agg_values_series = getattr(grouped["_value"], agg_fn)()

    agg_df = agg_values_series[cell_counts >= min_cells].reset_index()

    if agg_df.empty:
        raise ValueError(
            f"No samples remain after applying min_cells={min_cells}. "
            "Lower the threshold or check your grouping columns."
        )

    # Clip after aggregation
    if vmin is not None:
        agg_df["_value"] = agg_df["_value"].clip(lower=vmin)
    if vmax is not None:
        agg_df["_value"] = agg_df["_value"].clip(upper=vmax)

    # Restore categoricals using the filtered/ordered categories
    _final_x_cats = plot_df[x].cat.categories.tolist()
    agg_df[x] = pd.Categorical(agg_df[x], categories=_final_x_cats)
    if hue is not None:
        _final_hue_cats = plot_df[hue].cat.categories.tolist()
        agg_df[hue] = pd.Categorical(agg_df[hue], categories=_final_hue_cats)

    n_groups = len(groups_to_plot)
    nrows = int(np.ceil(n_groups / ncols))

    # ------------------------------------------------------------------
    # Palette setup
    # ------------------------------------------------------------------
    if hue is not None:
        hue_cats: List[str] = agg_df[hue].cat.categories.tolist()
        color_map = _resolve_palette(palette, hue_cats)
        if color_map is None:
            # Materialise a default palette keyed by ALL global hue categories
            # so colours stay consistent across panels even when some panels
            # are missing certain hue values.
            _default_colors = sns.color_palette(n_colors=len(hue_cats))
            color_map = {c: _default_colors[i] for i, c in enumerate(hue_cats)}
    else:
        group_color_map = _resolve_palette(palette, groups_to_plot)
        if group_color_map is None:
            _default_colors = sns.color_palette(n_colors=len(groups_to_plot))
            group_color_map = {
                g: _default_colors[i] for i, g in enumerate(groups_to_plot)
            }

    # ------------------------------------------------------------------
    # Shared vmin / vmax pre-computation
    # ------------------------------------------------------------------
    if shared_colorscale:
        _mask = agg_df[x].isin(groups_to_plot)
        _all_values = agg_df.loc[_mask, "_value"]
        _shared_vmin = _resolve_vmin_vmax(_all_values, vmin)
        _shared_vmax = _resolve_vmin_vmax(_all_values, vmax)

    # ------------------------------------------------------------------
    # Figure size
    # ------------------------------------------------------------------
    if figsize is not None:
        panel_w, panel_h = float(figsize[0]), float(figsize[1])
    else:
        if hue is not None:
            panel_w = max(len(hue_cats) * 0.8, 4.0)
        else:
            panel_w = 3.0
        panel_h = 4.0

    total_w = ncols * panel_w
    total_h = nrows * panel_h

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(total_w, total_h),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.01)

    # ------------------------------------------------------------------
    # Draw panels
    # ------------------------------------------------------------------
    all_used_axes: list = []
    all_panel_dfs: list = []

    for idx, group in enumerate(groups_to_plot):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        all_used_axes.append(ax)

        panel_df = agg_df[agg_df[x] == group].copy()
        panel_df[x] = panel_df[x].cat.remove_unused_categories()
        if hue is not None:
            panel_df[hue] = panel_df[hue].cat.remove_unused_categories()
        all_panel_dfs.append(panel_df)

        if hue is not None:
            box_kwargs: dict = dict(
                data=panel_df,
                x=hue if orient == "v" else "_value",
                y="_value" if orient == "v" else hue,
                hue=hue,
                palette=color_map,
                ax=ax,
                legend=False,
                showfliers=False,
                linewidth=1.5,
                dodge=False,
            )
            box_kwargs.update(kwargs)
            sns.boxplot(**box_kwargs)
        else:
            box_kwargs = dict(
                data=panel_df,
                x=x if orient == "v" else "_value",
                y="_value" if orient == "v" else x,
                hue=x,
                palette=group_color_map,
                ax=ax,
                legend=False,
                showfliers=False,
                linewidth=1.5,
            )
            box_kwargs.update(kwargs)
            sns.boxplot(**box_kwargs)

        if show_points:
            if hue is not None:
                strip_kwargs = dict(
                    data=panel_df,
                    x=hue if orient == "v" else "_value",
                    y="_value" if orient == "v" else hue,
                    hue=hue,
                    palette=color_map,
                    ax=ax,
                    size=point_size,
                    alpha=point_alpha,
                    dodge=False,
                    legend=False,
                    linewidth=0.5,
                    edgecolor="white",
                )
            else:
                strip_kwargs = dict(
                    data=panel_df,
                    x=x if orient == "v" else "_value",
                    y="_value" if orient == "v" else x,
                    hue=x,
                    palette=group_color_map,
                    ax=ax,
                    size=point_size,
                    alpha=point_alpha,
                    dodge=False,
                    legend=False,
                    linewidth=0.5,
                    edgecolor="white",
                )
            sns.stripplot(**strip_kwargs)

        # Per-panel axis limits when shared_colorscale=False
        if not shared_colorscale:
            _pv_min = _resolve_vmin_vmax(panel_df["_value"], vmin)
            _pv_max = _resolve_vmin_vmax(panel_df["_value"], vmax)
            if _pv_min is not None or _pv_max is not None:
                _cur = ax.get_ylim() if orient == "v" else ax.get_xlim()
                _lo = _pv_min if _pv_min is not None else _cur[0]
                _hi = _pv_max if _pv_max is not None else _cur[1]
                if orient == "v":
                    ax.set_ylim(_lo, _hi)
                else:
                    ax.set_xlim(_lo, _hi)

        # Title and axis labels
        ax.set_title(group)
        _cat_label = xlabel if xlabel is not None else (hue if hue else x)
        _val_label = ylabel if ylabel is not None else feature
        _is_bottom_row = (row == nrows - 1)
        if orient == "v":
            if border_ticks_only and not _is_bottom_row:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.tick_params(axis="x", labelrotation=xtick_rotation)
                ax.set_xlabel(_cat_label)
            ax.set_ylabel(_val_label)
        else:
            ax.set_ylabel(_cat_label)
            ax.tick_params(axis="x", labelrotation=xtick_rotation)
            ax.set_xlabel(_val_label)

    # ------------------------------------------------------------------
    # Hide unused panels
    # ------------------------------------------------------------------
    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # ------------------------------------------------------------------
    # Apply shared value-axis range
    # ------------------------------------------------------------------
    if shared_colorscale:
        if _shared_vmin is not None or _shared_vmax is not None:
            for ax in all_used_axes:
                _cur = ax.get_ylim() if orient == "v" else ax.get_xlim()
                _lo = _shared_vmin if _shared_vmin is not None else _cur[0]
                _hi = _shared_vmax if _shared_vmax is not None else _cur[1]
                if orient == "v":
                    ax.set_ylim(_lo, _hi)
                else:
                    ax.set_xlim(_lo, _hi)
        else:
            _all_lims = [
                ax.get_ylim() if orient == "v" else ax.get_xlim()
                for ax in all_used_axes
            ]
            _lo = min(lim[0] for lim in _all_lims)
            _hi = max(lim[1] for lim in _all_lims)
            for ax in all_used_axes:
                if orient == "v":
                    ax.set_ylim(_lo, _hi)
                else:
                    ax.set_xlim(_lo, _hi)

    # ------------------------------------------------------------------
    # Statistical annotations — drawn after limits are finalised
    # ------------------------------------------------------------------
    if show_stats and hue is not None:
        for ax, panel_df in zip(all_used_axes, all_panel_dfs):
            _annotate_pvalues_single_group(
                ax=ax,
                data=panel_df,
                hue=hue,
                value_col="_value",
                comparisons=comparisons,
                orient=orient,
                test=stat_test,
            )

    return fig
