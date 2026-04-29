"""Bar plots for visualising cell counts across AnnData obs categories."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure

from scutils.plotting._utils import _resolve_palette


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bar_colors(
    adata: AnnData,
    col: str,
    values: List[str],
    palette: Optional[Union[str, List[str], Dict[str, str]]],
) -> List[str]:
    """Return a list of hex colour strings for *values* in obs column *col*.

    Priority: *palette* → ``adata.uns["{col}_colors"]`` → ``"tab20"`` fallback.
    """
    color_map = _resolve_palette(palette, values)
    if color_map is not None:
        return [mcolors.to_hex(color_map.get(v, "#aaaaaa")) for v in values]

    colors_key = f"{col}_colors"
    if colors_key in adata.uns:
        obs_col = adata.obs[col]
        cats: List[str] = (
            [str(c) for c in obs_col.cat.categories.tolist()]
            if hasattr(obs_col, "cat")
            else values
        )
        uns_colors = adata.uns[colors_key]
        cat_to_color = {
            str(cat): mcolors.to_hex(uns_colors[i])
            for i, cat in enumerate(cats)
            if i < len(uns_colors)
        }
        if cat_to_color:
            return [cat_to_color.get(v, "#aaaaaa") for v in values]

    default = sns.color_palette("tab20", n_colors=max(len(values), 1))
    return [mcolors.to_hex(default[i % len(default)]) for i, _ in enumerate(values)]


def _ordered_values(adata: AnnData, col: str) -> List[str]:
    """Return ordered unique values for an obs column, preserving category order."""
    s = adata.obs[col]
    if hasattr(s, "cat"):
        return [str(v) for v in s.cat.categories.tolist()]
    seen: dict = {}
    for v in s:
        seen[str(v)] = None
    return list(seen)


def _label(value: float, normalize: bool) -> str:
    """Format a bar count/fraction label."""
    return f"{value:.2f}" if normalize else f"{int(value):,}"


# ---------------------------------------------------------------------------
# Internal drawing helper
# ---------------------------------------------------------------------------


def _draw_cell_count_barplot_on_ax(
    ax: Any,
    adata: AnnData,
    category: str,
    group_by: Optional[str],
    mode: Literal["grouped", "stacked"],
    normalize: bool,
    palette: Optional[Union[str, List[str], Dict[str, str]]],
    show_counts: bool,
    bar_linewidth: float,
    bar_edgecolor: str,
    group_label_position: Literal["inside", "above"],
    group_label_kwargs: Optional[Dict[str, Any]],
    show_category_separators: bool,
    rotation: int,
    xlabel: Optional[str],
    ylabel: Optional[str],
    sort_x: Optional[Literal["ascending", "descending"]] = None,
    sort_by_group: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Draw a cell-count bar chart onto an existing Axes.

    Internal helper shared by :func:`cell_count_barplot` and
    :func:`cell_count_barplot_multiplot`.  All argument validation and
    figure creation must be performed by the caller.

    Args:
        ax: The :class:`~matplotlib.axes.Axes` to draw on.
        adata: Annotated data matrix (may be a subset for one panel).
        category: Column in ``adata.obs`` for the x-axis groups.
        group_by: Optional second grouping column.
        mode: ``"grouped"`` or ``"stacked"`` when *group_by* is set.
        normalize: Express bar heights as fractions when ``True``.
        palette: Colour specification (see :func:`cell_count_barplot`).
        show_counts: Annotate bars with counts / fractions when ``True``.
        bar_linewidth: Width of bar edges.
        bar_edgecolor: Colour of bar edges.
        group_label_position: ``"inside"`` or ``"above"`` for group labels.
        group_label_kwargs: Extra kwargs forwarded to ``ax.text`` /
            ``ax.annotate`` for group labels in grouped mode.
        show_category_separators: Draw vertical separators between category
            blocks in grouped mode.
        rotation: x-axis tick-label rotation in degrees.
        xlabel: x-axis label override.  When ``None``, defaults to *category*.
        ylabel: y-axis label override.  When ``None``, defaults to
            ``"Number of cells"`` or ``"Fraction of cells"``.
        sort_x: When ``"ascending"`` or ``"descending"``, reorders
            x-axis ticks in the specified direction.  By default (when
            *sort_by_group* is ``None``) sorts by raw row totals.  When
            *sort_by_group* is set, sorts by the count / fraction of that
            specific *group_by* value instead.  When ``None``, the
            original categorical order is preserved.  Defaults to
            ``None``.
        sort_by_group: When set, the x-axis is sorted by the count
            (``normalize=False``) or fraction (``normalize=True``) of
            this specific *group_by* category value, instead of by the
            row total.  Requires *group_by* to be set and *sort_x* to
            specify the direction.  Defaults to ``None``.
        **kwargs: Forwarded to :func:`matplotlib.axes.Axes.bar`.
    """
    cat_values = _ordered_values(adata, category)
    x = np.arange(len(cat_values))

    # ------------------------------------------------------------------
    # Single category
    # ------------------------------------------------------------------
    if group_by is None:
        raw = adata.obs[category].astype(str).value_counts()
        heights = np.array([raw.get(v, 0) for v in cat_values], dtype=float)

        if sort_x is not None:
            _sort_idx = np.argsort(heights)
            if sort_x == "descending":
                _sort_idx = _sort_idx[::-1]
            cat_values = [cat_values[i] for i in _sort_idx]
            heights = heights[_sort_idx]

        if normalize:
            total = heights.sum()
            heights = heights / total if total > 0 else heights

        colors = _bar_colors(adata, category, cat_values, palette)
        bars = ax.bar(
            x, heights, color=colors,
            edgecolor=bar_edgecolor, linewidth=bar_linewidth,
            **kwargs,
        )

        if show_counts:
            for bar, h in zip(bars, heights):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    _label(h, normalize),
                    ha="center", va="bottom", fontsize=9,
                )

    # ------------------------------------------------------------------
    # Two categories
    # ------------------------------------------------------------------
    else:
        grp_values = _ordered_values(adata, group_by)
        counts_df = (
            adata.obs
            .groupby([category, group_by], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(index=cat_values, columns=grp_values, fill_value=0)
        )
        data = counts_df.values.astype(float)  # shape: (n_cat, n_grp)

        if sort_x is not None:
            if sort_by_group is not None:
                if sort_by_group not in grp_values:
                    raise ValueError(
                        f"sort_by_group='{sort_by_group}' not found in "
                        f"group_by='{group_by}' values. "
                        f"Available values: {grp_values}"
                    )
                grp_idx = grp_values.index(sort_by_group)
                if normalize:
                    _totals = data.sum(axis=1)
                    _totals = np.where(_totals == 0, 1, _totals)
                    _sort_key = data[:, grp_idx] / _totals
                else:
                    _sort_key = data[:, grp_idx]
            else:
                _sort_key = data.sum(axis=1)
            _sort_idx = np.argsort(_sort_key)
            if sort_x == "descending":
                _sort_idx = _sort_idx[::-1]
            cat_values = [cat_values[i] for i in _sort_idx]
            data = data[_sort_idx, :]

        if normalize:
            row_totals = data.sum(axis=1, keepdims=True)
            row_totals = np.where(row_totals == 0, 1, row_totals)
            data = data / row_totals

        if mode == "grouped":
            # Bars are packed with NO gaps.  A running position counter
            # advances by exactly the number of non-zero group_by values
            # in each category, so zero-count combinations never waste
            # x-axis space.  Categories with more present group_by values
            # naturally occupy proportionally more of the axis while every
            # bar stays the same width.
            cat_colors = _bar_colors(adata, category, cat_values, palette)
            n_grp = len(grp_values)
            bar_width = 0.8

            _glkw: Dict[str, Any] = {"fontsize": 8, "ha": "center"}
            _glkw.update(group_label_kwargs or {})
            _gl_va = _glkw.pop(
                "va",
                "center" if group_label_position == "inside" else "bottom",
            )

            tick_positions: List[float] = []
            separator_positions: List[float] = []
            current_pos: int = 0
            for i, (cat_val, cat_color) in enumerate(
                zip(cat_values, cat_colors)
            ):
                # Record boundary between the previous category block and
                # this one (midpoint of the 0.2-unit gap between the bars).
                if i > 0:
                    separator_positions.append(current_pos - 0.5)

                heights = data[i, :]
                # Only allocate positions for non-zero bars; invisible
                # zero-height bars would waste x-axis space.
                mask = heights > 0
                present_heights = heights[mask]
                present_grps = [
                    grp_values[j] for j, m in enumerate(mask) if m
                ]
                n_present = len(present_heights)

                if n_present == 0:
                    # No data for this category — reserve one placeholder
                    # slot so the tick still appears.
                    tick_positions.append(float(current_pos))
                    current_pos += 1
                    continue

                positions = [current_pos + j for j in range(n_present)]
                tick_positions.append(current_pos + (n_present - 1) / 2.0)
                current_pos += n_present

                bars = ax.bar(
                    positions, present_heights, width=bar_width,
                    color=cat_color,
                    edgecolor=bar_edgecolor, linewidth=bar_linewidth,
                    **kwargs,
                )
                for bar, h, grp_val in zip(
                    bars, present_heights, present_grps
                ):
                    xc = bar.get_x() + bar.get_width() / 2
                    if group_label_position == "inside":
                        ax.text(xc, h / 2, grp_val, va=_gl_va, **_glkw)
                    else:
                        # Use a fixed point offset so the gap between bar
                        # top and label is independent of the data scale.
                        ax.annotate(
                            grp_val, xy=(xc, h),
                            xytext=(0, 4), textcoords="offset points",
                            va=_gl_va, **_glkw,
                        )
                    if show_counts:
                        ax.text(
                            xc, h,
                            _label(h, normalize),
                            ha="center", va="bottom", fontsize=8,
                        )

            if show_category_separators:
                for sep_x in separator_positions:
                    ax.axvline(
                        sep_x, color="0.6", linewidth=1.0,
                        linestyle="-", zorder=0,
                    )

            # x is used by the shared axes-formatting block below.
            x = np.array(tick_positions)

        else:  # stacked
            colors = _bar_colors(adata, group_by, grp_values, palette)
            bottoms = np.zeros(len(cat_values))
            for i, (grp_val, color) in enumerate(zip(grp_values, colors)):
                heights = data[:, i]
                bars = ax.bar(
                    x, heights, bottom=bottoms,
                    color=color, label=grp_val,
                    edgecolor=bar_edgecolor, linewidth=bar_linewidth,
                    **kwargs,
                )
                if show_counts:
                    for bar, h, b in zip(bars, heights, bottoms):
                        if h > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                b + h / 2,
                                _label(h, normalize),
                                ha="center", va="center", fontsize=8,
                            )
                bottoms += heights

            ax.legend(
                title=group_by,
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                borderaxespad=0,
                frameon=False,
            )

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        cat_values,
        rotation=rotation,
        ha="right" if rotation not in (0, 360) else "center",
    )
    ax.set_xlabel(xlabel if xlabel is not None else category)
    ax.set_ylabel(
        ylabel if ylabel is not None
        else ("Fraction of cells" if normalize else "Number of cells")
    )
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cell_count_barplot(
    adata: AnnData,
    category: str,
    group_by: Optional[str] = None,
    mode: Literal["grouped", "stacked"] = "grouped",
    normalize: bool = False,
    palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    figsize: Tuple[float, float] = (6, 4),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    rotation: int = 45,
    show_counts: bool = False,
    bar_linewidth: float = 0.8,
    bar_edgecolor: str = "black",
    group_label_position: Literal["inside", "above"] = "inside",
    group_label_kwargs: Optional[Dict[str, Any]] = None,
    show_category_separators: bool = True,
    sort_x: Optional[Literal["ascending", "descending"]] = None,
    sort_by_group: Optional[str] = None,
    **kwargs: Any,
) -> Figure:
    """Bar plot showing cell counts per category or per pair of categories.

    Produces a Matplotlib bar chart where bar heights encode the number of
    cells (or fraction when *normalize* is ``True``).  When *group_by* is
    provided, bars are split by a second ``adata.obs`` column and rendered
    either side-by-side (``mode="grouped"``) or layered
    (``mode="stacked"``).

    Args:
        adata: Annotated data matrix.
        category: Column name in ``adata.obs`` used for the x-axis.  Each
            unique value in this column becomes a tick on the x-axis.
        group_by: Optional second column name in ``adata.obs`` used to split
            bars by a second grouping.  When ``None``, a simple single-series
            bar chart is produced.  Defaults to ``None``.
        mode: How to render the second grouping when *group_by* is set.

            - ``"grouped"`` *(default)* — one block of full-width bars per
              *category* value (one bar per non-zero *group_by* combination),
              packed with no gaps between category blocks.  Categories with
              more *group_by* values occupy proportionally more x-axis space
              so that every bar has the same width.  Each bar carries a text
              label identifying its *group_by* value.
            - ``"stacked"`` — one stacked bar per *category* value, with
              segments coloured by *group_by* value.

            Ignored when *group_by* is ``None``.
        normalize: When ``True``, bar heights are expressed as fractions of
            the total cell count (single-category) or fractions within each
            *category* value (two-category stacked/grouped).  Defaults to
            ``False``.
        palette: Colour specification for the bars.  Accepts:

            - ``None`` — use ``adata.uns["{col}_colors"]`` when available,
              else auto-generate from ``"tab20"``.
            - A single colour string — all bars share that colour.
            - A list of colours — cycled across the category values.
            - A Matplotlib / seaborn palette name string.
            - A ``dict`` mapping category labels to colours.

            In ``mode="grouped"``, colours are resolved from the *category*
            column (bars within a block share a colour; *group_by* values
            are identified by text labels).  In ``mode="stacked"``, colours
            are resolved from the *group_by* column.
            Do **not** pass ``color`` via ``**kwargs``; use this parameter
            instead.
        figsize: Figure size as ``(width, height)`` in inches.
            Defaults to ``(6, 4)``.
        title: Optional figure title.  Defaults to ``None``.
        xlabel: x-axis label.  Defaults to *category*.
        ylabel: y-axis label.  Defaults to ``"Number of cells"`` or
            ``"Fraction of cells"`` depending on *normalize*.
        rotation: Rotation angle for x-axis tick labels in degrees.
            Defaults to ``45``.
        show_counts: When ``True``, annotate each bar (or segment for
            ``mode="stacked"``) with its numeric value.  In grouped mode
            the count appears above the bar alongside the group label.
            Defaults to ``False``.
        bar_linewidth: Line width of the bar edges in points.  Defaults
            to ``0.8``.
        bar_edgecolor: Colour of the bar edges.  Defaults to ``"black"``.
        group_label_position: Where to place the *group_by* text label on
            each bar in ``mode="grouped"``.

            - ``"inside"`` *(default)* — label is centred vertically
              within the bar.  Works best when bars are tall enough;
              consider ``group_label_kwargs={"rotation": 90}`` for narrow
              bars.
            - ``"above"`` — label sits just above the top of the bar.

        group_label_kwargs: Keyword arguments forwarded to
            :func:`matplotlib.axes.Axes.text` for the group labels in
            ``mode="grouped"``.  Use this to control font size, colour,
            weight, rotation, etc.
            Example: ``{"fontsize": 7, "color": "white", "rotation": 90}``.
            The ``"va"`` key overrides the default vertical alignment
            inferred from *group_label_position*.  Defaults to ``None``.
        show_category_separators: When ``True`` (default), a thin vertical
            line is drawn between adjacent *category* blocks in
            ``mode="grouped"`` to make it visually clear which bars belong
            to the same *category* value.  Has no effect in other modes.
            Defaults to ``True``.
        sort_x: When set to ``"ascending"`` or ``"descending"``, the
            x-axis ticks (i.e. *category* values) are reordered in the
            specified direction.  By default (when *sort_by_group* is
            ``None``) sorts by total raw cell count per category.  When
            ``None``, the original categorical order is preserved.
            Defaults to ``None``.
        sort_by_group: Specifies which *group_by* category value to use
            as the sort key when *sort_x* is set.  When
            ``normalize=False``, categories are sorted by the raw count
            of *sort_by_group*.  When ``normalize=True``, categories are
            sorted by the **fraction** of cells belonging to
            *sort_by_group* within each *category*.  Requires *group_by*
            to be set.  When ``None``, sorts by total count (default
            behaviour).  Defaults to ``None``.
        **kwargs: Additional keyword arguments forwarded to
            :func:`matplotlib.axes.Axes.bar` (e.g. ``alpha``, ``zorder``).

    Returns:
        A :class:`matplotlib.figure.Figure`.  Call ``plt.show()`` to display
        it interactively, or ``fig.savefig()`` to export.

    Raises:
        ValueError: If *category* is not a column in ``adata.obs``.
        ValueError: If *group_by* is not a column in ``adata.obs``.
        ValueError: If *mode* is not ``"grouped"`` or ``"stacked"``.

    Example:
        >>> # Simple cell count per cell type
        >>> fig = scutils.pl.cell_count_barplot(adata, category="cell_type")

        >>> # Grouped bars: cell type on x-axis, condition as colour
        >>> fig = scutils.pl.cell_count_barplot(
        ...     adata,
        ...     category="cell_type",
        ...     group_by="condition",
        ...     mode="grouped",
        ... )

        >>> # Stacked normalised bars showing condition proportions per donor
        >>> fig = scutils.pl.cell_count_barplot(
        ...     adata,
        ...     category="donor",
        ...     group_by="condition",
        ...     mode="stacked",
        ...     normalize=True,
        ... )
    """
    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if category not in adata.obs.columns:
        raise ValueError(
            f"Column '{category}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if group_by is not None and group_by not in adata.obs.columns:
        raise ValueError(
            f"Column '{group_by}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if mode not in ("grouped", "stacked"):
        raise ValueError(f"mode must be 'grouped' or 'stacked', got '{mode}'.")
    if sort_by_group is not None and group_by is None:
        raise ValueError("sort_by_group requires group_by to be set.")

    fig, ax = plt.subplots(figsize=figsize)
    _draw_cell_count_barplot_on_ax(
        ax=ax,
        adata=adata,
        category=category,
        group_by=group_by,
        mode=mode,
        normalize=normalize,
        palette=palette,
        show_counts=show_counts,
        bar_linewidth=bar_linewidth,
        bar_edgecolor=bar_edgecolor,
        group_label_position=group_label_position,
        group_label_kwargs=group_label_kwargs,
        show_category_separators=show_category_separators,
        rotation=rotation,
        xlabel=xlabel,
        ylabel=ylabel,
        sort_x=sort_x,
        sort_by_group=sort_by_group,
        **kwargs,
    )
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def cell_count_barplot_multiplot(
    adata: AnnData,
    category: str,
    panel: str,
    group_by: Optional[str] = None,
    panel_order: Optional[List[str]] = None,
    mode: Literal["grouped", "stacked"] = "grouped",
    normalize: bool = False,
    palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
    ncols: int = 3,
    shared_y: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    hspace: float = 0.5,
    wspace: float = 0.3,
    border_ticks_only: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    rotation: int = 45,
    show_counts: bool = False,
    bar_linewidth: float = 0.8,
    bar_edgecolor: str = "black",
    group_label_position: Literal["inside", "above"] = "inside",
    group_label_kwargs: Optional[Dict[str, Any]] = None,
    show_category_separators: bool = True,
    sort_x: Optional[Literal["ascending", "descending"]] = None,
    sort_by_group: Optional[str] = None,
    **kwargs: Any,
) -> Figure:
    """Grid of cell-count bar plots — one panel per category of *panel*.

    Splits ``adata`` into subsets based on the unique values of the *panel*
    column and produces one :func:`cell_count_barplot` per subset, laid out
    in an ``nrows × ncols`` grid.  The y-axis can optionally be shared
    across all panels (``shared_y=True``) so that absolute counts are
    directly comparable.

    Args:
        adata: Annotated data matrix.
        category: Column in ``adata.obs`` used for the x-axis within each
            panel.  Each unique value becomes a tick.
        panel: Column in ``adata.obs`` whose unique values determine the
            panel split.  One panel is drawn per unique value.
        group_by: Optional second column for sub-grouping within each panel.
            Passed through to the underlying bar-chart drawing logic.
            Defaults to ``None``.
        panel_order: Explicit ordering of *panel* values.  Must be a subset
            of (or equal to) the unique values in the *panel* column.  When
            ``None``, the natural categorical / first-seen order is used.
            Defaults to ``None``.
        mode: Bar chart mode when *group_by* is set: ``"grouped"``
            (side-by-side bars) or ``"stacked"``.  Defaults to
            ``"grouped"``.
        normalize: When ``True``, bar heights are fractions within each
            *category* group.  Defaults to ``False``.
        palette: Colour specification.  See :func:`cell_count_barplot`.
            Defaults to ``None``.
        ncols: Number of columns in the panel grid.  Defaults to ``3``.
        shared_y: When ``True``, all panels share the same y-axis range,
            enabling direct count comparison.  When ``False``, each panel
            autoscales independently.  Defaults to ``True``.
        figsize: Size of a **single** panel ``(width, height)`` in inches.
            The total figure size is computed as
            ``(ncols × width, nrows × height)``.  When ``None``, defaults
            to ``(6, 4)`` per panel.  Defaults to ``None``.
        hspace: Vertical space between subplot rows as a fraction of the
            average axes height.  Defaults to ``0.5``.
        wspace: Horizontal space between subplot columns as a fraction of
            the average axes width.  Defaults to ``0.3``.
        border_ticks_only: When ``True``, x-axis tick labels and the x-axis
            label are shown only on the bottom row of panels, reducing
            clutter in multi-row grids.  Defaults to ``True``.
        title: Overall figure super-title.  Defaults to ``None``.
        xlabel: x-axis label override applied to all panels.  When ``None``,
            defaults to *category*.  Defaults to ``None``.
        ylabel: y-axis label override applied to all panels.  When ``None``,
            defaults to ``"Number of cells"`` or ``"Fraction of cells"``.
            Defaults to ``None``.
        rotation: Rotation angle for x-axis tick labels in degrees.
            Defaults to ``45``.
        show_counts: Annotate bars with numeric values.  Defaults to
            ``False``.
        bar_linewidth: Width of bar edges.  Defaults to ``0.8``.
        bar_edgecolor: Colour of bar edges.  Defaults to ``"black"``.
        group_label_position: ``"inside"`` or ``"above"`` for *group_by*
            labels in grouped mode.  Defaults to ``"inside"``.
        group_label_kwargs: Extra kwargs forwarded to ``ax.text`` /
            ``ax.annotate`` for group labels.  Defaults to ``None``.
        show_category_separators: Draw vertical separators between category
            blocks in grouped mode.  Defaults to ``True``.
        sort_x: When set to ``"ascending"`` or ``"descending"``, the
            x-axis ticks within every panel are reordered in the
            specified direction.  Each panel is sorted independently on
            its own subset of *adata*.  By default (when *sort_by_group*
            is ``None``) sorts by total raw cell count per category.
            When ``None``, the original categorical order is preserved.
            Defaults to ``None``.
        sort_by_group: Specifies which *group_by* category value to use
            as the sort key within each panel when *sort_x* is set.
            When ``normalize=False``, categories are sorted by the raw
            count of *sort_by_group*; when ``normalize=True``, they are
            sorted by the **fraction** of cells belonging to
            *sort_by_group* within each *category*.  Requires *group_by*
            to be set.  When ``None``, sorts by total count.
            Defaults to ``None``.
        **kwargs: Additional keyword arguments forwarded to
            :func:`matplotlib.axes.Axes.bar`.

    Returns:
        A :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If *category*, *panel*, or *group_by* is not a column
            in ``adata.obs``.
        ValueError: If *mode* is not ``"grouped"`` or ``"stacked"``.
        ValueError: If any value in *panel_order* is absent from the
            *panel* column.

    Example:
        >>> # One bar-plot panel per donor, x-axis = cell type
        >>> fig = scutils.pl.cell_count_barplot_multiplot(
        ...     adata,
        ...     category="cell_type",
        ...     panel="donor",
        ...     ncols=2,
        ... )

        >>> # Stacked + normalised, split by condition
        >>> fig = scutils.pl.cell_count_barplot_multiplot(
        ...     adata,
        ...     category="cell_type",
        ...     panel="condition",
        ...     group_by="donor",
        ...     mode="stacked",
        ...     normalize=True,
        ...     shared_y=False,
        ... )
    """
    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if category not in adata.obs.columns:
        raise ValueError(
            f"Column '{category}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if panel not in adata.obs.columns:
        raise ValueError(
            f"Column '{panel}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if group_by is not None and group_by not in adata.obs.columns:
        raise ValueError(
            f"Column '{group_by}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if mode not in ("grouped", "stacked"):
        raise ValueError(f"mode must be 'grouped' or 'stacked', got '{mode}'.")
    if sort_by_group is not None and group_by is None:
        raise ValueError("sort_by_group requires group_by to be set.")

    # ------------------------------------------------------------------
    # Panel values and ordering
    # ------------------------------------------------------------------
    all_panel_vals = _ordered_values(adata, panel)
    if panel_order is not None:
        invalid = [p for p in panel_order if p not in all_panel_vals]
        if invalid:
            raise ValueError(
                f"panel_order values {invalid} not found in panel='{panel}'. "
                f"Available: {all_panel_vals}"
            )
        panel_vals = [p for p in panel_order if p in all_panel_vals]
    else:
        panel_vals = all_panel_vals

    n_panels = len(panel_vals)
    _ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / _ncols))

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    if figsize is not None:
        panel_w, panel_h = float(figsize[0]), float(figsize[1])
    else:
        panel_w, panel_h = 6.0, 4.0

    fig, axes = plt.subplots(
        nrows, _ncols,
        figsize=(panel_w * _ncols, panel_h * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.01)

    # ------------------------------------------------------------------
    # Draw panels
    # ------------------------------------------------------------------
    used_axes: List[Any] = []
    panel_col_str = adata.obs[panel].astype(str)

    for idx, panel_val in enumerate(panel_vals):
        row = idx // _ncols
        col = idx % _ncols
        ax = axes[row, col]
        used_axes.append(ax)

        adata_panel = adata[panel_col_str == str(panel_val)]

        _draw_cell_count_barplot_on_ax(
            ax=ax,
            adata=adata_panel,
            category=category,
            group_by=group_by,
            mode=mode,
            normalize=normalize,
            palette=palette,
            show_counts=show_counts,
            bar_linewidth=bar_linewidth,
            bar_edgecolor=bar_edgecolor,
            group_label_position=group_label_position,
            group_label_kwargs=group_label_kwargs,
            show_category_separators=show_category_separators,
            rotation=rotation,
            xlabel=xlabel,
            ylabel=ylabel,
            sort_x=sort_x,
            sort_by_group=sort_by_group,
            **kwargs,
        )
        ax.set_title(str(panel_val))

        if border_ticks_only and row < nrows - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    # ------------------------------------------------------------------
    # Hide unused axes
    # ------------------------------------------------------------------
    for idx in range(n_panels, nrows * _ncols):
        axes[idx // _ncols, idx % _ncols].set_visible(False)

    # ------------------------------------------------------------------
    # Shared y-axis
    # ------------------------------------------------------------------
    if shared_y and used_axes:
        global_ymin = min(ax.get_ylim()[0] for ax in used_axes)
        global_ymax = max(ax.get_ylim()[1] for ax in used_axes)
        for ax in used_axes:
            ax.set_ylim(global_ymin, global_ymax)

    return fig
