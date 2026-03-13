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
              *category* value (one bar per *group_by* value), separated from
              adjacent category blocks by a small gap.  Each bar carries a
              text label identifying its *group_by* value.
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

            When *group_by* is set, colours are resolved from the *group_by*
            column (since that is what distinguishes the bars visually).
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

    cat_values = _ordered_values(adata, category)
    x = np.arange(len(cat_values))
    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------
    # Single category
    # ------------------------------------------------------------------
    if group_by is None:
        raw = adata.obs[category].astype(str).value_counts()
        heights = np.array([raw.get(v, 0) for v in cat_values], dtype=float)
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

        if normalize:
            row_totals = data.sum(axis=1, keepdims=True)
            row_totals = np.where(row_totals == 0, 1, row_totals)
            data = data / row_totals

        if mode == "grouped":
            # Each category value gets a consecutive block of full-width bars
            # (one per group_by value), with a one-bar-width gap between
            # blocks.  No sub-cluster offset → bars stay as wide as possible.
            cat_colors = _bar_colors(adata, category, cat_values, palette)
            n_grp = len(grp_values)
            bar_width = 0.8
            gap = 1  # empty-bar-width spacing between category blocks

            _glkw: Dict[str, Any] = {"fontsize": 8, "ha": "center"}
            _glkw.update(group_label_kwargs or {})
            _gl_va = _glkw.pop(
                "va",
                "center" if group_label_position == "inside" else "bottom",
            )

            tick_positions: List[float] = []
            for i, (cat_val, cat_color) in enumerate(
                zip(cat_values, cat_colors)
            ):
                block_start = i * (n_grp + gap)
                positions = [block_start + j for j in range(n_grp)]
                tick_positions.append(block_start + (n_grp - 1) / 2.0)

                heights = data[i, :]
                bars = ax.bar(
                    positions, heights, width=bar_width,
                    color=cat_color,
                    edgecolor=bar_edgecolor, linewidth=bar_linewidth,
                    **kwargs,
                )
                for bar, h, grp_val in zip(bars, heights, grp_values):
                    if h <= 0:
                        continue
                    xc = bar.get_x() + bar.get_width() / 2
                    yc = h / 2 if group_label_position == "inside" else h
                    ax.text(xc, yc, grp_val, va=_gl_va, **_glkw)
                    if show_counts:
                        ax.text(
                            xc, h,
                            _label(h, normalize),
                            ha="center", va="bottom", fontsize=8,
                        )

            # Update x so the shared axes-formatting block below places
            # ticks at the centre of each category's bar block.
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
    if title:
        ax.set_title(title)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
