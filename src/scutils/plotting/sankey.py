"""Sankey diagram for visualising cell distribution across AnnData obs categories."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from anndata import AnnData

from scutils.plotting._utils import _resolve_palette


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _category_ordered_values(adata: AnnData, col: str) -> List[str]:
    """Return ordered unique values for an obs column as strings.

    Preserves the category order for Categorical columns, or insertion
    order for plain object columns.
    """
    s = adata.obs[col]
    if hasattr(s, "cat"):
        return [str(v) for v in s.cat.categories.tolist()]
    seen: dict = {}
    for v in s:
        seen[str(v)] = None
    return list(seen)


def _to_hex(color) -> str:
    """Convert any matplotlib-compatible colour spec to a CSS hex string."""
    try:
        return mcolors.to_hex(color)
    except (ValueError, TypeError):
        return str(color)


def _to_rgba(hex_color: str, alpha: float = 0.35) -> str:
    """Convert a hex / named colour to an ``rgba(r, g, b, a)`` CSS string."""
    r, g, b, _ = mcolors.to_rgba(hex_color)
    return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"


def _resolve_level_colors(
    adata: AnnData,
    col: str,
    values: List[str],
    palette: Optional[Union[str, List[str], Dict[str, str]]],
) -> Dict[str, str]:
    """Return a ``{value: hex_color}`` dict for one obs column.

    Priority:

    1. *palette* (forwarded to :func:`~scutils.plotting._utils._resolve_palette`).
    2. ``adata.uns["{col}_colors"]``.
    3. Auto-generated ``"tab20"`` seaborn palette.
    """
    # 1. Explicit palette
    color_map = _resolve_palette(palette, values)
    if color_map is not None:
        return {k: _to_hex(v) for k, v in color_map.items()}

    # 2. adata.uns colors (set by sc.pl.* calls)
    colors_key = f"{col}_colors"
    if colors_key in adata.uns:
        obs_col = adata.obs[col]
        cats: List[str] = (
            [str(c) for c in obs_col.cat.categories.tolist()]
            if hasattr(obs_col, "cat")
            else values
        )
        uns_colors = adata.uns[colors_key]
        result = {
            str(cat): _to_hex(uns_colors[i])
            for i, cat in enumerate(cats)
            if i < len(uns_colors)
        }
        if result:
            return result

    # 3. Auto-generate — use tab20 so up to 20 distinct values get unique colours
    default = sns.color_palette("tab20", n_colors=max(len(values), 1))
    return {str(v): _to_hex(default[i % len(default)]) for i, v in enumerate(values)}


def _extract_per_col_palette(
    palette: Optional[
        Union[str, List[str], Dict[str, Union[str, List[str], Dict[str, str]]]]
    ],
    col: str,
    all_categories: List[str],
) -> Optional[Union[str, List[str], Dict[str, str]]]:
    """Extract the palette spec for *col* from the global *palette* argument.

    If *palette* is a ``dict`` whose keys are exactly the column names in
    *all_categories*, it is treated as a per-column mapping and
    ``palette[col]`` is returned (or ``None`` when *col* is absent).
    Otherwise *palette* is returned as-is (global spec applied to every level).
    """
    if isinstance(palette, dict) and all(k in all_categories for k in palette):
        return palette.get(col)
    return palette  # str, list, value→color dict, or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sankey_plot(
    adata: AnnData,
    categories: List[str],
    palette: Optional[
        Union[str, List[str], Dict[str, Union[str, List[str], Dict[str, str]]]]
    ] = None,
    height: int = 800,
    width: int = 1000,
    font_size: int = 12,
    node_pad: int = 10,
    node_thickness: int = 20,
    link_alpha: float = 0.35,
    title: Optional[str] = None,
) -> "go.Figure":  # noqa: F821 – plotly imported lazily
    """Sankey diagram showing cell counts across two or three ``adata.obs`` categories.

    Constructs an interactive Plotly Sankey diagram in which each column
    corresponds to one categorical ``adata.obs`` column.  Nodes represent
    individual category values; ribbons between adjacent columns encode the
    number of cells shared between the connected values.  Up to three levels
    (left → middle → right) are supported.

    Node colours are resolved in this priority order:

    1. The *palette* argument.
    2. ``adata.uns["{col}_colors"]`` (set automatically by Scanpy plotting
       functions such as ``sc.pl.umap``).
    3. Auto-generated ``"tab20"`` seaborn palette.

    Args:
        adata: Annotated data matrix.
        categories: Ordered list of **two or three** column names in
            ``adata.obs``.  The first column becomes the leftmost level of
            the diagram; adjacent columns are connected pairwise.
        palette: Colour specification.  Accepts:

            - ``None`` — use ``adata.uns`` colours when available, else
              auto-generate.
            - A single colour string — all nodes share that colour.
            - A list of colours — cycled across the nodes of each level
              independently.
            - A Matplotlib / seaborn palette name string — applied to each
              level independently.
            - A ``dict`` whose keys are the **column names** listed in
              *categories* — each value is a per-column spec (string, list,
              or value→colour dict).
              Example: ``{"leiden": "tab10", "cell_type": {"T cell": "red"}}``.

        height: Figure height in pixels. Defaults to ``800``.
        width: Figure width in pixels. Defaults to ``1000``.
        font_size: Global font size for node labels and hover text.
            Defaults to ``12``.
        node_pad: Vertical gap between nodes in the same column, in pixels.
            Defaults to ``10``.
        node_thickness: Width of each node rectangle in pixels.
            Defaults to ``20``.
        link_alpha: Opacity of the ribbon links (0 = fully transparent,
            1 = fully opaque).  Each ribbon inherits the colour of its
            source node, dimmed by this factor.  Defaults to ``0.35``.
        title: Optional figure title shown above the diagram.
            Defaults to ``None``.

    Returns:
        A :class:`plotly.graph_objects.Figure` containing the Sankey diagram.
        Call ``.show()`` to render it interactively, or ``.write_html()`` /
        ``.write_image()`` to export.

    Raises:
        ImportError: If ``plotly`` is not installed.
        ValueError: If *categories* does not contain exactly 2 or 3 items.
        ValueError: If any element of *categories* is not a column in
            ``adata.obs``.

    Example:
        >>> fig = scutils.pl.sankey_plot(adata, categories=["leiden", "cell_type"])
        >>> fig.show()

        >>> fig = scutils.pl.sankey_plot(
        ...     adata,
        ...     categories=["compartment", "cell_type", "cell_subtype"],
        ...     palette={"compartment": "Set2", "cell_type": "tab10"},
        ...     height=1200,
        ...     width=1400,
        ... )
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for sankey_plot. "
            "Install it with:  pip install plotly"
        ) from exc

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not (2 <= len(categories) <= 3):
        raise ValueError(
            f"categories must contain 2 or 3 column names, "
            f"got {len(categories)}: {categories}"
        )
    for col in categories:
        if col not in adata.obs.columns:
            raise ValueError(
                f"Column '{col}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

    # ------------------------------------------------------------------
    # Ordered unique values per level
    # ------------------------------------------------------------------
    level_values: List[List[str]] = [
        _category_ordered_values(adata, col) for col in categories
    ]

    # ------------------------------------------------------------------
    # Assign a flat integer index to every (level_idx, value) pair.
    # Using a tuple key avoids collisions when the same string appears
    # in multiple categories.
    # ------------------------------------------------------------------
    node_index: Dict[tuple, int] = {}
    node_labels: List[str] = []      # shown on the diagram (bold HTML)
    node_colors: List[str] = []      # hex strings
    node_customdata: List[str] = []  # column name — shown in hover

    for level_idx, (col, vals) in enumerate(zip(categories, level_values)):
        per_col_pal = _extract_per_col_palette(palette, col, categories)
        color_map = _resolve_level_colors(adata, col, vals, per_col_pal)
        for val in vals:
            idx = len(node_labels)
            node_index[(level_idx, val)] = idx
            node_labels.append(f"<b>{val}</b>")
            node_colors.append(color_map.get(val, "#aaaaaa"))
            node_customdata.append(col)

    # ------------------------------------------------------------------
    # Build links for each adjacent pair of levels
    # ------------------------------------------------------------------
    link_sources: List[int] = []
    link_targets: List[int] = []
    link_values: List[int] = []
    link_colors_rgba: List[str] = []

    for level_idx in range(len(categories) - 1):
        col_a = categories[level_idx]
        col_b = categories[level_idx + 1]

        counts: pd.DataFrame = (
            adata.obs
            .groupby([col_a, col_b], observed=True)
            .size()
            .reset_index(name="_count")
        )

        for _, row in counts.iterrows():
            val_a = str(row[col_a])
            val_b = str(row[col_b])
            count = int(row["_count"])
            if count == 0:
                continue
            src_idx = node_index.get((level_idx, val_a))
            tgt_idx = node_index.get((level_idx + 1, val_b))
            if src_idx is None or tgt_idx is None:
                continue
            link_sources.append(src_idx)
            link_targets.append(tgt_idx)
            link_values.append(count)
            link_colors_rgba.append(_to_rgba(node_colors[src_idx], link_alpha))

    # ------------------------------------------------------------------
    # Assemble Plotly figure
    # ------------------------------------------------------------------
    fig = go.Figure(
        go.Sankey(
            valueformat=",d",
            node=dict(
                pad=node_pad,
                thickness=node_thickness,
                line=dict(color="black", width=0.5),
                color=node_colors,
                label=node_labels,
                customdata=node_customdata,
                hovertemplate=(
                    "<b>%{customdata}</b>: %{label}<br>"
                    "Cells: <b>%{value:,}</b>"
                    "<extra></extra>"
                ),
            ),
            link=dict(
                source=link_sources,
                target=link_targets,
                value=link_values,
                color=link_colors_rgba,
                hovertemplate=(
                    "<b>%{source.customdata}</b>: %{source.label}<br>"
                    "<b>%{target.customdata}</b>: %{target.label}<br>"
                    "Cells: <b>%{value:,}</b>"
                    "<extra></extra>"
                ),
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5) if title else None,
        font=dict(color="black", size=font_size),
        height=height,
        width=width,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig
