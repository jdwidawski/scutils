"""UpSet plot visualisation for set intersections."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def upset_plot(
    contents: Dict[str, Set[str]],
    *,
    figsize: Tuple[float, float] = (10.0, 6.0),
    sort_by: Literal["cardinality", "degree", "-cardinality", "-degree", "input"] = "cardinality",
    sort_categories_by: Literal["cardinality", "input", "-cardinality"] | None = "cardinality",
    show_counts: bool | str = True,
    show_percentages: bool = False,
    min_subset_size: int = 0,
    min_degree: int = 0,
    max_degree: Optional[int] = None,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    facecolor: str = "black",
    element_size: Optional[float] = None,
    **kwargs: Any,
) -> Figure:
    """Create an UpSet plot from a dictionary of named sets.

    Wraps :func:`upsetplot.from_contents` and :class:`upsetplot.UpSet`
    to produce a publication-ready intersection plot.  Returns a
    :class:`matplotlib.figure.Figure` without calling ``plt.show()``.

    Args:
        contents: Mapping of set names to sets of member identifiers.
            For example
            ``{"Comparison_A": {"g1", "g2"}, "Comparison_B": {"g2", "g3"}}``.
        figsize: ``(width, height)`` of the figure in inches.  Defaults to
            ``(10.0, 6.0)``.
        sort_by: How to sort intersections.  ``"cardinality"`` (default)
            sorts by the size of each intersection; ``"degree"`` by the
            number of sets involved; prefix with ``"-"`` to reverse.
            ``"input"`` preserves the insertion order.
        sort_categories_by: How to sort the set (category) rows.
            ``"cardinality"`` (default) orders by total set size;
            ``"input"`` preserves the dict insertion order; ``None``
            applies no explicit sort.
        show_counts: If ``True`` or a format string (e.g. ``"%d"``),
            display intersection sizes above the bars.  Defaults to
            ``True``.
        show_percentages: Show intersection sizes as percentages of the
            total number of unique elements.  Defaults to ``False``.
        min_subset_size: Hide intersections smaller than this value.
            Defaults to ``0`` (show all).
        min_degree: Minimum intersection degree to display.  Defaults to
            ``0``.
        max_degree: Maximum intersection degree to display.  ``None``
            (default) imposes no upper limit.
        orientation: ``"horizontal"`` (default) or ``"vertical"`` layout.
        facecolor: Colour used for the intersection bars and matrix dots.
            Defaults to ``"black"``.
        element_size: Size scaling factor passed to
            :class:`upsetplot.UpSet`.  When ``None`` (default) the library
            chooses automatically.
        **kwargs: Additional keyword arguments forwarded to
            :class:`upsetplot.UpSet`.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the UpSet plot.

    Raises:
        ValueError: If *contents* is empty, has fewer than 2 sets, or
            all sets are empty.
        ImportError: If the ``upsetplot`` package is not installed.

    Example:
        >>> gene_sets = {
        ...     "Comp_A": {"TP53", "BRCA1", "MYC", "EGFR"},
        ...     "Comp_B": {"TP53", "MYC", "KRAS"},
        ...     "Comp_C": {"TP53", "BRCA1", "KRAS", "PTEN"},
        ... }
        >>> fig = upset_plot(gene_sets, show_counts=True)
        >>> fig.savefig("upset.png", dpi=150, bbox_inches="tight")
    """
    try:
        from upsetplot import UpSet, from_contents
    except ImportError as exc:
        raise ImportError(
            "upsetplot is required for UpSet plots.  "
            "Install it with:  pip install upsetplot"
        ) from exc

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if not contents:
        raise ValueError("contents must be a non-empty dict of sets.")

    if len(contents) < 2:
        raise ValueError(
            "UpSet plots require at least 2 sets to visualise "
            "intersections.  Got {0} set(s).".format(len(contents))
        )

    all_elements = set().union(*contents.values())
    if len(all_elements) == 0:
        raise ValueError(
            "All sets in contents are empty — nothing to plot."
        )

    # ------------------------------------------------------------------
    # Build the UpSet data structure
    # ------------------------------------------------------------------
    data = from_contents(contents)

    # ------------------------------------------------------------------
    # Configure and render
    # ------------------------------------------------------------------
    upset_kw: Dict[str, Any] = {
        "sort_by": sort_by,
        "sort_categories_by": sort_categories_by,
        "show_counts": show_counts,
        "show_percentages": show_percentages,
        "min_subset_size": min_subset_size,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "orientation": orientation,
        "facecolor": facecolor,
    }
    if element_size is not None:
        upset_kw["element_size"] = element_size

    upset_kw.update(kwargs)

    upset = UpSet(data, **upset_kw)

    fig = plt.figure(figsize=figsize)
    upset.plot(fig=fig)

    # UpSet.plot() may override the figure size internally, so re-apply.
    fig.set_size_inches(figsize)

    return fig
