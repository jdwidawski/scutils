"""Shared internal helpers used across scutils.plotting submodules."""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Union

import matplotlib
import seaborn as sns


def _resolve_palette(
    palette: Optional[Union[str, List[str], Dict[str, str]]],
    categories: List[str],
) -> Optional[Dict[str, str]]:
    """Normalise a palette argument into a ``{category: colour}`` dict.

    Args:
        palette: Colour specification. Accepts:

            - ``None``: returns ``None`` (seaborn default palette).
            - A single colour string: all categories share that colour.
            - A list of colours: cycled to cover all categories.
            - A dict mapping category labels to colours.
            - A seaborn / matplotlib palette name string.

        categories: Ordered list of category labels.

    Returns:
        A ``{category: colour}`` mapping, or ``None`` to use seaborn
        defaults.
    """
    if palette is None:
        return None
    if isinstance(palette, dict):
        return palette
    if isinstance(palette, str) and matplotlib.colors.is_color_like(palette):
        return {c: palette for c in categories}
    if isinstance(palette, str):
        cmap = sns.color_palette(palette, n_colors=len(categories))
        return {c: cmap[i] for i, c in enumerate(categories)}
    if isinstance(palette, (list, tuple)):
        cycled = list(
            itertools.islice(itertools.cycle(palette), len(categories))
        )
        return {c: cycled[i] for i, c in enumerate(categories)}
    return None
