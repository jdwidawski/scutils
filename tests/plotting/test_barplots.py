"""Tests for scutils.plotting.barplots.cell_count_barplot."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from scutils.plotting.barplots import (
    cell_count_barplot,
    _bar_colors,
    _ordered_values,
)


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


def test_ordered_values_categorical(adata_basic):
    vals = _ordered_values(adata_basic, "cell_type")
    assert set(vals) == {"T", "B", "NK", "Mono"}
    assert isinstance(vals, list)


def test_bar_colors_uns(adata_basic):
    cats = _ordered_values(adata_basic, "cell_type")  # ["B", "Mono", "NK", "T"]
    adata_basic.uns["cell_type_colors"] = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
    colors = _bar_colors(adata_basic, "cell_type", cats, palette=None)
    assert colors[0] == "#ff0000"  # "B" is index 0


def test_bar_colors_palette_dict(adata_basic):
    cats = ["ctrl", "stim"]
    adata_basic.uns.pop("condition_colors", None)
    colors = _bar_colors(
        adata_basic, "condition", cats,
        palette={"ctrl": "#aabbcc", "stim": "#112233"},
    )
    assert colors == ["#aabbcc", "#112233"]


def test_bar_colors_fallback(adata_basic):
    adata_basic.uns.pop("condition_colors", None)
    cats = ["ctrl", "stim"]
    colors = _bar_colors(adata_basic, "condition", cats, palette=None)
    assert len(colors) == 2
    for c in colors:
        assert c.startswith("#")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestCellCountBarplotReturnType:
    def test_single_category_returns_figure(self, adata_basic):
        fig = cell_count_barplot(adata_basic, category="cell_type")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_two_categories_grouped_returns_figure(self, adata_donors):
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_two_categories_stacked_returns_figure(self, adata_donors):
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Bar heights — correctness
# ---------------------------------------------------------------------------


class TestCellCountBarplotHeights:
    def test_single_category_bar_count_equals_n_cells(self, adata_basic):
        """Sum of all bar heights == adata.n_obs."""
        fig = cell_count_barplot(adata_basic, category="cell_type")
        ax = fig.axes[0]
        total = sum(p.get_height() for p in ax.patches)
        assert total == adata_basic.n_obs
        plt.close(fig)

    def test_two_category_grouped_total_equals_n_cells(self, adata_donors):
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped",
        )
        ax = fig.axes[0]
        total = sum(p.get_height() for p in ax.patches)
        assert total == adata_donors.n_obs
        plt.close(fig)

    def test_two_category_stacked_total_equals_n_cells(self, adata_donors):
        """For stacked, the final top of each stack-column sums to n_obs."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked",
        )
        ax = fig.axes[0]
        total = sum(p.get_height() for p in ax.patches)
        assert total == adata_donors.n_obs
        plt.close(fig)

    def test_normalize_single_sums_to_one(self, adata_basic):
        fig = cell_count_barplot(
            adata_basic, category="cell_type", normalize=True
        )
        ax = fig.axes[0]
        total = sum(p.get_height() for p in ax.patches)
        assert abs(total - 1.0) < 1e-6
        plt.close(fig)

    def test_normalize_stacked_each_column_sums_to_one(self, adata_donors):
        """Each stacked column should sum to 1.0 after normalisation."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked", normalize=True,
        )
        ax = fig.axes[0]
        n_cats = len(_ordered_values(adata_donors, "cell_type"))
        n_grp = len(_ordered_values(adata_donors, "condition"))
        patches = ax.patches  # ordered: all bars for grp0, then grp1, ...
        # Sum heights at each x-position across groups
        per_x = np.zeros(n_cats)
        for i, p in enumerate(patches):
            x_idx = i % n_cats
            per_x[x_idx] += p.get_height()
        for s in per_x:
            assert abs(s - 1.0) < 1e-6
        plt.close(fig)


# ---------------------------------------------------------------------------
# Node / bar count
# ---------------------------------------------------------------------------


class TestCellCountBarplotBarCount:
    def test_single_category_bar_count(self, adata_basic):
        """One bar per unique cell_type value (4)."""
        fig = cell_count_barplot(adata_basic, category="cell_type")
        ax = fig.axes[0]
        assert len(ax.patches) == 4
        plt.close(fig)

    def test_grouped_bar_count(self, adata_donors):
        """4 cell types × 2 conditions = 8 bars."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped",
        )
        ax = fig.axes[0]
        assert len(ax.patches) == 8
        plt.close(fig)

    def test_stacked_bar_count(self, adata_donors):
        """4 cell types × 2 stacked segments = 8 patches."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked",
        )
        ax = fig.axes[0]
        assert len(ax.patches) == 8
        plt.close(fig)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestCellCountBarplotValidation:
    def test_invalid_category_raises(self, adata_basic):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            cell_count_barplot(adata_basic, category="nonexistent")

    def test_invalid_group_by_raises(self, adata_basic):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            cell_count_barplot(
                adata_basic, category="cell_type", group_by="nonexistent"
            )

    def test_invalid_mode_raises(self, adata_basic):
        with pytest.raises(ValueError, match="mode must be"):
            cell_count_barplot(
                adata_basic, category="cell_type",
                group_by="condition", mode="sideways",
            )


# ---------------------------------------------------------------------------
# Layout and formatting
# ---------------------------------------------------------------------------


class TestCellCountBarplotLayout:
    def test_custom_figsize(self, adata_basic):
        fig = cell_count_barplot(
            adata_basic, category="cell_type", figsize=(10, 5)
        )
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1 and abs(h - 5) < 0.1
        plt.close(fig)

    def test_title_set(self, adata_basic):
        fig = cell_count_barplot(
            adata_basic, category="cell_type", title="My Title"
        )
        assert fig.axes[0].get_title() == "My Title"
        plt.close(fig)

    def test_default_ylabel_count(self, adata_basic):
        fig = cell_count_barplot(adata_basic, category="cell_type")
        assert fig.axes[0].get_ylabel() == "Number of cells"
        plt.close(fig)

    def test_default_ylabel_fraction(self, adata_basic):
        fig = cell_count_barplot(
            adata_basic, category="cell_type", normalize=True
        )
        assert fig.axes[0].get_ylabel() == "Fraction of cells"
        plt.close(fig)

    def test_custom_xlabel(self, adata_basic):
        fig = cell_count_barplot(
            adata_basic, category="cell_type", xlabel="Cell Types"
        )
        assert fig.axes[0].get_xlabel() == "Cell Types"
        plt.close(fig)

    def test_legend_present_with_group_by(self, adata_donors):
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition"
        )
        assert fig.axes[0].get_legend() is not None
        plt.close(fig)

    def test_no_legend_without_group_by(self, adata_basic):
        fig = cell_count_barplot(adata_basic, category="cell_type")
        assert fig.axes[0].get_legend() is None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Palette options
# ---------------------------------------------------------------------------


class TestCellCountBarplotPalette:
    def test_palette_string_does_not_raise(self, adata_basic):
        fig = cell_count_barplot(
            adata_basic, category="cell_type", palette="Set2"
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_palette_dict_applied(self, adata_basic):
        adata_basic.uns.pop("cell_type_colors", None)
        # "B" is the first bar (alphabetical category order)
        fig = cell_count_barplot(
            adata_basic, category="cell_type",
            palette={"B": "#abcdef", "Mono": "#abcdef", "NK": "#abcdef", "T": "#abcdef"},
        )
        ax = fig.axes[0]
        for p in ax.patches:
            r, g, b, _ = p.get_facecolor()
            assert abs(r - 0xab / 255) < 0.01
        plt.close(fig)

    def test_uns_colors_used(self, adata_basic):
        adata_basic.uns["cell_type_colors"] = [
            "#ff0000", "#00ff00", "#0000ff", "#ffff00"
        ]
        fig = cell_count_barplot(adata_basic, category="cell_type")
        ax = fig.axes[0]
        # first bar ("B") should be red
        r, g, b, _ = ax.patches[0].get_facecolor()
        assert abs(r - 1.0) < 0.01 and abs(g) < 0.01
        plt.close(fig)


# ---------------------------------------------------------------------------
# show_counts
# ---------------------------------------------------------------------------


class TestCellCountBarplotShowCounts:
    def test_show_counts_adds_text(self, adata_basic):
        fig = cell_count_barplot(
            adata_basic, category="cell_type", show_counts=True
        )
        ax = fig.axes[0]
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_no_counts_by_default(self, adata_basic):
        fig = cell_count_barplot(adata_basic, category="cell_type")
        ax = fig.axes[0]
        assert len(ax.texts) == 0
        plt.close(fig)
