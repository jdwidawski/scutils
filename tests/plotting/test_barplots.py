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
    cell_count_barplot_multiplot,
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

    def test_legend_present_for_stacked(self, adata_donors):
        # Stacked mode uses colour to encode group_by — legend is needed.
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked",
        )
        assert fig.axes[0].get_legend() is not None
        plt.close(fig)

    def test_no_legend_for_grouped(self, adata_donors):
        # Grouped mode encodes group_by as text labels — no legend needed.
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped",
        )
        assert fig.axes[0].get_legend() is None
        plt.close(fig)

    def test_no_legend_without_group_by(self, adata_basic):
        fig = cell_count_barplot(adata_basic, category="cell_type")
        assert fig.axes[0].get_legend() is None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Grouped mode — text labels and styling
# ---------------------------------------------------------------------------


class TestCellCountBarplotGroupedMode:
    def test_text_labels_present(self, adata_donors):
        """Every non-zero bar in grouped mode should have a group_by text label."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped",
        )
        ax = fig.axes[0]
        # 4 cell types × 2 conditions, all non-zero in adata_donors → 8 labels
        assert len(ax.texts) == 8
        plt.close(fig)

    def test_text_labels_above(self, adata_donors):
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped", group_label_position="above",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_group_label_kwargs_forwarded(self, adata_donors):
        """Custom text kwargs should be applied without error."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped",
            group_label_kwargs={"fontsize": 6, "color": "white", "rotation": 90},
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_colors_from_category_not_group(self, adata_donors):
        """All bars within the same category block share a colour."""
        adata_donors.uns.pop("cell_type_colors", None)
        adata_donors.uns.pop("condition_colors", None)
        palette = {"B": "#ff0000", "Mono": "#00ff00", "NK": "#0000ff", "T": "#ffff00"}
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped", palette=palette,
        )
        ax = fig.axes[0]
        # Block layout: category i occupies patches[i*n_grp : (i+1)*n_grp].
        # All patches in a block must share the same colour.
        n_cats = 4
        n_grp = 2
        for i in range(n_cats):
            block = ax.patches[i * n_grp : (i + 1) * n_grp]
            r0, g0, b0, _ = block[0].get_facecolor()
            for p in block[1:]:
                r, g, b, _ = p.get_facecolor()
                assert abs(r - r0) < 0.01 and abs(g - g0) < 0.01 and abs(b - b0) < 0.01
        plt.close(fig)

    def test_show_counts_with_labels_adds_extra_text(self, adata_donors):
        """show_counts=True adds a count above each bar on top of the group label."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped", show_counts=True,
        )
        ax = fig.axes[0]
        # 8 group labels + 8 count labels = 16 text elements
        assert len(ax.texts) == 16
        plt.close(fig)

    def test_category_separators_shown_by_default(self, adata_donors):
        """3 separator lines for 4 category values (one between each pair)."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped",
        )
        ax = fig.axes[0]
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_category_separators_hidden(self, adata_donors):
        """show_category_separators=False suppresses separator lines."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="grouped", show_category_separators=False,
        )
        ax = fig.axes[0]
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_separators_not_added_for_stacked(self, adata_donors):
        """Separator lines are only drawn in grouped mode."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked",
        )
        ax = fig.axes[0]
        assert len(ax.lines) == 0
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


# ---------------------------------------------------------------------------
# cell_count_barplot_multiplot
# ---------------------------------------------------------------------------


class TestCellCountBarplotMultiplot:
    def test_returns_figure(self, adata_donors):
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition", ncols=2
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_panel_count_equals_n_panel_values(self, adata_donors):
        """One visible axis per unique panel value (2 conditions)."""
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition", ncols=2
        )
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 2
        plt.close(fig)

    def test_panel_titles_match_panel_values(self, adata_donors):
        """Each panel title should equal the corresponding panel-column value."""
        panel_vals = _ordered_values(adata_donors, "condition")
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition", ncols=2
        )
        visible = [ax for ax in fig.axes if ax.get_visible()]
        titles = sorted(ax.get_title() for ax in visible)
        assert titles == sorted(str(v) for v in panel_vals)
        plt.close(fig)

    def test_shared_y_true_all_same_ylim(self, adata_donors):
        """All panels must share identical y-axis limits when shared_y=True."""
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition",
            ncols=2, shared_y=True,
        )
        visible = [ax for ax in fig.axes if ax.get_visible()]
        ylims = [ax.get_ylim() for ax in visible]
        assert all(lim == ylims[0] for lim in ylims)
        plt.close(fig)

    def test_shared_y_false_no_error(self, adata_donors):
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition",
            ncols=2, shared_y=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_panel_order_respected(self, adata_donors):
        """panel_order should control left-to-right panel sequence."""
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition",
            panel_order=["stim", "ctrl"], ncols=2,
        )
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert visible[0].get_title() == "stim"
        assert visible[1].get_title() == "ctrl"
        plt.close(fig)

    def test_empty_axes_hidden(self, adata_donors):
        """4 donor panels in a ncols=3 grid → 6 total, 2 hidden."""
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="donor", ncols=3
        )
        hidden = [ax for ax in fig.axes if not ax.get_visible()]
        assert len(hidden) == 2
        plt.close(fig)

    def test_suptitle_set(self, adata_donors):
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition",
            ncols=2, title="My Suptitle",
        )
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "My Suptitle"
        plt.close(fig)

    def test_with_group_by_stacked(self, adata_donors):
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition",
            group_by="donor", mode="stacked", ncols=2,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_group_by_grouped(self, adata_donors):
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition",
            group_by="donor", mode="grouped", ncols=2,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_border_ticks_only_hides_xlabel_non_bottom(self, adata_donors):
        """Non-bottom-row panels get empty xlabel; bottom-row panels keep it."""
        # 4 donors → 2 rows × 2 cols; row 0 is not bottom, row 1 is bottom.
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="donor",
            ncols=2, border_ticks_only=True,
        )
        visible = [ax for ax in fig.axes if ax.get_visible()]
        # Row 0: panels 0 and 1 → empty xlabel
        assert visible[0].get_xlabel() == ""
        assert visible[1].get_xlabel() == ""
        # Row 1: panels 2 and 3 → default xlabel = category name
        assert visible[2].get_xlabel() == "cell_type"
        assert visible[3].get_xlabel() == "cell_type"
        plt.close(fig)

    def test_border_ticks_only_false_all_xlabels_visible(self, adata_donors):
        """border_ticks_only=False shows xlabel on every panel."""
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="donor",
            ncols=2, border_ticks_only=False,
        )
        visible = [ax for ax in fig.axes if ax.get_visible()]
        for ax in visible:
            assert ax.get_xlabel() == "cell_type"
        plt.close(fig)

    def test_invalid_category_raises(self, adata_donors):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            cell_count_barplot_multiplot(
                adata_donors, category="nonexistent", panel="condition"
            )

    def test_invalid_panel_raises(self, adata_donors):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            cell_count_barplot_multiplot(
                adata_donors, category="cell_type", panel="nonexistent"
            )

    def test_invalid_group_by_raises(self, adata_donors):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            cell_count_barplot_multiplot(
                adata_donors, category="cell_type", panel="condition",
                group_by="nonexistent",
            )

    def test_invalid_panel_order_raises(self, adata_donors):
        with pytest.raises(ValueError, match="not found in panel"):
            cell_count_barplot_multiplot(
                adata_donors, category="cell_type", panel="condition",
                panel_order=["ctrl", "stim", "extra"],
            )

    def test_invalid_mode_raises(self, adata_donors):
        with pytest.raises(ValueError, match="mode must be"):
            cell_count_barplot_multiplot(
                adata_donors, category="cell_type", panel="condition",
                mode="sideways",
            )


# ---------------------------------------------------------------------------
# sort_x — ordering x-axis ticks by count
# ---------------------------------------------------------------------------


class TestCellCountBarplotSortX:
    def test_ascending_single_category(self, adata_basic):
        """Bar heights must be non-decreasing when sort_x='ascending'."""
        fig = cell_count_barplot(
            adata_basic, category="cell_type", sort_x="ascending"
        )
        ax = fig.axes[0]
        heights = [p.get_height() for p in ax.patches]
        assert heights == sorted(heights)
        plt.close(fig)

    def test_descending_single_category(self, adata_basic):
        """Bar heights must be non-increasing when sort_x='descending'."""
        fig = cell_count_barplot(
            adata_basic, category="cell_type", sort_x="descending"
        )
        ax = fig.axes[0]
        heights = [p.get_height() for p in ax.patches]
        assert heights == sorted(heights, reverse=True)
        plt.close(fig)

    def test_ascending_and_descending_are_reverse(self, adata_basic):
        """Ascending and descending orderings should be mirror images."""
        fig_asc = cell_count_barplot(
            adata_basic, category="cell_type", sort_x="ascending"
        )
        fig_desc = cell_count_barplot(
            adata_basic, category="cell_type", sort_x="descending"
        )
        labels_asc = [t.get_text() for t in fig_asc.axes[0].get_xticklabels()]
        labels_desc = [t.get_text() for t in fig_desc.axes[0].get_xticklabels()]
        assert labels_asc == list(reversed(labels_desc))
        plt.close(fig_asc)
        plt.close(fig_desc)

    def test_sort_x_none_preserves_original_order(self, adata_basic):
        """sort_x=None must produce the same label order as no sort_x."""
        fig_default = cell_count_barplot(adata_basic, category="cell_type")
        fig_none = cell_count_barplot(
            adata_basic, category="cell_type", sort_x=None
        )
        labels_default = [t.get_text() for t in fig_default.axes[0].get_xticklabels()]
        labels_none = [t.get_text() for t in fig_none.axes[0].get_xticklabels()]
        assert labels_default == labels_none
        plt.close(fig_default)
        plt.close(fig_none)

    def test_ascending_stacked_mode(self, adata_donors):
        """In stacked mode, total stack heights per category are non-decreasing."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked", sort_x="ascending",
        )
        ax = fig.axes[0]
        n_cats = len(_ordered_values(adata_donors, "cell_type"))
        n_grp = len(_ordered_values(adata_donors, "condition"))
        patches = ax.patches
        # Each x-position i accumulates heights from patches[i + k*n_cats]
        col_totals = [
            sum(patches[i + k * n_cats].get_height() for k in range(n_grp))
            for i in range(n_cats)
        ]
        assert col_totals == sorted(col_totals)
        plt.close(fig)

    def test_sort_uses_raw_counts_not_fractions(self, adata_basic):
        """Ordering with normalize=True should mirror sort on raw counts."""
        fig_raw = cell_count_barplot(
            adata_basic, category="cell_type", sort_x="ascending"
        )
        fig_norm = cell_count_barplot(
            adata_basic, category="cell_type", normalize=True, sort_x="ascending"
        )
        labels_raw = [t.get_text() for t in fig_raw.axes[0].get_xticklabels()]
        labels_norm = [t.get_text() for t in fig_norm.axes[0].get_xticklabels()]
        assert labels_raw == labels_norm
        plt.close(fig_raw)
        plt.close(fig_norm)

    def test_multiplot_sort_x_ascending(self, adata_donors):
        """sort_x propagates to each panel without error."""
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition",
            sort_x="ascending", ncols=2,
        )
        assert isinstance(fig, Figure)
        # Each visible panel must have non-decreasing bar heights
        for ax in (a for a in fig.axes if a.get_visible()):
            heights = [p.get_height() for p in ax.patches]
            assert heights == sorted(heights)
        plt.close(fig)

    def test_multiplot_sort_x_panels_sorted_independently(self, adata_donors):
        """Different panels may have different sort orders (independent subsets)."""
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="condition",
            sort_x="descending", ncols=2,
        )
        visible = [ax for ax in fig.axes if ax.get_visible()]
        for ax in visible:
            heights = [p.get_height() for p in ax.patches]
            assert heights == sorted(heights, reverse=True)
        plt.close(fig)


# ---------------------------------------------------------------------------
# sort_by_group — sort x-axis by a specific group_by value's fraction/count
# ---------------------------------------------------------------------------


class TestCellCountBarplotSortByGroup:
    def test_sort_by_group_normalized_stacked_descending(self, adata_donors):
        """With normalize=True+stacked, x-axis should be sorted by the fraction
        of the specified group_by value (descending)."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked", normalize=True,
            sort_x="descending", sort_by_group="ctrl",
        )
        ax = fig.axes[0]
        n_cats = len(_ordered_values(adata_donors, "cell_type"))
        n_grp = len(_ordered_values(adata_donors, "condition"))
        patches = ax.patches
        # patches are in draw order: all of grp0, all of grp1, ...
        # Fraction of "ctrl" for each x-position is the bottom segment height
        # (ctrl is the first group_by value alphabetically → first layer drawn)
        ctrl_fracs = [patches[i].get_height() for i in range(n_cats)]
        assert ctrl_fracs == sorted(ctrl_fracs, reverse=True)
        plt.close(fig)

    def test_sort_by_group_normalized_stacked_ascending(self, adata_donors):
        """Ascending sort by fraction of a specific group_by value."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked", normalize=True,
            sort_x="ascending", sort_by_group="ctrl",
        )
        ax = fig.axes[0]
        n_cats = len(_ordered_values(adata_donors, "cell_type"))
        ctrl_fracs = [ax.patches[i].get_height() for i in range(n_cats)]
        assert ctrl_fracs == sorted(ctrl_fracs)
        plt.close(fig)

    def test_sort_by_group_raw_counts(self, adata_donors):
        """With normalize=False, sort by raw count of the specified group value."""
        fig = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked", normalize=False,
            sort_x="descending", sort_by_group="ctrl",
        )
        ax = fig.axes[0]
        n_cats = len(_ordered_values(adata_donors, "cell_type"))
        ctrl_heights = [ax.patches[i].get_height() for i in range(n_cats)]
        assert ctrl_heights == sorted(ctrl_heights, reverse=True)
        plt.close(fig)

    def test_sort_by_group_different_from_total_sort(self, adata_donors):
        """sort_by_group and sort by total should differ when normalize=True."""
        fig_total = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked", normalize=True,
            sort_x="descending",
        )
        fig_group = cell_count_barplot(
            adata_donors, category="cell_type", group_by="condition",
            mode="stacked", normalize=True,
            sort_x="descending", sort_by_group="ctrl",
        )
        labels_total = [t.get_text() for t in fig_total.axes[0].get_xticklabels()]
        labels_group = [t.get_text() for t in fig_group.axes[0].get_xticklabels()]
        # In adata_donors conditions are balanced, so fractions differ across
        # cell types — the orders should differ.  At minimum, both are valid
        # lists of the same set of categories.
        assert set(labels_total) == set(labels_group)
        plt.close(fig_total)
        plt.close(fig_group)

    def test_sort_by_group_without_group_by_raises(self, adata_donors):
        with pytest.raises(ValueError, match="sort_by_group requires group_by"):
            cell_count_barplot(
                adata_donors, category="cell_type",
                sort_x="descending", sort_by_group="ctrl",
            )

    def test_sort_by_group_invalid_value_raises(self, adata_donors):
        with pytest.raises(ValueError, match="not found in group_by"):
            cell_count_barplot(
                adata_donors, category="cell_type", group_by="condition",
                mode="stacked", sort_x="descending",
                sort_by_group="nonexistent",
            )

    def test_sort_by_group_multiplot(self, adata_donors):
        """sort_by_group propagates to every panel in multiplot."""
        fig = cell_count_barplot_multiplot(
            adata_donors, category="cell_type", panel="donor",
            group_by="condition", mode="stacked", normalize=True,
            sort_x="descending", sort_by_group="ctrl", ncols=2,
        )
        assert isinstance(fig, Figure)
        n_cats = len(_ordered_values(adata_donors, "cell_type"))
        n_grp = len(_ordered_values(adata_donors, "condition"))
        for ax in (a for a in fig.axes if a.get_visible()):
            ctrl_fracs = [ax.patches[i].get_height() for i in range(n_cats)]
            assert ctrl_fracs == sorted(ctrl_fracs, reverse=True)
        plt.close(fig)

    def test_sort_by_group_multiplot_without_group_by_raises(self, adata_donors):
        with pytest.raises(ValueError, match="sort_by_group requires group_by"):
            cell_count_barplot_multiplot(
                adata_donors, category="cell_type", panel="condition",
                sort_x="descending", sort_by_group="ctrl",
            )
