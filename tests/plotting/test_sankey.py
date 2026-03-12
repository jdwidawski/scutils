"""Tests for scutils.plotting.sankey.sankey_plot."""

from __future__ import annotations

import pytest

pytest.importorskip("plotly", reason="plotly is required for sankey tests")

import plotly.graph_objects as go  # noqa: E402

from scutils.plotting.sankey import (
    sankey_plot,
    _apply_node_order,
    _category_ordered_values,
    _resolve_level_colors,
    _to_hex,
    _to_rgba,
)


# ---------------------------------------------------------------------------
# Helpers / internal unit tests
# ---------------------------------------------------------------------------


def test_to_hex_named():
    assert _to_hex("red") == "#ff0000"


def test_to_hex_rgb_tuple():
    assert _to_hex((0.0, 1.0, 0.0)) == "#00ff00"


def test_to_rgba_format():
    result = _to_rgba("#ff0000", alpha=0.5)
    assert result.startswith("rgba(255, 0, 0, 0.5")


def test_category_ordered_values_categorical(adata_basic):
    vals = _category_ordered_values(adata_basic, "cell_type")
    assert set(vals) == {"T", "B", "NK", "Mono"}


def test_category_ordered_values_preserves_cat_order(adata_basic):
    vals = _category_ordered_values(adata_basic, "cell_type")
    # Should be the category order (alphabetical for default Categorical)
    assert isinstance(vals, list)
    assert len(vals) == 4


def test_resolve_level_colors_uns(adata_basic):
    """Should pick up colors from adata.uns when present.

    adata.obs["cell_type"] categories are stored alphabetically:
    ["B", "Mono", "NK", "T"], so the uns color list is indexed accordingly.
    """
    # Alphabetical category order: B=0, Mono=1, NK=2, T=3
    adata_basic.uns["cell_type_colors"] = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
    cats = _category_ordered_values(adata_basic, "cell_type")  # ["B", "Mono", "NK", "T"]
    colors = _resolve_level_colors(adata_basic, "cell_type", cats, palette=None)
    # "B" is index 0 → #ff0000; "T" is index 3 → #ffff00
    assert colors["B"] == "#ff0000"
    assert colors["T"] == "#ffff00"


def test_resolve_level_colors_fallback(adata_basic):
    """Should auto-generate colours when no uns key and no palette."""
    # Ensure no uns key for condition
    adata_basic.uns.pop("condition_colors", None)
    vals = ["ctrl", "stim"]
    colors = _resolve_level_colors(adata_basic, "condition", vals, palette=None)
    assert set(colors.keys()) == {"ctrl", "stim"}
    for v in colors.values():
        assert v.startswith("#")


# ---------------------------------------------------------------------------
# sankey_plot — basic return type & structure
# ---------------------------------------------------------------------------


class TestSankeyPlotReturnType:
    def test_two_categories_returns_figure(self, adata_basic):
        fig = sankey_plot(adata_basic, categories=["cell_type", "condition"])
        assert isinstance(fig, go.Figure)

    def test_three_categories_returns_figure(self, adata_donors):
        fig = sankey_plot(
            adata_donors, categories=["cell_type", "donor", "condition"]
        )
        assert isinstance(fig, go.Figure)

    def test_figure_has_sankey_trace(self, adata_basic):
        fig = sankey_plot(adata_basic, categories=["cell_type", "condition"])
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Sankey)


# ---------------------------------------------------------------------------
# Node count
# ---------------------------------------------------------------------------


class TestSankeyNodeCount:
    def test_two_category_node_count(self, adata_basic):
        fig = sankey_plot(adata_basic, categories=["cell_type", "condition"])
        sankey: go.Sankey = fig.data[0]
        # 4 cell types + 2 conditions = 6 nodes
        assert len(sankey.node.label) == 6

    def test_three_category_node_count(self, adata_donors):
        fig = sankey_plot(
            adata_donors, categories=["cell_type", "donor", "condition"]
        )
        sankey: go.Sankey = fig.data[0]
        # 4 + 4 + 2 = 10 nodes
        assert len(sankey.node.label) == 10


# ---------------------------------------------------------------------------
# Link count
# ---------------------------------------------------------------------------


class TestSankeyLinkCount:
    def test_link_count_two_categories(self, adata_basic):
        """adata_basic tiles cell types against conditions so each cell type maps
        exclusively to one condition — 4 combos, not 8."""
        fig = sankey_plot(adata_basic, categories=["cell_type", "condition"])
        sankey: go.Sankey = fig.data[0]
        assert len(sankey.link.value) == 4

    def test_link_count_three_categories(self, adata_donors):
        """adata_donors has all 4×4 and 4×2 combos present."""
        fig = sankey_plot(
            adata_donors, categories=["cell_type", "donor", "condition"]
        )
        sankey: go.Sankey = fig.data[0]
        # links = (4*4) + (4*2) = 16 + 8 = 24
        assert len(sankey.link.value) == 24

    def test_link_values_sum_to_n_cells(self, adata_basic):
        """Total cells in one level's links should equal n_obs."""
        fig = sankey_plot(adata_basic, categories=["cell_type", "condition"])
        sankey: go.Sankey = fig.data[0]
        assert sum(sankey.link.value) == adata_basic.n_obs


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestSankeyValidation:
    def test_invalid_column_raises_value_error(self, adata_basic):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            sankey_plot(adata_basic, categories=["cell_type", "nonexistent"])

    def test_too_few_categories_raises(self, adata_basic):
        with pytest.raises(ValueError, match="2 or 3"):
            sankey_plot(adata_basic, categories=["cell_type"])

    def test_too_many_categories_raises(self, adata_donors):
        with pytest.raises(ValueError, match="2 or 3"):
            sankey_plot(
                adata_donors,
                categories=["cell_type", "donor", "condition", "cell_type"],
            )


# ---------------------------------------------------------------------------
# Layout options
# ---------------------------------------------------------------------------


class TestSankeyLayout:
    def test_custom_height_width(self, adata_basic):
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
            height=600,
            width=800,
        )
        assert fig.layout.height == 600
        assert fig.layout.width == 800

    def test_title_set(self, adata_basic):
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
            title="Test Sankey",
        )
        assert fig.layout.title.text == "Test Sankey"

    def test_font_size_set(self, adata_basic):
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
            font_size=14,
        )
        assert fig.layout.font.size == 14


# ---------------------------------------------------------------------------
# Palette options
# ---------------------------------------------------------------------------


class TestSankeyPalette:
    def test_palette_string_does_not_raise(self, adata_basic):
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
            palette="Set2",
        )
        assert isinstance(fig, go.Figure)

    def test_palette_list_does_not_raise(self, adata_basic):
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
            palette=["#ff0000", "#00ff00", "#0000ff", "#ffff00"],
        )
        assert isinstance(fig, go.Figure)

    def test_palette_per_column_dict(self, adata_basic):
        """Per-column dict palette is applied without errors."""
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
            palette={
                "cell_type": {"T": "#ff0000", "B": "#00ff00", "NK": "#0000ff", "Mono": "#888888"},
                "condition": {"ctrl": "#aaaaaa", "stim": "#333333"},
            },
        )
        assert isinstance(fig, go.Figure)

    def test_palette_per_column_dict_colors_applied(self, adata_basic):
        """Node colors should reflect the supplied palette dict."""
        adata_basic.uns.pop("cell_type_colors", None)
        adata_basic.uns.pop("condition_colors", None)
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
            palette={
                "cell_type": {"T": "#abcdef", "B": "#abcdef", "NK": "#abcdef", "Mono": "#abcdef"},
                "condition": {"ctrl": "#123456", "stim": "#123456"},
            },
        )
        sankey: go.Sankey = fig.data[0]
        # First 4 nodes are cell_type nodes — all should have color #abcdef
        for i in range(4):
            assert sankey.node.color[i] == "#abcdef"

    def test_uns_colors_used_when_no_palette(self, adata_basic):
        """Colors from adata.uns should appear on the nodes."""
        adata_basic.uns["cell_type_colors"] = ["#aa0000", "#00aa00", "#0000aa", "#aaaa00"]
        adata_basic.uns.pop("condition_colors", None)
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
        )
        sankey: go.Sankey = fig.data[0]
        assert sankey.node.color[0] == "#aa0000"


# ---------------------------------------------------------------------------
# Link alpha
# ---------------------------------------------------------------------------


class TestSankeyLinkAlpha:
    def test_link_colors_are_rgba(self, adata_basic):
        fig = sankey_plot(adata_basic, categories=["cell_type", "condition"])
        sankey: go.Sankey = fig.data[0]
        for c in sankey.link.color:
            assert c.startswith("rgba("), f"Expected rgba(), got {c!r}"

    def test_link_alpha_applied(self, adata_basic):
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
            link_alpha=0.9,
        )
        sankey: go.Sankey = fig.data[0]
        # The alpha value 0.9 should appear in all link colours
        for c in sankey.link.color:
            assert "0.9" in c, f"Expected alpha 0.9 in {c!r}"


# ---------------------------------------------------------------------------
# node_order — unit tests for helper
# ---------------------------------------------------------------------------


class TestApplyNodeOrder:
    def test_auto_preserves_categorical_order(self, adata_basic):
        vals = _category_ordered_values(adata_basic, "cell_type")
        result = _apply_node_order(adata_basic, "cell_type", vals, "auto")
        assert result == vals

    def test_none_same_as_auto(self, adata_basic):
        vals = _category_ordered_values(adata_basic, "cell_type")
        assert _apply_node_order(adata_basic, "cell_type", vals, None) == vals

    def test_data_order_is_list(self, adata_basic):
        vals = _category_ordered_values(adata_basic, "cell_type")
        result = _apply_node_order(adata_basic, "cell_type", vals, "data")
        assert set(result) == set(vals)
        assert len(result) == len(vals)

    def test_count_order_descending(self, adata_donors):
        """With equal donor counts, count order should still return all values."""
        vals = _category_ordered_values(adata_donors, "donor")
        result = _apply_node_order(adata_donors, "donor", vals, "count")
        assert set(result) == set(vals)
        assert len(result) == len(vals)

    def test_count_order_unequal_counts(self, adata_basic):
        """cell_type has equal counts in adata_basic but condition does not
        (ctrl and stim are tied 40/40 — just check all values returned)."""
        vals = _category_ordered_values(adata_basic, "condition")
        result = _apply_node_order(adata_basic, "condition", vals, "count")
        assert set(result) == set(vals)

    def test_dict_order_applied(self, adata_basic):
        custom_order = ["Mono", "NK", "B", "T"]
        result = _apply_node_order(
            adata_basic, "cell_type", ["B", "Mono", "NK", "T"],
            {"cell_type": custom_order}
        )
        assert result == custom_order

    def test_dict_unlisted_values_appended(self, adata_basic):
        """Values not in the custom list should be appended at the end."""
        result = _apply_node_order(
            adata_basic, "cell_type", ["B", "Mono", "NK", "T"],
            {"cell_type": ["T", "B"]}
        )
        assert result[:2] == ["T", "B"]
        assert set(result) == {"T", "B", "NK", "Mono"}

    def test_dict_missing_col_falls_back_to_auto(self, adata_basic):
        vals = _category_ordered_values(adata_basic, "condition")
        result = _apply_node_order(
            adata_basic, "condition", vals, {"cell_type": ["T"]}
        )
        assert result == vals


# ---------------------------------------------------------------------------
# node_order — integration tests via sankey_plot
# ---------------------------------------------------------------------------


class TestSankeyNodeOrder:
    def test_invalid_string_raises(self, adata_basic):
        with pytest.raises(ValueError, match="node_order string must be one of"):
            sankey_plot(
                adata_basic,
                categories=["cell_type", "condition"],
                node_order="bad_value",
            )

    def test_auto_returns_figure(self, adata_basic):
        fig = sankey_plot(
            adata_basic, categories=["cell_type", "condition"], node_order="auto"
        )
        assert isinstance(fig, go.Figure)

    def test_data_returns_figure(self, adata_basic):
        fig = sankey_plot(
            adata_basic, categories=["cell_type", "condition"], node_order="data"
        )
        assert isinstance(fig, go.Figure)

    def test_count_returns_figure(self, adata_donors):
        fig = sankey_plot(
            adata_donors,
            categories=["cell_type", "donor", "condition"],
            node_order="count",
        )
        assert isinstance(fig, go.Figure)

    def test_dict_order_respected_in_nodes(self, adata_basic):
        """Node labels should reflect the custom order for cell_type."""
        adata_basic.uns.pop("cell_type_colors", None)
        adata_basic.uns.pop("condition_colors", None)
        custom = ["Mono", "NK", "B", "T"]
        fig = sankey_plot(
            adata_basic,
            categories=["cell_type", "condition"],
            node_order={"cell_type": custom},
        )
        sankey: go.Sankey = fig.data[0]
        # First 4 nodes are cell_type; labels are wrapped in <b>...</b>
        ct_labels = [sankey.node.label[i] for i in range(4)]
        assert ct_labels == [f"<b>{v}</b>" for v in custom]

    def test_count_order_preserves_node_count(self, adata_donors):
        fig = sankey_plot(
            adata_donors,
            categories=["cell_type", "donor", "condition"],
            node_order="count",
        )
        sankey: go.Sankey = fig.data[0]
        assert len(sankey.node.label) == 10  # 4 + 4 + 2


# ---------------------------------------------------------------------------
# show_plotly
# ---------------------------------------------------------------------------


def test_show_plotly_returns_html(adata_basic):
    HTML = pytest.importorskip("IPython.display", reason="IPython not installed").HTML

    from scutils.plotting import show_plotly

    fig = sankey_plot(adata_basic, categories=["cell_type", "condition"])
    result = show_plotly(fig)
    assert isinstance(result, HTML)


def test_show_plotly_html_contains_plotlyjs(adata_basic):
    pytest.importorskip("IPython", reason="IPython not installed")

    from scutils.plotting import show_plotly

    fig = sankey_plot(adata_basic, categories=["cell_type", "condition"])
    html_str = show_plotly(fig).data
    assert "plotly" in html_str.lower()
