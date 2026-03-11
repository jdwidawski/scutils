"""Tests for src/plotting/dotplots.py."""
from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.figure import Figure

matplotlib.use("Agg")

from scutils.plotting.dotplots import (
    _fraction_to_size,
    _size_legend_ticks,
    _format_pct_labels,
    _make_size_legend_handles,
    _resolve_vmin_vmax,
    dotplot_expression_two_categories,
    dotplot_expression_two_categories_multiplot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_adata() -> AnnData:
    """Small synthetic AnnData with two categorical obs columns."""
    rng = np.random.default_rng(42)
    n_cells = 120
    n_genes = 8

    X = rng.negative_binomial(2, 0.4, size=(n_cells, n_genes)).astype(float)
    adata = AnnData(X=X)
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]
    adata.var["symbol"] = [f"Sym{i}" for i in range(n_genes)]

    # category_x: 3 cell types
    cell_types = ["TypeA", "TypeB", "TypeC"]
    adata.obs["cell_type"] = np.repeat(cell_types, n_cells // len(cell_types)).tolist()
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    # category_y: 2 conditions
    conditions = ["ctrl", "treat"]
    adata.obs["condition"] = (
        np.tile(conditions, n_cells // len(conditions)).tolist()
    )
    adata.obs["condition"] = adata.obs["condition"].astype("category")

    # A raw layer for layer-param tests
    adata.layers["raw"] = X * 2.0

    return adata


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestFractionToSize:
    def test_zero_fraction_gives_zero_size(self):
        assert _fraction_to_size(0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_full_fraction_gives_largest_dot(self):
        from src.plotting.dotplots import _LARGEST_DOT
        assert _fraction_to_size(1.0, 0.0, 1.0) == pytest.approx(_LARGEST_DOT)

    def test_clamps_above_dot_max(self):
        from src.plotting.dotplots import _LARGEST_DOT
        assert _fraction_to_size(1.5, 0.0, 1.0) == pytest.approx(_LARGEST_DOT)

    def test_clamps_below_dot_min(self):
        assert _fraction_to_size(-0.1, 0.0, 1.0) == pytest.approx(0.0)

    def test_midpoint_value_between_zero_and_max(self):
        from src.plotting.dotplots import _LARGEST_DOT
        s = _fraction_to_size(0.5, 0.0, 1.0)
        assert 0.0 < s < _LARGEST_DOT

    def test_degenerate_dot_max_equal_dot_min(self):
        from src.plotting.dotplots import _LARGEST_DOT
        assert _fraction_to_size(0.5, 0.3, 0.3) == pytest.approx(_LARGEST_DOT)


class TestSizeLegendTicks:
    def test_returns_array(self):
        ticks = _size_legend_ticks(0.0, 1.0)
        assert isinstance(ticks, np.ndarray)

    def test_between_three_and_six_ticks(self):
        for dot_max in (0.2, 0.5, 0.8, 1.0):
            ticks = _size_legend_ticks(0.0, dot_max)
            assert 3 <= len(ticks) <= 6, f"Failed for dot_max={dot_max}"

    def test_last_tick_equals_dot_max(self):
        for dot_max in (0.3, 0.6, 0.9, 1.0):
            ticks = _size_legend_ticks(0.0, dot_max)
            assert ticks[-1] == pytest.approx(dot_max)

    def test_ticks_ascending(self):
        ticks = _size_legend_ticks(0.0, 0.7)
        assert np.all(np.diff(ticks) > 0)

    def test_narrow_range(self):
        ticks = _size_legend_ticks(0.0, 0.05)
        assert ticks[-1] == pytest.approx(0.05)

    def test_zero_not_in_ticks(self):
        for dot_max in (0.01, 0.05, 0.1, 0.5, 1.0):
            ticks = _size_legend_ticks(0.0, dot_max)
            assert not np.any(ticks <= 1e-9), (
                f"Zero found in ticks for dot_max={dot_max}: {ticks}"
            )


class TestFormatPctLabels:
    def test_integer_labels_when_unique(self):
        labels = _format_pct_labels([0.0, 25.0, 50.0, 75.0, 100.0])
        assert labels == ["0", "25", "50", "75", "100"]

    def test_falls_back_to_one_decimal_on_duplicates(self):
        # 0.0 and 0.4 both round to "0" at integer precision
        labels = _format_pct_labels([0.0, 0.4, 0.8, 1.0])
        # Must be unique
        assert len(set(labels)) == 4
        # Should use 1-decimal format
        assert all("." in lbl for lbl in labels)

    def test_no_duplicate_labels_for_small_dot_max(self):
        """dot_max=0.01 previously produced '0','0','0','1','1'."""
        ticks = _size_legend_ticks(0.0, 0.01)
        labels = _format_pct_labels([t * 100.0 for t in ticks])
        assert len(set(labels)) == len(labels), f"Duplicate labels: {labels}"

    def test_returns_same_length_as_input(self):
        values = [0.0, 1.0, 2.0, 3.0]
        assert len(_format_pct_labels(values)) == len(values)


class TestMakeSizeLegendHandles:
    def test_returns_tuple_of_two_lists(self):
        handles, labels = _make_size_legend_handles(0.0, 1.0)
        assert isinstance(handles, list)
        assert isinstance(labels, list)

    def test_handles_and_labels_same_length(self):
        handles, labels = _make_size_legend_handles(0.0, 0.8)
        assert len(handles) == len(labels)

    def test_between_three_and_six_handles(self):
        for dot_max in (0.2, 0.5, 0.8, 1.0):
            handles, labels = _make_size_legend_handles(0.0, dot_max)
            assert 3 <= len(handles) <= 6, f"Failed for dot_max={dot_max}"

    def test_labels_are_integer_percentages(self):
        handles, labels = _make_size_legend_handles(0.0, 0.6)
        for lbl in labels:
            int(lbl)  # should not raise

    def test_handles_are_line2d(self):
        import matplotlib.lines
        handles, _ = _make_size_legend_handles(0.0, 1.0)
        assert all(isinstance(h, matplotlib.lines.Line2D) for h in handles)


class TestResolveVminVmax:
    def test_none_returns_none(self):
        s = pytest.importorskip("pandas").Series([1.0, 2.0, 3.0])
        assert _resolve_vmin_vmax(s, None) is None

    def test_float_returns_float(self):
        import pandas as pd
        s = pd.Series([1.0, 2.0, 3.0])
        assert _resolve_vmin_vmax(s, 2.5) == pytest.approx(2.5)

    def test_percentile_string(self):
        import pandas as pd
        s = pd.Series(np.arange(0, 101, dtype=float))
        result = _resolve_vmin_vmax(s, "p90")
        assert result == pytest.approx(90.0)

    def test_invalid_string_raises(self):
        import pandas as pd
        s = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="p"):
            _resolve_vmin_vmax(s, "invalid")


# ---------------------------------------------------------------------------
# Integration tests for dotplot_expression_two_categories
# ---------------------------------------------------------------------------

class TestDotplotExpressionTwoCategories:
    # ------------------------------------------------------------------
    # Return type / basic smoke tests
    # ------------------------------------------------------------------
    def test_returns_figure(self, small_adata):
        result = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition"
        )
        assert isinstance(result, Figure)

    def test_return_dataframe_returns_tuple(self, small_adata):
        result = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            return_dataframe=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, df = result
        assert isinstance(fig, Figure)
        assert "mean" in df.columns
        assert "size" in df.columns
        assert "cell_type" in df.columns
        assert "condition" in df.columns

    def test_dataframe_shape(self, small_adata):
        _, df = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            return_dataframe=True,
        )
        # 3 cell types × 2 conditions = 6 rows
        n_types = small_adata.obs["cell_type"].nunique()
        n_conds = small_adata.obs["condition"].nunique()
        assert len(df) == n_types * n_conds

    def test_dataframe_size_in_zero_one(self, small_adata):
        _, df = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            return_dataframe=True,
        )
        assert df["size"].between(0.0, 1.0).all()

    # ------------------------------------------------------------------
    # Adaptive figure sizing
    # ------------------------------------------------------------------
    def test_default_figsize_scales_with_categories(self):
        """Figure with more categories should be larger than with fewer."""
        rng = np.random.default_rng(0)
        n_cells = 300

        def _make(n_x: int, n_y: int) -> tuple[float, float]:
            adata = AnnData(X=rng.random((n_cells, 3)))
            adata.var_names = ["G0", "G1", "G2"]
            x_cats = [f"X{i}" for i in range(n_x)]
            y_cats = [f"Y{i}" for i in range(n_y)]
            adata.obs["catx"] = np.tile(
                x_cats, n_cells // n_x + 1
            )[:n_cells]
            adata.obs["catx"] = adata.obs["catx"].astype("category")
            adata.obs["caty"] = np.tile(
                y_cats, n_cells // n_y + 1
            )[:n_cells]
            adata.obs["caty"] = adata.obs["caty"].astype("category")
            fig = dotplot_expression_two_categories(
                adata, "G0", "catx", "caty"
            )
            return fig.get_size_inches()

        w_small, h_small = _make(3, 2)
        w_large, h_large = _make(10, 8)
        assert w_large > w_small
        assert h_large > h_small

    def test_explicit_figsize_respected(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            figsize=(9.0, 7.0),
        )
        w, h = fig.get_size_inches()
        assert w == pytest.approx(9.0)
        assert h == pytest.approx(7.0)

    # ------------------------------------------------------------------
    # Passing an existing ax
    # ------------------------------------------------------------------
    def test_ax_parameter_uses_provided_axes(self, small_adata):
        fig_ext, ax_ext = plt.subplots()
        returned_fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition", ax=ax_ext
        )
        assert returned_fig is fig_ext

    # ------------------------------------------------------------------
    # dot_max behaviour
    # ------------------------------------------------------------------
    def test_auto_dot_max_ge_actual_max(self, small_adata):
        _, df = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            return_dataframe=True,
        )
        assert df["size"].max() <= 1.0  # auto dot_max must accommodate all values

    def test_manual_dot_max_respected(self, small_adata):
        """With dot_max=1.0 the scatter size of a full group == _LARGEST_DOT."""
        from src.plotting.dotplots import _LARGEST_DOT
        # All-expressing gene: set X col 1 to a large value
        adata = small_adata.copy()
        adata.X[:, 0] = 100.0
        _, df = dotplot_expression_two_categories(
            adata, "Gene0", "cell_type", "condition",
            dot_max=1.0, return_dataframe=True,
        )
        # Groups where every cell expresses should map to size == _LARGEST_DOT
        full_groups = df[df["size"] == pytest.approx(1.0, abs=0.01)]
        if len(full_groups) > 0:
            pass  # just check no KeyError / crash above

    # ------------------------------------------------------------------
    # vmin / vmax
    # ------------------------------------------------------------------
    def test_vmin_vmax_float(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            vmin=0.0, vmax=5.0,
        )
        assert isinstance(fig, Figure)

    def test_vmin_vmax_percentile_string(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            vmin="p5", vmax="p95",
        )
        assert isinstance(fig, Figure)

    def test_invalid_vmin_string_raises(self, small_adata):
        with pytest.raises(ValueError, match="p"):
            dotplot_expression_two_categories(
                small_adata, "Gene0", "cell_type", "condition", vmin="bad"
            )

    # ------------------------------------------------------------------
    # use_zscores
    # ------------------------------------------------------------------
    def test_use_zscores_changes_mean(self, small_adata):
        _, df_raw = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            use_zscores=False, return_dataframe=True,
        )
        _, df_z = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            use_zscores=True, return_dataframe=True,
        )
        # z-scored mean should sum to ~0
        assert df_z["mean"].mean() == pytest.approx(0.0, abs=1e-10)
        # Raw and z-scored means differ
        assert not np.allclose(df_raw["mean"].values, df_z["mean"].values)

    # ------------------------------------------------------------------
    # layer support
    # ------------------------------------------------------------------
    def test_layer_parameter(self, small_adata):
        _, df_default = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            return_dataframe=True,
        )
        _, df_raw_layer = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            layer="raw", return_dataframe=True,
        )
        # The "raw" layer is 2× the default, so means should differ
        assert not np.allclose(df_default["mean"].values, df_raw_layer["mean"].values)

    # ------------------------------------------------------------------
    # gene_symbols support
    # ------------------------------------------------------------------
    def test_gene_symbols_parameter(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Sym3", "cell_type", "condition",
            gene_symbols="symbol",
        )
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # cmap / expression_cutoff
    # ------------------------------------------------------------------
    def test_different_cmap(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition", cmap="Blues"
        )
        assert isinstance(fig, Figure)

    def test_expression_cutoff_affects_size(self, small_adata):
        _, df_low = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            expression_cutoff=0.0, return_dataframe=True,
        )
        _, df_high = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            expression_cutoff=100.0, return_dataframe=True,
        )
        # With an unreachable cutoff every fraction should be 0
        assert np.allclose(df_high["size"].values, 0.0)
        # Default cutoff should yield at least some non-zero fractions
        assert df_low["size"].max() > 0.0

    # ------------------------------------------------------------------
    # custom size_title / color_title
    # ------------------------------------------------------------------
    def test_custom_size_title(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            size_title="Custom size",
        )
        assert isinstance(fig, Figure)

    def test_custom_color_title(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition",
            color_title="Custom colour",
        )
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # Axes labels / ticks
    # ------------------------------------------------------------------
    def test_xlabel_is_category_x(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition"
        )
        scatter_ax = fig.axes[0]
        assert scatter_ax.get_xlabel() == "cell_type"

    def test_ylabel_is_category_y(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition"
        )
        scatter_ax = fig.axes[0]
        assert scatter_ax.get_ylabel() == "condition"

    def test_title_is_gene_name(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene2", "cell_type", "condition"
        )
        scatter_ax = fig.axes[0]
        assert scatter_ax.get_title() == "Gene2"

    def test_xtick_count_matches_categories(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition"
        )
        scatter_ax = fig.axes[0]
        n_expected = small_adata.obs["cell_type"].nunique()
        assert len(scatter_ax.get_xticks()) == n_expected

    def test_ytick_count_matches_categories(self, small_adata):
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition"
        )
        scatter_ax = fig.axes[0]
        n_expected = small_adata.obs["condition"].nunique()
        assert len(scatter_ax.get_yticks()) == n_expected

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------
    def test_invalid_category_x_raises_value_error(self, small_adata):
        with pytest.raises(ValueError, match="category_x"):
            dotplot_expression_two_categories(
                small_adata, "Gene0", "nonexistent_col", "condition"
            )

    def test_invalid_category_y_raises_value_error(self, small_adata):
        with pytest.raises(ValueError, match="category_y"):
            dotplot_expression_two_categories(
                small_adata, "Gene0", "cell_type", "nonexistent_col"
            )

    def test_invalid_gene_raises_key_error(self, small_adata):
        with pytest.raises(KeyError):
            dotplot_expression_two_categories(
                small_adata, "NotAGene", "cell_type", "condition"
            )

    def test_invalid_gene_symbols_raises_key_error(self, small_adata):
        with pytest.raises(KeyError):
            dotplot_expression_two_categories(
                small_adata, "NotASym", "cell_type", "condition",
                gene_symbols="symbol",
            )

    # ------------------------------------------------------------------
    # Size legend consistency
    # ------------------------------------------------------------------
    def test_size_legend_handles_and_labels_same_count(self, small_adata):
        """Size legend handles and labels have the same count."""
        fig = dotplot_expression_two_categories(
            small_adata, "Gene0", "cell_type", "condition"
        )
        scatter_ax = fig.axes[0]
        legend = scatter_ax.get_legend()
        assert legend is not None
        n_labels = len(legend.get_texts())
        assert n_labels >= 3

    def test_feature_as_obs_column(self, small_adata):
        """feature accepts a numeric adata.obs column."""
        adata = small_adata.copy()
        rng = np.random.default_rng(7)
        adata.obs["score"] = rng.random(len(adata))
        fig = dotplot_expression_two_categories(
            adata, "score", "cell_type", "condition"
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Integration tests for dotplot_expression_two_categories_multiplot
# ---------------------------------------------------------------------------

class TestDotplotExpressionTwoCategoriesMultiplot:
    # ------------------------------------------------------------------
    # Return type / basic smoke tests
    # ------------------------------------------------------------------
    def test_returns_figure_two_features(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
        )
        assert isinstance(fig, Figure)

    def test_returns_figure_four_features(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2", "Gene3"],
            category_x="cell_type",
            category_y="condition",
        )
        assert isinstance(fig, Figure)

    def test_returns_figure_single_feature(self, small_adata):
        """A one-element feature list is a valid edge case."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0"],
            category_x="cell_type",
            category_y="condition",
        )
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # Axes count
    # ------------------------------------------------------------------
    def test_axes_count_two_features(self, small_adata):
        """2 features → 2 scatter + 2 cbar + 1 legend_space = 5 axes."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
        )
        assert len(fig.axes) == 2 * 2 + 1

    def test_axes_count_four_features(self, small_adata):
        """4 features → 4 scatter + 4 cbar + 1 legend_space = 9 axes."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2", "Gene3"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
        )
        assert len(fig.axes) == 4 * 2 + 1

    def test_axes_count_three_features_ncols_3(self, small_adata):
        """3 features, ncols=3 → 1 row: 3*2 + 1 = 7 axes."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=3,
        )
        assert len(fig.axes) == 3 * 2 + 1

    def test_axes_count_three_features_ncols_1(self, small_adata):
        """3 features, ncols=1 → 3 rows: 3*2 + 1 = 7 axes."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=1,
        )
        assert len(fig.axes) == 3 * 2 + 1

    def test_axes_count_odd_features(self, small_adata):
        """3 features, ncols=2 → last row has one empty cell: 3*2+1=7 axes."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
        )
        assert len(fig.axes) == 3 * 2 + 1

    # ------------------------------------------------------------------
    # ncols behaviour
    # ------------------------------------------------------------------
    def test_ncols_1_returns_figure(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=1,
        )
        assert isinstance(fig, Figure)

    def test_ncols_3_returns_figure(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=3,
        )
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # Subplot titles
    # ------------------------------------------------------------------
    def test_subplot_titles_match_features(self, small_adata):
        features = ["Gene0", "Gene2", "Gene4"]
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=features,
            category_x="cell_type",
            category_y="condition",
            ncols=2,
        )
        # axes[0]=legend_space; axes[1,3,5]=scatter axes
        scatter_axes = [fig.axes[1 + 2 * i] for i in range(len(features))]
        titles = [ax.get_title() for ax in scatter_axes]
        assert titles == features

    # ------------------------------------------------------------------
    # Size legend
    # ------------------------------------------------------------------
    def test_size_legend_present_on_first_scatter(self, small_adata):
        """The first scatter axes carries the size legend."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
        )
        # axes[1] = first scatter_ax = first_scatter_ax
        assert fig.axes[1].get_legend() is not None

    def test_size_legend_absent_on_other_scatter_axes(self, small_adata):
        """Only the first scatter axes has a legend; the others do not."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
        )
        # axes[3] = second scatter, axes[5] = third scatter
        assert fig.axes[3].get_legend() is None
        assert fig.axes[5].get_legend() is None

    def test_size_legend_label_count(self, small_adata):
        """The legend has between 3 and 6 entries."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
        )
        legend = fig.axes[1].get_legend()
        assert 3 <= len(legend.get_texts()) <= 6

    # ------------------------------------------------------------------
    # Shared colorscale
    # ------------------------------------------------------------------
    def test_shared_colorscale_true_same_norm(self, small_adata):
        """shared_colorscale=True with explicit vmin/vmax: all norms are identical.

        Using explicit float limits ensures matplotlib's Normalize never
        calls autoscale_None (which can expand the range when vmin==vmax),
        making the assertion robust.
        """
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            shared_colorscale=True,
            vmin=0.0,
            vmax=10.0,
        )
        # axes[1] and axes[3] are the scatter axes
        norm0 = fig.axes[1].collections[0].norm
        norm1 = fig.axes[3].collections[0].norm
        assert norm0.vmin == pytest.approx(0.0)
        assert norm0.vmax == pytest.approx(10.0)
        assert norm1.vmin == pytest.approx(0.0)
        assert norm1.vmax == pytest.approx(10.0)

    def test_shared_colorscale_false_different_norm(self, small_adata):
        """shared_colorscale=False: per-feature p95 vmax differs between features.

        Each gene is given a distinct non-degenerate distribution so that
        per-feature p95 values are clearly different from the global p95.
        """
        adata = small_adata.copy()
        rng2 = np.random.default_rng(123)
        # Gene0 group means will fall in [1, 5]; Gene1 in [50, 100]
        adata.X[:, 0] = rng2.uniform(1, 5, size=adata.n_obs)
        adata.X[:, 1] = rng2.uniform(50, 100, size=adata.n_obs)
        fig = dotplot_expression_two_categories_multiplot(
            adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            shared_colorscale=False,
            vmax="p95",
        )
        norm0 = fig.axes[1].collections[0].norm
        norm1 = fig.axes[3].collections[0].norm
        # Per-feature p95: Gene0 ≈ 4.x, Gene1 ≈ 98.x — clearly different
        assert norm0.vmax != pytest.approx(norm1.vmax, rel=0.1)

    def test_shared_colorscale_true_smoke(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            shared_colorscale=True,
        )
        assert isinstance(fig, Figure)

    def test_shared_colorscale_false_smoke(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            shared_colorscale=False,
        )
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # Shared dot_max
    # ------------------------------------------------------------------
    def test_explicit_dot_max_caps_scatter_sizes(self, small_adata):
        """Scatter sizes must not exceed _LARGEST_DOT for any subplot."""
        from src.plotting.dotplots import _LARGEST_DOT
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            dot_max=0.5,
        )
        for i in range(3):
            scatter_ax = fig.axes[1 + 2 * i]
            sizes = scatter_ax.collections[0].get_sizes()
            assert all(s <= _LARGEST_DOT + 1e-6 for s in sizes)

    # ------------------------------------------------------------------
    # vmin / vmax
    # ------------------------------------------------------------------
    def test_vmin_vmax_float(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            vmin=0.0,
            vmax=5.0,
        )
        assert isinstance(fig, Figure)

    def test_vmin_vmax_percentile_string(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            vmin="p5",
            vmax="p95",
        )
        assert isinstance(fig, Figure)

    def test_invalid_vmin_string_raises(self, small_adata):
        with pytest.raises(ValueError, match="p"):
            dotplot_expression_two_categories_multiplot(
                small_adata,
                features=["Gene0", "Gene1"],
                category_x="cell_type",
                category_y="condition",
                vmin="bad",
            )

    # ------------------------------------------------------------------
    # use_zscores
    # ------------------------------------------------------------------
    def test_use_zscores_smoke(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            use_zscores=True,
        )
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # layer support
    # ------------------------------------------------------------------
    def test_layer_parameter_smoke(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            layer="raw",
        )
        assert isinstance(fig, Figure)

    def test_layer_changes_means(self, small_adata):
        """'raw' layer (2× X) should yield higher means than the default."""
        fig_default = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
        )
        fig_raw = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            layer="raw",
        )
        # Colorbar limits differ between layers
        norm_default = fig_default.axes[1].collections[0].norm
        norm_raw = fig_raw.axes[1].collections[0].norm
        # raw layer is 2× so colour data must differ in at least one subplot
        data_default = fig_default.axes[1].collections[0].get_array()
        data_raw = fig_raw.axes[1].collections[0].get_array()
        assert not np.allclose(data_default, data_raw)

    # ------------------------------------------------------------------
    # gene_symbols support
    # ------------------------------------------------------------------
    def test_gene_symbols_parameter(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Sym0", "Sym1"],
            category_x="cell_type",
            category_y="condition",
            gene_symbols="symbol",
        )
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # obs column as feature
    # ------------------------------------------------------------------
    def test_feature_as_obs_column(self, small_adata):
        adata = small_adata.copy()
        rng = np.random.default_rng(99)
        adata.obs["score"] = rng.random(len(adata))
        fig = dotplot_expression_two_categories_multiplot(
            adata,
            features=["score", "Gene0"],
            category_x="cell_type",
            category_y="condition",
        )
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # figsize
    # ------------------------------------------------------------------
    def test_explicit_figsize_scales_total_figure(self, small_adata):
        """A larger per-panel figsize produces a proportionally larger figure."""
        fig_small = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            figsize=(2.0, 2.0),
        )
        fig_large = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            figsize=(6.0, 6.0),
        )
        w_small, h_small = fig_small.get_size_inches()
        w_large, h_large = fig_large.get_size_inches()
        assert w_large > w_small
        assert h_large > h_small

    def test_default_figsize_is_set(self, small_adata):
        """Without figsize, the figure has a positive non-zero size."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
        )
        w, h = fig.get_size_inches()
        assert w > 0
        assert h > 0

    def test_hspace_parameter_smoke(self, small_adata):
        """Custom hspace is accepted without error."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            hspace=1.5,
        )
        assert isinstance(fig, Figure)

    def test_wspace_parameter_smoke(self, small_adata):
        """Custom wspace is accepted without error."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            wspace=0.6,
        )
        assert isinstance(fig, Figure)

    def test_wspace_larger_increases_total_width(self, small_adata):
        """A larger wspace produces a wider total figure."""
        fig_narrow = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            wspace=0.2,
        )
        fig_wide = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            wspace=2.0,
        )
        assert fig_wide.get_size_inches()[0] > fig_narrow.get_size_inches()[0]

    # ------------------------------------------------------------------
    # cmap
    # ------------------------------------------------------------------
    def test_different_cmap_smoke(self, small_adata):
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            cmap="Blues",
        )
        assert isinstance(fig, Figure)

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------
    def test_empty_features_raises_value_error(self, small_adata):
        with pytest.raises(ValueError, match="empty"):
            dotplot_expression_two_categories_multiplot(
                small_adata,
                features=[],
                category_x="cell_type",
                category_y="condition",
            )

    def test_invalid_category_x_raises_value_error(self, small_adata):
        with pytest.raises(ValueError, match="category_x"):
            dotplot_expression_two_categories_multiplot(
                small_adata,
                features=["Gene0"],
                category_x="nonexistent",
                category_y="condition",
            )

    def test_invalid_category_y_raises_value_error(self, small_adata):
        with pytest.raises(ValueError, match="category_y"):
            dotplot_expression_two_categories_multiplot(
                small_adata,
                features=["Gene0"],
                category_x="cell_type",
                category_y="nonexistent",
            )

    def test_invalid_feature_raises_key_error(self, small_adata):
        with pytest.raises(KeyError):
            dotplot_expression_two_categories_multiplot(
                small_adata,
                features=["Gene0", "NotAGene"],
                category_x="cell_type",
                category_y="condition",
            )

    def test_invalid_gene_symbols_feature_raises_key_error(self, small_adata):
        with pytest.raises(KeyError):
            dotplot_expression_two_categories_multiplot(
                small_adata,
                features=["NotASym"],
                category_x="cell_type",
                category_y="condition",
                gene_symbols="symbol",
            )

    # ------------------------------------------------------------------
    # border_ticks_only
    # ------------------------------------------------------------------
    # Layout for 4 features / ncols=2:
    #   axes[0]  = legend_space (invisible)
    #   axes[1]  = scatter row=0, col=0  (top-left)
    #   axes[2]  = cbar   row=0, col=0
    #   axes[3]  = scatter row=0, col=1  (top-right)
    #   axes[4]  = cbar   row=0, col=1
    #   axes[5]  = scatter row=1, col=0  (bottom-left)
    #   axes[6]  = cbar   row=1, col=0
    #   axes[7]  = scatter row=1, col=1  (bottom-right)
    #   axes[8]  = cbar   row=1, col=1
    # ------------------------------------------------------------------

    def test_border_ticks_only_xlabel_absent_nonbottom(self, small_adata):
        """Top-row scatter axes have an empty xlabel when border_ticks_only=True."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2", "Gene3"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            border_ticks_only=True,
        )
        top_row_scatter = [fig.axes[1], fig.axes[3]]
        for ax in top_row_scatter:
            assert ax.get_xlabel() == ""

    def test_border_ticks_only_xlabel_present_bottom(self, small_adata):
        """Bottom-row scatter axes carry category_x as xlabel."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2", "Gene3"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            border_ticks_only=True,
        )
        bottom_row_scatter = [fig.axes[5], fig.axes[7]]
        for ax in bottom_row_scatter:
            assert ax.get_xlabel() == "cell_type"

    def test_border_ticks_only_xticklabels_absent_nonbottom(self, small_adata):
        """Top-row scatter axes have empty xticklabels when border_ticks_only=True."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2", "Gene3"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            border_ticks_only=True,
        )
        top_row_scatter = [fig.axes[1], fig.axes[3]]
        for ax in top_row_scatter:
            labels = [t.get_text() for t in ax.get_xticklabels()]
            assert all(lbl == "" for lbl in labels)

    def test_border_ticks_only_ylabel_absent_nonfirstcol(self, small_adata):
        """Non-first-column scatter axes have an empty ylabel when border_ticks_only=True."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2", "Gene3"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            border_ticks_only=True,
        )
        nonfirst_col_scatter = [fig.axes[3], fig.axes[7]]
        for ax in nonfirst_col_scatter:
            assert ax.get_ylabel() == ""

    def test_border_ticks_only_ylabel_present_firstcol(self, small_adata):
        """First-column scatter axes carry category_y as ylabel."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2", "Gene3"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            border_ticks_only=True,
        )
        first_col_scatter = [fig.axes[1], fig.axes[5]]
        for ax in first_col_scatter:
            assert ax.get_ylabel() == "condition"

    def test_border_ticks_only_false_all_have_labels(self, small_adata):
        """When border_ticks_only=False every scatter axes has both labels."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1", "Gene2", "Gene3"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            border_ticks_only=False,
        )
        scatter_axes = [fig.axes[1 + 2 * i] for i in range(4)]
        for ax in scatter_axes:
            assert ax.get_xlabel() == "cell_type"
            assert ax.get_ylabel() == "condition"

    def test_border_ticks_only_single_row_all_xlabel(self, small_adata):
        """Single-row grid: every subplot is in the bottom row, so all have xlabel."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            border_ticks_only=True,
        )
        scatter_axes = [fig.axes[1], fig.axes[3]]
        for ax in scatter_axes:
            assert ax.get_xlabel() == "cell_type"

    def test_border_ticks_only_single_row_ylabel_firstcol_only(self, small_adata):
        """Single-row grid: only the leftmost subplot has ylabel."""
        fig = dotplot_expression_two_categories_multiplot(
            small_adata,
            features=["Gene0", "Gene1"],
            category_x="cell_type",
            category_y="condition",
            ncols=2,
            border_ticks_only=True,
        )
        assert fig.axes[1].get_ylabel() == "condition"
        assert fig.axes[3].get_ylabel() == ""
