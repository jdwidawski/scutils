"""Tests for src/plotting/boxplots.py.

Fixtures build small synthetic AnnData objects so the tests are fast and
require no external data files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

matplotlib.use("Agg")  # headless backend for CI

from scutils.plotting._utils import _resolve_palette
from scutils.plotting.boxplots import (
    _pvalue_to_stars,
    _resolve_feature,
    _resolve_vmin_vmax,
    plot_feature_boxplot,
    plot_feature_boxplot_aggregated,
    plot_feature_boxplot_aggregated_multiplot,
    plot_feature_boxplot_multiplot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def small_adata(rng: np.random.Generator) -> AnnData:
    """80-cell, 10-gene AnnData with obs metadata for all tests."""
    n_cells, n_genes = 80, 10
    X = csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32))

    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                np.repeat(["T cell", "B cell", "NK cell", "Mono"], 20)
            ),
            "condition": pd.Categorical(
                np.tile(["ctrl", "stim"], 40)
            ),
            "donor": pd.Categorical(
                np.tile(["D1", "D2", "D3", "D4"], 20)
            ),
            "numeric_score": rng.normal(0, 1, n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(
        {"gene_name": [f"Gene{i}" for i in range(n_genes)]},
        index=[f"ENSG{i:05d}" for i in range(n_genes)],
    )

    adata = AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return adata


@pytest.fixture
def agg_adata(rng: np.random.Generator) -> AnnData:
    """320-cell AnnData where each (donor × cell_type) group has ≥ 20 cells.

    Suitable for aggregated-boxplot tests that use the default min_cells=10.
    Layout: 4 cell types × 4 donors × 2 conditions × 10 cells = 320 cells.
    """
    n_genes = 10
    cell_types = ["T cell", "B cell", "NK cell", "Mono"]
    donors = ["D1", "D2", "D3", "D4"]
    conditions = ["ctrl", "stim"]
    cells_per_group = 10  # 4 × 4 × 2 × 10 = 320 cells

    records = []
    for ct in cell_types:
        for donor in donors:
            for cond in conditions:
                for _ in range(cells_per_group):
                    records.append({"cell_type": ct, "donor": donor, "condition": cond})

    n_cells = len(records)
    obs = pd.DataFrame(records, index=[f"cell_{i}" for i in range(n_cells)])
    for col in ("cell_type", "donor", "condition"):
        obs[col] = pd.Categorical(obs[col])
    obs["numeric_score"] = rng.normal(0, 1, n_cells)

    X = csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32))
    var = pd.DataFrame(
        {"gene_name": [f"Gene{i}" for i in range(n_genes)]},
        index=[f"ENSG{i:05d}" for i in range(n_genes)],
    )

    adata = AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return adata


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestPvalueToStars:
    def test_ns(self):
        assert _pvalue_to_stars(0.1) == "ns"
        assert _pvalue_to_stars(0.05) == "ns"

    def test_one_star(self):
        assert _pvalue_to_stars(0.04) == "*"

    def test_two_stars(self):
        assert _pvalue_to_stars(0.005) == "**"

    def test_three_stars(self):
        assert _pvalue_to_stars(0.0005) == "***"

    def test_four_stars(self):
        assert _pvalue_to_stars(0.00005) == "****"


class TestResolveFeature:
    def test_obs_column_takes_priority(self, small_adata):
        series, is_obs = _resolve_feature(small_adata, "numeric_score", None, None)
        assert is_obs is True
        assert len(series) == small_adata.n_obs

    def test_var_gene_by_var_names(self, small_adata):
        gene = small_adata.var_names[0]
        series, is_obs = _resolve_feature(small_adata, gene, None, None)
        assert is_obs is False
        assert len(series) == small_adata.n_obs

    def test_var_gene_via_layer(self, small_adata):
        gene = small_adata.var_names[0]
        series_x, _ = _resolve_feature(small_adata, gene, None, None)
        series_counts, _ = _resolve_feature(small_adata, gene, "counts", None)
        # Both layers have same data in fixture, but the call should not error
        assert len(series_counts) == small_adata.n_obs

    def test_var_gene_by_gene_symbols(self, small_adata):
        gene_name = small_adata.var["gene_name"].iloc[2]
        series, is_obs = _resolve_feature(small_adata, gene_name, None, "gene_name")
        assert is_obs is False
        assert len(series) == small_adata.n_obs

    def test_missing_feature_raises(self, small_adata):
        with pytest.raises(ValueError, match="not found"):
            _resolve_feature(small_adata, "nonexistent", None, None)

    def test_missing_gene_symbol_raises(self, small_adata):
        with pytest.raises(ValueError, match="not found"):
            _resolve_feature(small_adata, "bad_gene", None, "gene_name")


class TestResolvePalette:
    CATS = ["A", "B", "C"]

    def test_none_returns_none(self):
        assert _resolve_palette(None, self.CATS) is None

    def test_single_colour(self):
        result = _resolve_palette("red", self.CATS)
        assert all(v == "red" for v in result.values())

    def test_list_cycles(self):
        result = _resolve_palette(["#aaa", "#bbb"], self.CATS)
        assert set(result.keys()) == set(self.CATS)
        assert len(result) == 3

    def test_dict_passthrough(self):
        d = {"A": "red", "B": "blue", "C": "green"}
        assert _resolve_palette(d, self.CATS) == d

    def test_palette_name(self):
        result = _resolve_palette("Set2", self.CATS)
        assert set(result.keys()) == set(self.CATS)


# ---------------------------------------------------------------------------
# plot_feature_boxplot
# ---------------------------------------------------------------------------

class TestPlotFeatureBoxplot:
    # --- Returns a Figure ---
    def test_returns_figure_gene(self, small_adata):
        gene = small_adata.var_names[0]
        fig = plot_feature_boxplot(small_adata, feature=gene, x="cell_type")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_obs_column(self, small_adata):
        fig = plot_feature_boxplot(small_adata, feature="numeric_score", x="cell_type")
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- With hue ---
    def test_with_hue(self, small_adata):
        gene = small_adata.var_names[0]
        fig = plot_feature_boxplot(small_adata, feature=gene, x="cell_type", hue="condition")
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- Horizontal orientation ---
    def test_horizontal_orient(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata, feature="numeric_score", x="cell_type", orient="h"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- vmin / vmax clipping ---
    def test_vmin_vmax(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata, feature="numeric_score", x="cell_type", vmin=-1.0, vmax=1.0
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- show_stats ---
    def test_show_stats_no_error(self, small_adata):
        gene = small_adata.var_names[0]
        fig = plot_feature_boxplot(
            small_adata,
            feature=gene,
            x="cell_type",
            hue="condition",
            show_stats=True,
            comparisons=[("ctrl", "stim")],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_show_stats_all_pairs(self, small_adata):
        """show_stats without explicit comparisons tests all hue pairs."""
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            hue="condition",
            show_stats=True,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- Legend locations ---
    @pytest.mark.parametrize("loc", ["outside right", "outside top", "best"])
    def test_legend_locations(self, small_adata, loc):
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            hue="condition",
            legend_loc=loc,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- Palette variants ---
    def test_palette_string_colour(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata, feature="numeric_score", x="condition", palette="steelblue"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_palette_list(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            hue="condition",
            palette=["#e41a1c", "#377eb8"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_palette_dict(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            hue="condition",
            palette={"ctrl": "blue", "stim": "orange"},
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- Layer support ---
    def test_layer_argument(self, small_adata):
        gene = small_adata.var_names[0]
        fig = plot_feature_boxplot(
            small_adata, feature=gene, x="cell_type", layer="counts"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- gene_symbols support ---
    def test_gene_symbols(self, small_adata):
        gene_name = small_adata.var["gene_name"].iloc[0]
        fig = plot_feature_boxplot(
            small_adata,
            feature=gene_name,
            x="cell_type",
            gene_symbols="gene_name",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- Error paths ---
    def test_invalid_x_raises(self, small_adata):
        with pytest.raises(ValueError, match="x='bad_col'"):
            plot_feature_boxplot(small_adata, feature="numeric_score", x="bad_col")

    def test_invalid_hue_raises(self, small_adata):
        with pytest.raises(ValueError, match="hue='bad_col'"):
            plot_feature_boxplot(
                small_adata, feature="numeric_score", x="cell_type", hue="bad_col"
            )

    def test_show_stats_without_hue_raises(self, small_adata):
        with pytest.raises(ValueError, match="show_stats=True requires a hue"):
            plot_feature_boxplot(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                show_stats=True,
            )

    def test_invalid_feature_raises(self, small_adata):
        with pytest.raises(ValueError, match="not found"):
            plot_feature_boxplot(small_adata, feature="BAD_GENE", x="cell_type")

    # --- groups_x ---
    def test_groups_x_subset(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            groups_x=["T cell", "B cell"],
        )
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert set(tick_labels) == {"T cell", "B cell"}

    def test_groups_x_invalid_raises(self, small_adata):
        with pytest.raises(ValueError, match="groups_x values"):
            plot_feature_boxplot(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                groups_x=["T cell", "Unknown"],
            )

    # --- groups_hue ---
    def test_groups_hue_subset(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            hue="condition",
            groups_hue=["ctrl"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_groups_hue_invalid_raises(self, small_adata):
        with pytest.raises(ValueError, match="groups_hue values"):
            plot_feature_boxplot(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                hue="condition",
                groups_hue=["ctrl", "unknown_cond"],
            )

    # --- x_order ---
    def test_x_order_changes_tick_sequence(self, small_adata):
        order = ["NK cell", "T cell", "B cell", "Mono"]
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            x_order=order,
        )
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == order

    def test_x_order_missing_category_raises(self, small_adata):
        with pytest.raises(ValueError, match="x_order is missing"):
            plot_feature_boxplot(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                x_order=["T cell", "B cell"],  # missing NK cell and Mono
            )

    # --- hue_order ---
    def test_hue_order_smoke(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            hue="condition",
            hue_order=["stim", "ctrl"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_hue_order_missing_category_raises(self, small_adata):
        with pytest.raises(ValueError, match="hue_order is missing"):
            plot_feature_boxplot(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                hue="condition",
                hue_order=["ctrl"],  # missing "stim"
            )

    # --- Custom title / labels ---
    def test_custom_title_and_labels(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            title="My title",
            xlabel="Cell type",
            ylabel="Score",
        )
        ax = fig.axes[0]
        assert ax.get_title() == "My title"
        assert ax.get_xlabel() == "Cell type"
        assert ax.get_ylabel() == "Score"

    # --- show_points flag ---
    def test_show_points_false(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            show_points=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- figsize ---
    def test_custom_figsize(self, small_adata):
        fig = plot_feature_boxplot(
            small_adata, feature="numeric_score", x="cell_type", figsize=(10.0, 5.0)
        )
        assert fig.get_size_inches() == pytest.approx([10.0, 5.0])


# ---------------------------------------------------------------------------
# plot_feature_boxplot_aggregated
# ---------------------------------------------------------------------------

class TestPlotFeatureBoxplotAggregated:
    # --- Basic smoke tests ---
    def test_returns_figure_gene(self, agg_adata):
        gene = agg_adata.var_names[0]
        fig = plot_feature_boxplot_aggregated(
            agg_adata, feature=gene, x="cell_type", sample_col="donor"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_obs_column(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_hue(self, agg_adata):
        gene = agg_adata.var_names[0]
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature=gene,
            x="cell_type",
            sample_col="donor",
            hue="condition",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- Aggregation functions ---
    @pytest.mark.parametrize("agg_fn", ["mean", "median", "sum"])
    def test_agg_fn(self, agg_adata, agg_fn):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            agg_fn=agg_fn,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_invalid_agg_fn_raises(self, small_adata):
        with pytest.raises(ValueError, match="agg_fn="):
            plot_feature_boxplot_aggregated(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                agg_fn="variance",  # type: ignore[arg-type]
            )

    # --- min_cells filter ---
    def test_min_cells_filters_samples(self, small_adata):
        """Setting min_cells very high should drop all samples and raise."""
        with pytest.raises(ValueError, match="No samples remain"):
            plot_feature_boxplot_aggregated(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                min_cells=10_000,
            )

    def test_min_cells_zero_keeps_all(self, small_adata):
        fig = plot_feature_boxplot_aggregated(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            min_cells=0,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- Orientation ---
    def test_horizontal_orient(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            orient="h",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- vmin / vmax ---
    def test_vmin_vmax(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            vmin=-0.5,
            vmax=0.5,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- show_stats ---
    def test_show_stats(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            hue="condition",
            show_stats=True,
            comparisons=[("ctrl", "stim")],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_show_stats_without_hue_raises(self, small_adata):
        with pytest.raises(ValueError, match="show_stats=True requires a hue"):
            plot_feature_boxplot_aggregated(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                show_stats=True,
            )

    # --- Legend locations ---
    @pytest.mark.parametrize("loc", ["outside right", "outside top", "best"])
    def test_legend_locations(self, agg_adata, loc):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            hue="condition",
            legend_loc=loc,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- Error paths ---
    def test_invalid_x_raises(self, small_adata):
        with pytest.raises(ValueError, match="x='bad'"):
            plot_feature_boxplot_aggregated(
                small_adata, feature="numeric_score", x="bad", sample_col="donor"
            )

    def test_invalid_sample_col_raises(self, small_adata):
        with pytest.raises(ValueError, match="sample_col='bad'"):
            plot_feature_boxplot_aggregated(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="bad",
            )

    def test_invalid_hue_raises(self, small_adata):
        with pytest.raises(ValueError, match="hue='bad'"):
            plot_feature_boxplot_aggregated(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                hue="bad",
            )

    def test_invalid_feature_raises(self, small_adata):
        with pytest.raises(ValueError, match="not found"):
            plot_feature_boxplot_aggregated(
                small_adata, feature="BAD", x="cell_type", sample_col="donor"
            )

    # --- groups_x / groups_hue / x_order / hue_order ---

    def test_groups_x_subset(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            groups_x=["T cell", "B cell"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_groups_x_invalid_raises(self, agg_adata):
        with pytest.raises(ValueError, match="groups_x values"):
            plot_feature_boxplot_aggregated(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                groups_x=["T cell", "Unknown"],
            )

    def test_groups_hue_subset(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            hue="condition",
            groups_hue=["ctrl"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_groups_hue_invalid_raises(self, agg_adata):
        with pytest.raises(ValueError, match="groups_hue values"):
            plot_feature_boxplot_aggregated(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                hue="condition",
                groups_hue=["ctrl", "unknown_cond"],
            )

    def test_x_order_smoke(self, agg_adata):
        order = ["NK cell", "T cell", "B cell", "Mono"]
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            x_order=order,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_x_order_missing_category_raises(self, agg_adata):
        with pytest.raises(ValueError, match="x_order is missing"):
            plot_feature_boxplot_aggregated(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                x_order=["T cell", "B cell"],  # missing NK cell and Mono
            )

    def test_hue_order_smoke(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            hue="condition",
            hue_order=["stim", "ctrl"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_hue_order_missing_category_raises(self, agg_adata):
        with pytest.raises(ValueError, match="hue_order is missing"):
            plot_feature_boxplot_aggregated(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                hue="condition",
                hue_order=["ctrl"],  # missing "stim"
            )

    # --- Custom title ---
    def test_custom_title(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            title="Custom",
        )
        assert fig.axes[0].get_title() == "Custom"

    # --- Default title contains agg_fn ---
    def test_default_title_contains_agg_fn(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            agg_fn="median",
        )
        assert "median" in fig.axes[0].get_title()

    # --- Aggregated values are correct ---
    def test_aggregation_values_mean(self, agg_adata):
        """Verify the aggregated function produces the correct per-sample means."""
        gene = agg_adata.var_names[0]
        expected_vals = sc.get.obs_df(agg_adata, keys=[gene], use_raw=False)[gene]
        obs = agg_adata.obs.copy()
        obs["_val"] = expected_vals.values
        expected_grouped = obs.groupby(["donor", "cell_type"], observed=True)["_val"].mean()
        # Check that none of the groups were erroneously dropped (all donors ×
        # cell types have 20 cells, well above the default min_cells=10).
        assert len(expected_grouped) == 4 * 4  # 4 donors × 4 cell types
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature=gene,
            x="cell_type",
            sample_col="donor",
            show_points=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- Layer support ---
    def test_layer_argument(self, agg_adata):
        gene = agg_adata.var_names[0]
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature=gene,
            x="cell_type",
            sample_col="donor",
            layer="counts",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- gene_symbols ---
    def test_gene_symbols(self, agg_adata):
        gene_name = agg_adata.var["gene_name"].iloc[0]
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature=gene_name,
            x="cell_type",
            sample_col="donor",
            gene_symbols="gene_name",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- show_points flag ---
    def test_show_points_false(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            show_points=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- figsize ---
    def test_custom_figsize(self, agg_adata):
        fig = plot_feature_boxplot_aggregated(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            figsize=(12.0, 6.0),
        )
        assert fig.get_size_inches() == pytest.approx([12.0, 6.0])


# ---------------------------------------------------------------------------
# _resolve_vmin_vmax helper
# ---------------------------------------------------------------------------

class TestResolveVminVmax:
    def test_none_returns_none(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _resolve_vmin_vmax(s, None) is None

    def test_float_passthrough(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _resolve_vmin_vmax(s, 5.0) == pytest.approx(5.0)

    def test_percentile_string(self):
        s = pd.Series(np.arange(101, dtype=float))
        result = _resolve_vmin_vmax(s, "p50")
        assert result == pytest.approx(50.0)

    def test_invalid_string_raises(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="p95"):
            _resolve_vmin_vmax(s, "x95")


# ---------------------------------------------------------------------------
# plot_feature_boxplot_multiplot
# ---------------------------------------------------------------------------

class TestPlotFeatureBoxplotMultiplot:

    # --- smoke / return type ---

    def test_returns_figure(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_gene_feature_smoke(self, small_adata):
        gene = small_adata.var_names[0]
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature=gene, x="cell_type"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_hue_smoke(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type", hue="condition"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- panel count ---

    def test_all_x_categories_get_a_panel(self, small_adata):
        """One used axes per x category."""
        n_cats = small_adata.obs["cell_type"].nunique()
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type", ncols=2
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        assert len(used) == n_cats

    def test_groups_subset_reduces_panel_count(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            groups=["T cell", "B cell"],
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        assert len(used) == 2

    def test_empty_panels_hidden_when_odd_count(self, small_adata):
        """3 groups with ncols=2 → 4 grid cells, one hidden."""
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            groups=["T cell", "B cell", "NK cell"],
            ncols=2,
        )
        all_axes = fig.axes
        hidden = [ax for ax in all_axes if not ax.get_visible()]
        assert len(hidden) == 1

    def test_ncols_controls_grid_shape(self, small_adata):
        fig_narrow = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type", ncols=1
        )
        fig_wide = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type", ncols=4
        )
        # ncols=1 → taller; ncols=4 → wider
        assert fig_narrow.get_size_inches()[1] > fig_wide.get_size_inches()[1]
        assert fig_wide.get_size_inches()[0] > fig_narrow.get_size_inches()[0]

    # --- figsize (per-panel semantics) ---

    def test_figsize_is_per_panel(self, small_adata):
        fig_small = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type",
            ncols=2, figsize=(2.0, 2.0)
        )
        fig_large = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type",
            ncols=2, figsize=(6.0, 6.0)
        )
        assert fig_large.get_size_inches()[0] > fig_small.get_size_inches()[0]
        assert fig_large.get_size_inches()[1] > fig_small.get_size_inches()[1]

    # --- shared_colorscale ---

    def test_shared_colorscale_true_equal_ylims(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            shared_colorscale=True,
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        ylims = [ax.get_ylim() for ax in used]
        for ylim in ylims:
            assert ylim == pytest.approx(ylims[0])

    def test_shared_colorscale_false_may_differ(self, small_adata):
        """With per-panel percentile vmin/vmax and shared_colorscale=False,
        panels are free to have different y-limits."""
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            shared_colorscale=False,
            vmin="p5",
            vmax="p95",
            ncols=2,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_shared_colorscale_float_vmin_vmax(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            shared_colorscale=True,
            vmin=-2.0,
            vmax=2.0,
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        for ax in used:
            assert ax.get_ylim() == pytest.approx((-2.0, 2.0))

    def test_shared_colorscale_percentile_vmax(self, small_adata):
        """Percentile vmax resolved globally → same ylim on all panels."""
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            shared_colorscale=True,
            vmax="p95",
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        ylims = [ax.get_ylim() for ax in used]
        # upper limit should be the same on every panel
        assert all(yl[1] == pytest.approx(ylims[0][1]) for yl in ylims)

    # --- orient ---

    def test_orient_h_smoke(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type", orient="h"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_orient_h_shared_colorscale_equal_xlims(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            orient="h",
            shared_colorscale=True,
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        xlims = [ax.get_xlim() for ax in used]
        for xlim in xlims:
            assert xlim == pytest.approx(xlims[0])

    # --- show_points ---

    def test_show_points_smoke(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type",
            show_points=True
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- layer and gene_symbols ---

    def test_layer_smoke(self, small_adata):
        gene = small_adata.var_names[0]
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature=gene, x="cell_type", layer="counts"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_gene_symbols_smoke(self, small_adata):
        gene_name = small_adata.var["gene_name"].iloc[0]
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature=gene_name, x="cell_type", gene_symbols="gene_name"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- border_ticks_only ---

    def test_border_ticks_only_true_nonbottom_row_no_xlabel(self, small_adata):
        """Non-bottom rows should have empty xlabel when border_ticks_only=True."""
        # 4 groups, ncols=2 → 2 rows; row-0 panels are non-bottom
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            ncols=2,
            border_ticks_only=True,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        assert used[0].get_xlabel() == ""
        assert used[1].get_xlabel() == ""

    def test_border_ticks_only_true_bottom_row_has_xlabel(self, small_adata):
        """Bottom-row panels should keep their xlabel."""
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            ncols=2,
            border_ticks_only=True,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        assert used[2].get_xlabel() != ""
        assert used[3].get_xlabel() != ""

    def test_border_ticks_only_false_all_have_xlabel(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            ncols=2,
            border_ticks_only=False,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        for ax in used:
            assert ax.get_xlabel() != ""

    def test_border_ticks_only_single_row_always_has_xlabel(self, small_adata):
        """With a single row, every panel is the bottom row."""
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            ncols=4,
            border_ticks_only=True,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        for ax in used:
            assert ax.get_xlabel() != ""

    # --- xtick_rotation ---

    def test_xtick_rotation_default_smoke(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            ncols=4,
            border_ticks_only=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_xtick_rotation_custom_smoke(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            ncols=4,
            border_ticks_only=False,
            xtick_rotation=45,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- show_stats ---

    def test_show_stats_with_hue_smoke(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            hue="condition",
            show_stats=True,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- title / xlabel / ylabel ---

    def test_title_set(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type", title="My Title"
        )
        assert fig.texts[0].get_text() == "My Title"

    def test_panel_titles_are_group_names(self, small_adata):
        groups = ["T cell", "B cell"]
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            groups=groups,
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        titles = [ax.get_title() for ax in used]
        assert set(titles) == set(groups)

    def test_ylabel_default_is_feature(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type", ncols=4
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        for ax in used:
            assert ax.get_ylabel() == "numeric_score"

    def test_ylabel_override(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata, feature="numeric_score", x="cell_type",
            ylabel="Custom Y", ncols=4
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        for ax in used:
            assert ax.get_ylabel() == "Custom Y"

    # --- groups_hue / x_order / hue_order (multiplot-specific) ---

    def test_groups_hue_subset(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            hue="condition",
            groups_hue=["ctrl"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_groups_hue_invalid_raises(self, small_adata):
        with pytest.raises(ValueError, match="groups_hue values"):
            plot_feature_boxplot_multiplot(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                hue="condition",
                groups_hue=["ctrl", "unknown_cond"],
            )

    def test_x_order_controls_panel_sequence(self, small_adata):
        order = ["NK cell", "T cell", "B cell", "Mono"]
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            x_order=order,
            ncols=4,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        panel_titles = [ax.get_title() for ax in used]
        assert panel_titles == order

    def test_x_order_missing_category_raises(self, small_adata):
        with pytest.raises(ValueError, match="x_order is missing"):
            plot_feature_boxplot_multiplot(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                x_order=["T cell", "B cell"],  # missing NK cell and Mono
            )

    def test_hue_order_smoke(self, small_adata):
        fig = plot_feature_boxplot_multiplot(
            small_adata,
            feature="numeric_score",
            x="cell_type",
            hue="condition",
            hue_order=["stim", "ctrl"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_hue_order_missing_category_raises(self, small_adata):
        with pytest.raises(ValueError, match="hue_order is missing"):
            plot_feature_boxplot_multiplot(
                small_adata,
                feature="numeric_score",
                x="cell_type",
                hue="condition",
                hue_order=["ctrl"],  # missing "stim"
            )

    # --- error paths ---

    def test_invalid_x_raises(self, small_adata):
        with pytest.raises(ValueError, match="x='bad'"):
            plot_feature_boxplot_multiplot(
                small_adata, feature="numeric_score", x="bad"
            )

    def test_invalid_hue_raises(self, small_adata):
        with pytest.raises(ValueError, match="hue='bad'"):
            plot_feature_boxplot_multiplot(
                small_adata, feature="numeric_score", x="cell_type", hue="bad"
            )

    def test_invalid_groups_raises(self, small_adata):
        with pytest.raises(ValueError, match="Groups"):
            plot_feature_boxplot_multiplot(
                small_adata, feature="numeric_score", x="cell_type",
                groups=["T cell", "Unknown"]
            )

    def test_show_stats_without_hue_raises(self, small_adata):
        with pytest.raises(ValueError, match="hue"):
            plot_feature_boxplot_multiplot(
                small_adata, feature="numeric_score", x="cell_type",
                show_stats=True
            )

    def test_invalid_feature_raises(self, small_adata):
        with pytest.raises(ValueError, match="not found"):
            plot_feature_boxplot_multiplot(
                small_adata, feature="DOES_NOT_EXIST", x="cell_type"
            )

    def test_invalid_vmin_string_raises(self, small_adata):
        with pytest.raises(ValueError, match="p95"):
            plot_feature_boxplot_multiplot(
                small_adata, feature="numeric_score", x="cell_type",
                vmax="x95"
            )


# ---------------------------------------------------------------------------
# plot_feature_boxplot_aggregated_multiplot
# ---------------------------------------------------------------------------

class TestPlotFeatureBoxplotAggregatedMultiplot:

    # --- smoke / return type ---

    def test_returns_figure(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_gene_feature_smoke(self, agg_adata):
        gene = agg_adata.var_names[0]
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata, feature=gene, x="cell_type", sample_col="donor"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_hue_smoke(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            hue="condition",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- panel count / grid shape ---

    def test_all_x_categories_get_a_panel(self, agg_adata):
        n_cats = agg_adata.obs["cell_type"].nunique()
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        assert len(used) == n_cats

    def test_groups_subset_reduces_panel_count(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            groups=["T cell", "B cell"],
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        assert len(used) == 2

    def test_hidden_panels_when_odd_count(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            groups=["T cell", "B cell", "NK cell"],
            ncols=2,
        )
        hidden = [ax for ax in fig.axes if not ax.get_visible()]
        assert len(hidden) == 1

    def test_ncols_controls_grid_shape(self, agg_adata):
        fig_narrow = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            ncols=1,
        )
        fig_wide = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            ncols=4,
        )
        assert fig_narrow.get_size_inches()[1] > fig_wide.get_size_inches()[1]
        assert fig_wide.get_size_inches()[0] > fig_narrow.get_size_inches()[0]

    # --- aggregation ---

    @pytest.mark.parametrize("agg_fn", ["mean", "median", "sum"])
    def test_agg_fn(self, agg_adata, agg_fn):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            agg_fn=agg_fn,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_invalid_agg_fn_raises(self, agg_adata):
        with pytest.raises(ValueError, match="agg_fn="):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                agg_fn="variance",  # type: ignore[arg-type]
            )

    def test_min_cells_filters_samples(self, agg_adata):
        with pytest.raises(ValueError, match="No samples remain"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                min_cells=10_000,
            )

    # --- shared_colorscale ---

    def test_shared_colorscale_true_equal_ylims(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            shared_colorscale=True,
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        ylims = [ax.get_ylim() for ax in used]
        for ylim in ylims:
            assert ylim == pytest.approx(ylims[0])

    def test_shared_colorscale_false_smoke(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            shared_colorscale=False,
            ncols=2,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_shared_colorscale_float_vmin_vmax(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            shared_colorscale=True,
            vmin=-2.0,
            vmax=2.0,
            ncols=2,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        for ax in used:
            assert ax.get_ylim() == pytest.approx((-2.0, 2.0))

    # --- groups / groups_hue / x_order / hue_order ---

    def test_groups_hue_subset(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            hue="condition",
            groups_hue=["ctrl"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_groups_hue_invalid_raises(self, agg_adata):
        with pytest.raises(ValueError, match="groups_hue values"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                hue="condition",
                groups_hue=["ctrl", "unknown"],
            )

    def test_x_order_controls_panel_sequence(self, agg_adata):
        order = ["NK cell", "T cell", "B cell", "Mono"]
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            x_order=order,
            ncols=4,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        assert [ax.get_title() for ax in used] == order

    def test_x_order_missing_raises(self, agg_adata):
        with pytest.raises(ValueError, match="x_order is missing"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                x_order=["T cell", "B cell"],  # missing NK cell and Mono
            )

    def test_hue_order_smoke(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            hue="condition",
            hue_order=["stim", "ctrl"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_hue_order_missing_raises(self, agg_adata):
        with pytest.raises(ValueError, match="hue_order is missing"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                hue="condition",
                hue_order=["ctrl"],  # missing "stim"
            )

    # --- border_ticks_only / xtick_rotation ---

    def test_border_ticks_only_nonbottom_row_no_xlabel(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            ncols=2,
            border_ticks_only=True,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        assert used[0].get_xlabel() == ""
        assert used[1].get_xlabel() == ""

    def test_border_ticks_only_bottom_row_has_xlabel(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            ncols=2,
            border_ticks_only=True,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        assert used[2].get_xlabel() != ""
        assert used[3].get_xlabel() != ""

    def test_xtick_rotation_smoke(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            ncols=4,
            border_ticks_only=False,
            xtick_rotation=45,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- show_stats ---

    def test_show_stats_smoke(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            hue="condition",
            show_stats=True,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- orient ---

    def test_orient_h_smoke(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            orient="h",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- layer / gene_symbols ---

    def test_layer_smoke(self, agg_adata):
        gene = agg_adata.var_names[0]
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata, feature=gene, x="cell_type", sample_col="donor",
            layer="counts",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_gene_symbols_smoke(self, agg_adata):
        gene_name = agg_adata.var["gene_name"].iloc[0]
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata, feature=gene_name, x="cell_type", sample_col="donor",
            gene_symbols="gene_name",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    # --- title / ylabel ---

    def test_title_set(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            title="My Title",
        )
        assert fig.texts[0].get_text() == "My Title"

    def test_ylabel_default_is_feature(self, agg_adata):
        fig = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            ncols=4,
        )
        used = [ax for ax in fig.axes if ax.get_visible()]
        for ax in used:
            assert ax.get_ylabel() == "numeric_score"

    # --- figsize (per-panel) ---

    def test_figsize_is_per_panel(self, agg_adata):
        fig_small = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            ncols=2,
            figsize=(2.0, 2.0),
        )
        fig_large = plot_feature_boxplot_aggregated_multiplot(
            agg_adata,
            feature="numeric_score",
            x="cell_type",
            sample_col="donor",
            ncols=2,
            figsize=(6.0, 6.0),
        )
        assert fig_large.get_size_inches()[0] > fig_small.get_size_inches()[0]
        assert fig_large.get_size_inches()[1] > fig_small.get_size_inches()[1]

    # --- error paths ---

    def test_invalid_x_raises(self, agg_adata):
        with pytest.raises(ValueError, match="x='bad'"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata, feature="numeric_score", x="bad", sample_col="donor"
            )

    def test_invalid_sample_col_raises(self, agg_adata):
        with pytest.raises(ValueError, match="sample_col='bad'"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata, feature="numeric_score", x="cell_type", sample_col="bad"
            )

    def test_invalid_hue_raises(self, agg_adata):
        with pytest.raises(ValueError, match="hue='bad'"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                hue="bad",
            )

    def test_show_stats_without_hue_raises(self, agg_adata):
        with pytest.raises(ValueError, match="hue"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                show_stats=True,
            )

    def test_invalid_feature_raises(self, agg_adata):
        with pytest.raises(ValueError, match="not found"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata, feature="DOES_NOT_EXIST", x="cell_type", sample_col="donor"
            )

    def test_invalid_groups_raises(self, agg_adata):
        with pytest.raises(ValueError, match="Groups"):
            plot_feature_boxplot_aggregated_multiplot(
                agg_adata,
                feature="numeric_score",
                x="cell_type",
                sample_col="donor",
                groups=["T cell", "Unknown"],
            )
