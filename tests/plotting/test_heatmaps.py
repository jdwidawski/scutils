"""Tests for scutils.plotting.heatmaps."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.figure import Figure

matplotlib.use("Agg")

from scutils.plotting.heatmaps import (
    heatmap_expression_two_categories,
    heatmap_expression_two_categories_multiplot,
    heatmap_feature_aggregated_three_categories,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adata_heatmap() -> AnnData:
    """80 cells × 5 genes; obs: cell_type (A/B), condition (ctrl/stim)."""
    rng = np.random.default_rng(0)
    n_obs, n_vars = 80, 5
    X = rng.poisson(3, size=(n_obs, n_vars)).astype(float)
    import pandas as pd
    from anndata import AnnData

    obs = pd.DataFrame(
        {
            "cell_type": (["A"] * 40 + ["B"] * 40),
            "condition": (["ctrl", "stim"] * 40),
        }
    )
    obs["cell_type"] = obs["cell_type"].astype("category")
    obs["condition"] = obs["condition"].astype("category")
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    return AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# heatmap_expression_two_categories
# ---------------------------------------------------------------------------


def test_heatmap_returns_figure(adata_heatmap: AnnData) -> None:
    result = heatmap_expression_two_categories(
        adata_heatmap, feature="gene0", category_x="cell_type", category_y="condition"
    )
    assert isinstance(result, Figure)


def test_heatmap_returns_dataframe_when_requested(adata_heatmap: AnnData) -> None:
    result = heatmap_expression_two_categories(
        adata_heatmap,
        feature="gene0",
        category_x="cell_type",
        category_y="condition",
        return_dataframe=True,
    )
    import pandas as pd

    assert isinstance(result, tuple)
    assert len(result) == 2
    fig, df = result
    assert isinstance(fig, Figure)
    assert isinstance(df, pd.DataFrame)


def test_heatmap_zscore(adata_heatmap: AnnData) -> None:
    fig = heatmap_expression_two_categories(
        adata_heatmap,
        feature="gene0",
        category_x="cell_type",
        category_y="condition",
        use_zscores=True,
    )
    assert isinstance(fig, Figure)


def test_heatmap_invalid_feature(adata_heatmap: AnnData) -> None:
    with pytest.raises((KeyError, ValueError)):
        heatmap_expression_two_categories(
            adata_heatmap,
            feature="nonexistent_gene",
            category_x="cell_type",
            category_y="condition",
        )


def test_heatmap_invalid_category(adata_heatmap: AnnData) -> None:
    with pytest.raises((KeyError, ValueError)):
        heatmap_expression_two_categories(
            adata_heatmap,
            feature="gene0",
            category_x="nonexistent_col",
            category_y="condition",
        )


# ---------------------------------------------------------------------------
# heatmap_expression_two_categories_multiplot
# ---------------------------------------------------------------------------


def test_heatmap_multiplot_returns_figure(adata_heatmap: AnnData) -> None:
    fig = heatmap_expression_two_categories_multiplot(
        adata_heatmap,
        features=["gene0", "gene1"],
        category_x="cell_type",
        category_y="condition",
    )
    assert isinstance(fig, Figure)


def test_heatmap_multiplot_single_feature(adata_heatmap: AnnData) -> None:
    fig = heatmap_expression_two_categories_multiplot(
        adata_heatmap,
        features=["gene0"],
        category_x="cell_type",
        category_y="condition",
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# heatmap_feature_aggregated_three_categories — fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def adata_three_cat() -> AnnData:
    """360 cells with 3 conditions, 3 cell types, 2 tissues, 4 donors.

    Each (condition, donor, tissue, cell_type) combination has exactly 5 cells,
    so every group passes a min_cells threshold of 5.  Additionally the
    category "NK" is absent from tissue "liver" so that fixture exercises
    the per-panel empty-combination removal logic.
    """
    rng = np.random.default_rng(42)
    conditions = ["ctrl", "disease1", "disease2"]
    cell_types = ["T", "B", "NK"]
    tissues = ["lung", "liver"]
    donors = [f"d{i}" for i in range(4)]

    records = []
    for cond in conditions:
        for donor in donors:
            for tissue in tissues:
                for ct in cell_types:
                    # NK cells only present in lung, not liver
                    if ct == "NK" and tissue == "liver":
                        continue
                    for _ in range(5):
                        records.append(
                            {
                                "condition": cond,
                                "donor": donor,
                                "tissue": tissue,
                                "cell_type": ct,
                            }
                        )
    obs = pd.DataFrame(records)
    n_obs = len(obs)
    n_vars = 5
    X = rng.poisson(3, size=(n_obs, n_vars)).astype(float)
    for col in ["condition", "cell_type", "tissue", "donor"]:
        obs[col] = obs[col].astype("category")
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    return AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# heatmap_feature_aggregated_three_categories — tests
# ---------------------------------------------------------------------------


def test_three_cat_returns_figure(adata_three_cat: AnnData) -> None:
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
    )
    assert isinstance(fig, Figure)


def test_three_cat_show_ns(adata_three_cat: AnnData) -> None:
    """show_ns=True should not raise and should return a Figure."""
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        show_ns=True,
    )
    assert isinstance(fig, Figure)


def test_three_cat_no_stats(adata_three_cat: AnnData) -> None:
    """show_stats=False should suppress all p-value text."""
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        show_stats=False,
    )
    assert isinstance(fig, Figure)


def test_three_cat_zscores(adata_three_cat: AnnData) -> None:
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        use_zscores=True,
    )
    assert isinstance(fig, Figure)


def test_three_cat_ttest(adata_three_cat: AnnData) -> None:
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        stat_test="t-test",
    )
    assert isinstance(fig, Figure)


def test_three_cat_groups_subset(adata_three_cat: AnnData) -> None:
    """groups_x and groups_panel subsetting should work together."""
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        groups_x=["ctrl", "disease1"],
        groups_panel=["lung"],
    )
    assert isinstance(fig, Figure)


def test_three_cat_groups_x_without_ref_still_includes_ref(
    adata_three_cat: AnnData,
) -> None:
    """x_ref is always retained even when omitted from groups_x."""
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        # groups_x does not include "ctrl"
        groups_x=["disease1", "disease2"],
    )
    assert isinstance(fig, Figure)
    # The number of x-tick labels on any axes should equal 3 (ctrl included)
    axes = fig.get_axes()
    heatmap_axes = [ax for ax in axes if len(ax.get_xticks()) > 1]
    if heatmap_axes:
        assert len(heatmap_axes[0].get_xticks()) == 3


def test_three_cat_explicit_order(adata_three_cat: AnnData) -> None:
    """groups_* lists control both subset and display order."""
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        groups_x=["ctrl", "disease2", "disease1"],
        groups_y=["NK", "B", "T"],
        groups_panel=["liver", "lung"],
    )
    assert isinstance(fig, Figure)


def test_three_cat_single_panel(adata_three_cat: AnnData) -> None:
    """A single panel value should produce a figure without errors."""
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        groups_panel=["lung"],
    )
    assert isinstance(fig, Figure)


def test_three_cat_shared_colorscale_false(adata_three_cat: AnnData) -> None:
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        shared_colorscale=False,
    )
    assert isinstance(fig, Figure)


def test_three_cat_invalid_x_ref(adata_three_cat: AnnData) -> None:
    with pytest.raises(ValueError, match="x_ref"):
        heatmap_feature_aggregated_three_categories(
            adata_three_cat,
            feature="gene0",
            x="condition",
            y="cell_type",
            panel="tissue",
            sample_col="donor",
            x_ref="nonexistent",
            min_cells=5,
        )


def test_three_cat_invalid_groups_x(adata_three_cat: AnnData) -> None:
    with pytest.raises(ValueError, match="groups_x"):
        heatmap_feature_aggregated_three_categories(
            adata_three_cat,
            feature="gene0",
            x="condition",
            y="cell_type",
            panel="tissue",
            sample_col="donor",
            x_ref="ctrl",
            min_cells=5,
            groups_x=["ctrl", "nonexistent"],
        )


def test_three_cat_invalid_column(adata_three_cat: AnnData) -> None:
    with pytest.raises(ValueError):
        heatmap_feature_aggregated_three_categories(
            adata_three_cat,
            feature="gene0",
            x="no_such_col",
            y="cell_type",
            panel="tissue",
            sample_col="donor",
            x_ref="ctrl",
            min_cells=5,
        )


def test_three_cat_obs_feature(adata_three_cat: AnnData) -> None:
    """Feature can be an adata.obs column (numeric)."""
    adata_three_cat.obs["score"] = np.random.default_rng(1).normal(
        size=len(adata_three_cat)
    )
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="score",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
    )
    assert isinstance(fig, Figure)


def test_three_cat_median_agg(adata_three_cat: AnnData) -> None:
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        agg_fn="median",
    )
    assert isinstance(fig, Figure)


def test_three_cat_min_samples_gray(adata_three_cat: AnnData) -> None:
    """min_samples should gray out low-sample cells without raising."""
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        min_samples=2,
    )
    assert isinstance(fig, Figure)


def test_three_cat_min_samples_none(adata_three_cat: AnnData) -> None:
    """min_samples=None (default) disables masking without raising."""
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        min_samples=None,
        show_ns=True,
    )
    assert isinstance(fig, Figure)


def test_three_cat_min_cells_too_strict(adata_three_cat: AnnData) -> None:
    with pytest.raises(ValueError, match="min_cells"):
        heatmap_feature_aggregated_three_categories(
            adata_three_cat,
            feature="gene0",
            x="condition",
            y="cell_type",
            panel="tissue",
            sample_col="donor",
            x_ref="ctrl",
            min_cells=9999,
        )


def test_three_cat_per_panel_empty_combinations_removed(
    adata_three_cat: AnnData,
) -> None:
    """Each panel should only show the y categories that have data in it.

    The fixture has NK cells only in 'lung', not 'liver'.  The 'liver' panel
    should therefore have 2 y-tick labels (T, B) while 'lung' has 3 (T, B, NK).
    """
    fig = heatmap_feature_aggregated_three_categories(
        adata_three_cat,
        feature="gene0",
        x="condition",
        y="cell_type",
        panel="tissue",
        sample_col="donor",
        x_ref="ctrl",
        min_cells=5,
        border_ticks_only=False,  # labels visible on every panel
    )
    axes = fig.get_axes()
    # Collect heatmap axes (exclude colourbar axes which have no ytick labels)
    heatmap_axes = [ax for ax in axes if len(ax.get_yticks()) > 1]
    # Find panel axes by title
    lung_ax = next((ax for ax in heatmap_axes if ax.get_title() == "lung"), None)
    liver_ax = next((ax for ax in heatmap_axes if ax.get_title() == "liver"), None)
    assert lung_ax is not None, "lung panel not found"
    assert liver_ax is not None, "liver panel not found"
    assert len(lung_ax.get_yticks()) == 3, "lung should show T, B, NK"
    assert len(liver_ax.get_yticks()) == 2, "liver should show only T, B"
