"""Tests for scutils.plotting.heatmaps."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from anndata import AnnData
from matplotlib.figure import Figure

matplotlib.use("Agg")

from scutils.plotting.heatmaps import (
    heatmap_expression_two_categories,
    heatmap_expression_two_categories_multiplot,
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
