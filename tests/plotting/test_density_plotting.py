"""Tests for scutils.plotting.density_plotting."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.figure import Figure

matplotlib.use("Agg")

from scutils.plotting.density_plotting import (
    plot_density_embedding,
    plot_density_embedding_comparison,
    plot_density_embedding_multiplot,
    plot_embedding_categories,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adata_density() -> AnnData:
    """160 cells × 5 genes with a 2-D UMAP embedding and categorical obs."""
    rng = np.random.default_rng(1)
    n_obs, n_vars = 160, 5
    X = rng.poisson(2, size=(n_obs, n_vars)).astype(float)
    umap = rng.normal(size=(n_obs, 2)).astype(float)
    obs = pd.DataFrame(
        {
            "cell_type": (["T"] * 40 + ["B"] * 40 + ["NK"] * 40 + ["Mono"] * 40),
            "condition": (["ctrl", "stim"] * 80),
        }
    )
    obs["cell_type"] = obs["cell_type"].astype("category")
    obs["condition"] = obs["condition"].astype("category")
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = umap
    return adata


# ---------------------------------------------------------------------------
# plot_embedding_categories
# ---------------------------------------------------------------------------


def test_plot_embedding_categories_returns_figure(adata_density: AnnData) -> None:
    fig = plot_embedding_categories(
        adata_density,
        category_dict={"cell_type": ["T", "B"]},
    )
    assert isinstance(fig, Figure)


def test_plot_embedding_categories_multiple_columns(adata_density: AnnData) -> None:
    fig = plot_embedding_categories(
        adata_density,
        category_dict={"cell_type": ["T"], "condition": ["ctrl"]},
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_density_embedding
# ---------------------------------------------------------------------------


def test_plot_density_embedding_returns_figure(adata_density: AnnData) -> None:
    fig = plot_density_embedding(
        adata_density,
        category_col="cell_type",
        category_value="T",
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_density_embedding_multiplot
# ---------------------------------------------------------------------------


def test_plot_density_embedding_multiplot_returns_figure(
    adata_density: AnnData,
) -> None:
    fig = plot_density_embedding_multiplot(
        adata_density,
        category_col="cell_type",
        category_values=["T", "B"],
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_density_embedding_comparison
# ---------------------------------------------------------------------------


def test_plot_density_embedding_comparison_returns_figure(
    adata_density: AnnData,
) -> None:
    fig = plot_density_embedding_comparison(
        adata_density,
        category_col="cell_type",
        reference_value="T",
        comparison_values=["B", "NK"],
    )
    assert isinstance(fig, Figure)
