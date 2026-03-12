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
    plot_embedding_categories,
    plot_density_outlines,
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
    # Pre-compute a fake density column for plot_density_outlines
    obs["umap_density_cell_type"] = rng.uniform(0.0, 5.0, size=n_obs)
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


def test_plot_embedding_categories_no_others(adata_density: AnnData) -> None:
    fig = plot_embedding_categories(
        adata_density,
        category_dict={"cell_type": ["T", "B"]},
        show_others=False,
    )
    assert isinstance(fig, Figure)


def test_plot_embedding_categories_invalid_category(adata_density: AnnData) -> None:
    with pytest.raises(KeyError):
        plot_embedding_categories(
            adata_density,
            category_dict={"nonexistent_col": ["T"]},
        )


def test_plot_embedding_categories_invalid_basis(adata_density: AnnData) -> None:
    with pytest.raises(KeyError):
        plot_embedding_categories(
            adata_density,
            category_dict={"cell_type": ["T"]},
            basis="pca",  # not in adata.obsm
        )


# ---------------------------------------------------------------------------
# plot_density_outlines
# ---------------------------------------------------------------------------


def test_plot_density_outlines_returns_figure(adata_density: AnnData) -> None:
    fig = plot_density_outlines(
        adata_density,
        category_dict={"cell_type": ["T", "B"]},
        density_colname="umap_density",
        density_cutoff=1.0,
    )
    assert isinstance(fig, Figure)


def test_plot_density_outlines_custom_figsize(adata_density: AnnData) -> None:
    fig = plot_density_outlines(
        adata_density,
        category_dict={"cell_type": ["T"]},
        density_colname="umap_density",
        figsize=(10.0, 8.0),
    )
    assert isinstance(fig, Figure)
    w, h = fig.get_size_inches()
    assert abs(w - 10.0) < 1e-6
    assert abs(h - 8.0) < 1e-6
