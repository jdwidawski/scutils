"""Shared pytest fixtures for all test suites."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Small synthetic AnnData fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded RNG for reproducible synthetic data."""
    return np.random.default_rng(42)


@pytest.fixture
def adata_basic(rng: np.random.Generator) -> AnnData:
    """80 cells × 10 genes; obs columns: cell_type (4 cats), condition (2 cats)."""
    n_cells, n_genes = 80, 10
    X = csr_matrix(rng.poisson(1.5, size=(n_cells, n_genes)).astype(np.float32))
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                np.tile(["T", "B", "NK", "Mono"], n_cells // 4)
            ),
            "condition": pd.Categorical(
                np.tile(["ctrl", "stim"], n_cells // 2)
            ),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_name": [f"Gene{i}" for i in range(n_genes)]},
        index=[f"gene_{i}" for i in range(n_genes)],
    )
    adata = AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return adata


@pytest.fixture
def adata_donors(rng: np.random.Generator) -> AnnData:
    """320 cells: 4 cell_types × 4 donors × 2 conditions × 10 cells each."""
    n_genes = 10
    cell_types = ["T", "B", "NK", "Mono"]
    donors = ["D1", "D2", "D3", "D4"]
    conditions = ["ctrl", "stim"]
    records = []
    for ct in cell_types:
        for don in donors:
            for cond in conditions:
                for _ in range(10):
                    records.append({"cell_type": ct, "donor": don, "condition": cond})
    obs = pd.DataFrame(records)
    obs["cell_type"] = pd.Categorical(obs["cell_type"])
    obs["donor"] = pd.Categorical(obs["donor"])
    obs["condition"] = pd.Categorical(obs["condition"])
    n_cells = len(obs)
    X = csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32))
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    obs.index = [f"cell_{i}" for i in range(n_cells)]
    adata = AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def adata_umap(rng: np.random.Generator) -> AnnData:
    """120 cells with UMAP coordinates and a leiden clustering."""
    adata = sc.datasets.pbmc68k_reduced()
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="leiden", resolution=0.5)


@pytest.fixture
def adata_pseudobulk_ready(rng: np.random.Generator) -> AnnData:
    """600 cells designed for pseudobulk + DE testing.

    6 donors × 2 conditions × 50 cells each = 600 cells × 30 genes.
    Uses Poisson counts with a small fold-change injected into the first
    5 genes for the 'treated' condition so that DE testing produces at
    least some signal.
    """
    n_genes = 30
    donors = ["D1", "D2", "D3", "D4", "D5", "D6"]
    conditions = ["control", "treated"]
    records = []
    x_rows = []
    for donor in donors:
        for cond in conditions:
            mean = 50 if cond == "control" else 50
            for _ in range(50):
                row = rng.poisson(mean, size=n_genes)
                # Inject fold-change in genes 0-4 for treated
                if cond == "treated":
                    row[:5] = rng.poisson(120, size=5)
                x_rows.append(row)
                records.append({"donor": donor, "condition": cond})

    X = csr_matrix(np.vstack(x_rows).astype(np.float32))
    obs = pd.DataFrame(records)
    obs["donor"] = pd.Categorical(obs["donor"])
    obs["condition"] = pd.Categorical(obs["condition"])
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = AnnData(X=X, obs=obs, var=var)
    return adata
    return adata
