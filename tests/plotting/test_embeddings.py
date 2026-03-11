"""Tests for scutils.plotting.embeddings."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from matplotlib.figure import Figure

matplotlib.use("Agg")

from scutils.plotting.embeddings import (
    embedding_category_multiplot,
    embedding_gene_expression_multiplot,
    flatten_list_of_lists,
    is_sequence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adata_umap() -> AnnData:
    """pbmc68k_reduced with UMAP and leiden clustering."""
    adata = sc.datasets.pbmc68k_reduced()
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="leiden")
    return adata


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


def test_is_sequence_list() -> None:
    assert is_sequence([1, 2, 3]) is True


def test_is_sequence_str() -> None:
    # str supports __len__ and __getitem__ — this is intentional behaviour
    assert is_sequence("hello") is True


def test_is_sequence_int() -> None:
    assert is_sequence(42) is False


def test_flatten_list_of_lists() -> None:
    assert flatten_list_of_lists([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]


def test_flatten_empty() -> None:
    assert flatten_list_of_lists([]) == []


# ---------------------------------------------------------------------------
# embedding_category_multiplot
# ---------------------------------------------------------------------------


def test_embedding_category_multiplot_returns_figure(adata_umap: AnnData) -> None:
    fig = embedding_category_multiplot(adata_umap, column="leiden")
    assert isinstance(fig, Figure)


def test_embedding_category_multiplot_palette_list(adata_umap: AnnData) -> None:
    n_cats = adata_umap.obs["leiden"].nunique()
    palette = ["#ff0000"] * n_cats
    fig = embedding_category_multiplot(adata_umap, column="leiden", palette=palette)
    assert isinstance(fig, Figure)


def test_embedding_category_multiplot_palette_single_colour(
    adata_umap: AnnData,
) -> None:
    fig = embedding_category_multiplot(
        adata_umap, column="leiden", palette="steelblue"
    )
    assert isinstance(fig, Figure)


def test_embedding_category_multiplot_groups_subset(adata_umap: AnnData) -> None:
    cats = adata_umap.obs["leiden"].cat.categories[:2].tolist()
    fig = embedding_category_multiplot(
        adata_umap, column="leiden", groups=cats, ncols=2
    )
    assert isinstance(fig, Figure)


def test_embedding_category_multiplot_invalid_column(adata_umap: AnnData) -> None:
    with pytest.raises(ValueError, match="not found"):
        embedding_category_multiplot(adata_umap, column="nonexistent_col")


# ---------------------------------------------------------------------------
# embedding_gene_expression_multiplot
# ---------------------------------------------------------------------------


def test_embedding_gene_expression_multiplot_returns_figure(
    adata_umap: AnnData,
) -> None:
    gene = adata_umap.var_names[0]
    fig = embedding_gene_expression_multiplot(
        adata_umap, column="leiden", feature=gene
    )
    assert isinstance(fig, Figure)


def test_embedding_gene_expression_multiplot_obs_feature(
    adata_umap: AnnData,
) -> None:
    fig = embedding_gene_expression_multiplot(
        adata_umap, column="leiden", feature="n_counts"
    )
    assert isinstance(fig, Figure)


def test_embedding_gene_expression_multiplot_invalid_column(
    adata_umap: AnnData,
) -> None:
    gene = adata_umap.var_names[0]
    with pytest.raises(ValueError, match="not found"):
        embedding_gene_expression_multiplot(
            adata_umap, column="nonexistent_col", feature=gene
        )


def test_embedding_gene_expression_multiplot_invalid_feature(
    adata_umap: AnnData,
) -> None:
    with pytest.raises(ValueError, match="not found"):
        embedding_gene_expression_multiplot(
            adata_umap, column="leiden", feature="NOT_A_GENE"
        )
