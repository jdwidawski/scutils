"""Tests for scutils.preprocessing.concatenation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from scutils.preprocessing.concatenation import (
    concat_anndata_with_zeros,
    filter_genes_by_presence,
    get_datasets_missing_gene,
    get_zero_filled_genes_for_dataset,
    get_zero_filling_stats,
    print_zero_filling_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adata_A() -> AnnData:
    """3 cells × genes [A, B, C] (dense)."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    obs = pd.DataFrame(index=["c1", "c2", "c3"])
    var = pd.DataFrame(index=["A", "B", "C"])
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata_B() -> AnnData:
    """3 cells × genes [B, C, D] (dense) — overlapping genes B, C."""
    X = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
    obs = pd.DataFrame(index=["c4", "c5", "c6"])
    var = pd.DataFrame(index=["B", "C", "D"])
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata_A_sparse() -> AnnData:
    """Sparse version of adata_A."""
    X = csr_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    obs = pd.DataFrame(index=["c1", "c2", "c3"])
    var = pd.DataFrame(index=["A", "B", "C"])
    return AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# concat_anndata_with_zeros
# ---------------------------------------------------------------------------


def test_concat_shape(adata_A: AnnData, adata_B: AnnData) -> None:
    combined = concat_anndata_with_zeros(adata_A, adata_B)
    # 3 + 3 cells; A ∪ B genes = {A, B, C, D}
    assert combined.n_obs == 6
    assert combined.n_vars == 4


def test_concat_all_gene_names_present(adata_A: AnnData, adata_B: AnnData) -> None:
    combined = concat_anndata_with_zeros(adata_A, adata_B)
    assert set(combined.var_names) == {"A", "B", "C", "D"}


def test_concat_zero_filling_table_exists(adata_A: AnnData, adata_B: AnnData) -> None:
    combined = concat_anndata_with_zeros(adata_A, adata_B)
    assert "zero_filled_genes" in combined.uns


def test_concat_zero_filling_correct_entries(
    adata_A: AnnData, adata_B: AnnData
) -> None:
    combined = concat_anndata_with_zeros(
        adata_A, adata_B, left_name="A", right_name="B"
    )
    table = combined.uns["zero_filled_genes"]
    # Gene "D" is only in B → should be zero-filled in A
    assert table.loc["D", "A"] is True or table.loc["D", "A"] == True  # noqa: E712
    # Gene "A" is only in A → should be zero-filled in B
    assert table.loc["A", "B"] is True or table.loc["A", "B"] == True  # noqa: E712
    # Gene "B" is in both → not zero-filled in either
    assert not table.loc["B", "A"]
    assert not table.loc["B", "B"]


def test_concat_sparse_input(adata_A_sparse: AnnData, adata_B: AnnData) -> None:
    combined = concat_anndata_with_zeros(adata_A_sparse, adata_B)
    assert combined.n_obs == 6
    assert combined.n_vars == 4


def test_concat_incremental(adata_A: AnnData, adata_B: AnnData) -> None:
    """Concatenating three datasets incrementally should preserve tracking."""
    import pandas as pd

    adata_C_X = np.array([[19.0, 20.0]])
    adata_C = AnnData(
        X=adata_C_X,
        obs=pd.DataFrame(index=["c7"]),
        var=pd.DataFrame(index=["C"]),
    )
    combined_ab = concat_anndata_with_zeros(
        adata_A, adata_B, left_name="A", right_name="B"
    )
    combined_abc = concat_anndata_with_zeros(
        combined_ab, adata_C, left_name=None, right_name="C"
    )
    assert combined_abc.n_obs == 7
    assert "C" in combined_abc.uns["zero_filled_genes"].columns


# ---------------------------------------------------------------------------
# get_zero_filled_genes_for_dataset
# ---------------------------------------------------------------------------


def test_get_zero_filled_genes_for_dataset(
    adata_A: AnnData, adata_B: AnnData
) -> None:
    combined = concat_anndata_with_zeros(
        adata_A, adata_B, left_name="A", right_name="B"
    )
    zero_filled_in_B = get_zero_filled_genes_for_dataset(combined, "B")
    assert "A" in zero_filled_in_B


def test_get_zero_filled_genes_missing_dataset(
    adata_A: AnnData, adata_B: AnnData
) -> None:
    combined = concat_anndata_with_zeros(adata_A, adata_B)
    result = get_zero_filled_genes_for_dataset(combined, "nonexistent")
    assert result == []


# ---------------------------------------------------------------------------
# get_datasets_missing_gene
# ---------------------------------------------------------------------------


def test_get_datasets_missing_gene(adata_A: AnnData, adata_B: AnnData) -> None:
    combined = concat_anndata_with_zeros(
        adata_A, adata_B, left_name="A", right_name="B"
    )
    # Gene "D" is only in B → "A" should have it zero-filled
    missing = get_datasets_missing_gene(combined, "D")
    assert "A" in missing


def test_get_datasets_missing_gene_absent(
    adata_A: AnnData, adata_B: AnnData
) -> None:
    combined = concat_anndata_with_zeros(adata_A, adata_B)
    result = get_datasets_missing_gene(combined, "NONEXISTENT")
    assert result == []


# ---------------------------------------------------------------------------
# get_zero_filling_stats
# ---------------------------------------------------------------------------


def test_get_zero_filling_stats_shape(adata_A: AnnData, adata_B: AnnData) -> None:
    combined = concat_anndata_with_zeros(
        adata_A, adata_B, left_name="A", right_name="B"
    )
    stats = get_zero_filling_stats(combined)
    assert isinstance(stats, pd.DataFrame)
    assert "n_zero_filled" in stats.columns
    assert len(stats) == 2  # two datasets: A and B


def test_get_zero_filling_stats_no_info() -> None:
    adata = AnnData(np.zeros((5, 3)))
    result = get_zero_filling_stats(adata)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# print_zero_filling_summary (smoke test — just check no exceptions)
# ---------------------------------------------------------------------------


def test_print_zero_filling_summary_runs(
    adata_A: AnnData, adata_B: AnnData, capsys
) -> None:
    combined = concat_anndata_with_zeros(
        adata_A, adata_B, left_name="A", right_name="B"
    )
    print_zero_filling_summary(combined)
    captured = capsys.readouterr()
    assert "ZERO-FILLING SUMMARY" in captured.out


def test_print_zero_filling_summary_no_info(capsys) -> None:
    adata = AnnData(np.zeros((5, 3)))
    print_zero_filling_summary(adata)
    captured = capsys.readouterr()
    assert "No zero filling" in captured.out


# ---------------------------------------------------------------------------
# filter_genes_by_presence
# ---------------------------------------------------------------------------


def test_filter_genes_by_presence_removes_exclusive_genes(
    adata_A: AnnData, adata_B: AnnData
) -> None:
    combined = concat_anndata_with_zeros(
        adata_A, adata_B, left_name="A", right_name="B"
    )
    # min_datasets=2 should keep only B and C (present in both A and B)
    filtered = filter_genes_by_presence(combined, min_datasets=2)
    assert set(filtered.var_names) == {"B", "C"}


def test_filter_genes_by_presence_keeps_all_with_min_1(
    adata_A: AnnData, adata_B: AnnData
) -> None:
    combined = concat_anndata_with_zeros(
        adata_A, adata_B, left_name="A", right_name="B"
    )
    filtered = filter_genes_by_presence(combined, min_datasets=1)
    assert filtered.n_vars == 4  # all genes present in at least one dataset


def test_filter_genes_table_updated(adata_A: AnnData, adata_B: AnnData) -> None:
    combined = concat_anndata_with_zeros(
        adata_A, adata_B, left_name="A", right_name="B"
    )
    filtered = filter_genes_by_presence(combined, min_datasets=2)
    table = filtered.uns["zero_filled_genes"]
    assert set(table.index) == {"B", "C"}
