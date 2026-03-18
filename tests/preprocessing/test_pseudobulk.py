"""Tests for scutils.preprocessing.pseudobulk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from scutils.preprocessing.pseudobulk import pseudobulk, _check_raw_counts


# ---------------------------------------------------------------------------
# Raw count validation
# ---------------------------------------------------------------------------


class TestCheckRawCounts:
    def test_valid_integer_counts(self):
        """No error for a well-behaved integer count matrix."""
        X = csr_matrix(np.array([[1, 0, 3], [0, 5, 2]], dtype=np.float32))
        _check_raw_counts(X)  # should not raise

    def test_negative_values_raise(self):
        X = np.array([[1, -1, 3], [0, 5, 2]], dtype=np.float32)
        with pytest.raises(ValueError, match="negative values"):
            _check_raw_counts(X)

    def test_float_values_raise(self):
        X = np.array([[1.5, 0.0, 3.2]], dtype=np.float32)
        with pytest.raises(ValueError, match="non-integer values"):
            _check_raw_counts(X)

    def test_empty_matrix_does_not_raise(self):
        X = csr_matrix((0, 5), dtype=np.float32)
        _check_raw_counts(X)  # should not raise


# ---------------------------------------------------------------------------
# Pseudobulk aggregation — basic
# ---------------------------------------------------------------------------


class TestPseudobulkBasic:
    def test_returns_anndata(self, adata_pseudobulk_ready):
        pb = pseudobulk(adata_pseudobulk_ready, sample_col="donor")
        assert isinstance(pb, AnnData)

    def test_output_is_dense(self, adata_pseudobulk_ready):
        pb = pseudobulk(adata_pseudobulk_ready, sample_col="donor")
        assert not issparse(pb.X)

    def test_shape_sample_only(self, adata_pseudobulk_ready):
        """6 donors → 6 pseudobulk observations, 30 genes."""
        pb = pseudobulk(adata_pseudobulk_ready, sample_col="donor")
        assert pb.n_obs == 6
        assert pb.n_vars == 30

    def test_shape_sample_and_group(self, adata_pseudobulk_ready):
        """6 donors × 2 conditions → 12 observations."""
        pb = pseudobulk(
            adata_pseudobulk_ready,
            sample_col="donor",
            groups_col="condition",
        )
        assert pb.n_obs == 12
        assert pb.n_vars == 30

    def test_total_counts_preserved(self, adata_pseudobulk_ready):
        """Summed counts across pseudobulk == total counts in original."""
        pb = pseudobulk(adata_pseudobulk_ready, sample_col="donor")
        original_total = np.asarray(adata_pseudobulk_ready.X.sum()).item()
        pb_total = pb.X.sum()
        assert abs(original_total - pb_total) < 1.0

    def test_n_cells_column_present(self, adata_pseudobulk_ready):
        pb = pseudobulk(adata_pseudobulk_ready, sample_col="donor")
        assert "n_cells" in pb.obs.columns
        assert pb.obs["n_cells"].sum() == adata_pseudobulk_ready.n_obs

    def test_var_index_preserved(self, adata_pseudobulk_ready):
        pb = pseudobulk(adata_pseudobulk_ready, sample_col="donor")
        assert list(pb.var_names) == list(adata_pseudobulk_ready.var_names)


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


class TestPseudobulkMetadata:
    def test_condition_propagated_with_groups(self, adata_pseudobulk_ready):
        """When groups_col='condition', condition column should be propagated."""
        pb = pseudobulk(
            adata_pseudobulk_ready,
            sample_col="donor",
            groups_col="condition",
        )
        assert "condition" in pb.obs.columns
        assert pb.obs["condition"].notna().all()

    def test_donor_in_obs(self, adata_pseudobulk_ready):
        pb = pseudobulk(adata_pseudobulk_ready, sample_col="donor")
        assert "donor" in pb.obs.columns


# ---------------------------------------------------------------------------
# min_cells filtering
# ---------------------------------------------------------------------------


class TestPseudobulkMinCells:
    def test_min_cells_filters_small_groups(self, rng):
        """Groups below min_cells threshold are dropped."""
        n_genes = 5
        X = csr_matrix(
            rng.poisson(5, size=(30, n_genes)).astype(np.float32)
        )
        obs = pd.DataFrame(
            {
                "sample": pd.Categorical(
                    ["S1"] * 20 + ["S2"] * 5 + ["S3"] * 5
                ),
            },
            index=[f"c{i}" for i in range(30)],
        )
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
        adata = AnnData(X=X, obs=obs, var=var)

        pb = pseudobulk(adata, sample_col="sample", min_cells=10)
        assert pb.n_obs == 1  # only S1 passes
        assert pb.obs["n_cells"].values[0] == 20

    def test_all_groups_filtered_raises(self, rng):
        """ValueError when no group passes the threshold."""
        X = csr_matrix(rng.poisson(5, size=(6, 3)).astype(np.float32))
        obs = pd.DataFrame(
            {"sample": pd.Categorical(["S1"] * 3 + ["S2"] * 3)},
            index=[f"c{i}" for i in range(6)],
        )
        var = pd.DataFrame(index=["g0", "g1", "g2"])
        adata = AnnData(X=X, obs=obs, var=var)

        with pytest.raises(ValueError, match="min_cells"):
            pseudobulk(adata, sample_col="sample", min_cells=100)


# ---------------------------------------------------------------------------
# Layer support
# ---------------------------------------------------------------------------


class TestPseudobulkLayer:
    def test_layer_used_instead_of_X(self, rng):
        """When layer is specified, counts are taken from that layer."""
        n_cells, n_genes = 20, 5
        raw_counts = rng.poisson(10, size=(n_cells, n_genes)).astype(
            np.float32
        )
        # .X is normalised (non-integer) → would fail count check
        X_norm = raw_counts / raw_counts.sum(axis=1, keepdims=True) * 1e4

        obs = pd.DataFrame(
            {"sample": pd.Categorical(["S1"] * 10 + ["S2"] * 10)},
            index=[f"c{i}" for i in range(n_cells)],
        )
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
        adata = AnnData(X=X_norm, obs=obs, var=var)
        adata.layers["counts"] = csr_matrix(raw_counts)

        pb = pseudobulk(adata, sample_col="sample", layer="counts")
        assert pb.n_obs == 2
        # Sums should match the raw counts
        expected_total = raw_counts.sum()
        assert abs(pb.X.sum() - expected_total) < 1.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestPseudobulkValidation:
    def test_invalid_sample_col(self, adata_pseudobulk_ready):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            pseudobulk(adata_pseudobulk_ready, sample_col="nonexistent")

    def test_invalid_groups_col(self, adata_pseudobulk_ready):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            pseudobulk(
                adata_pseudobulk_ready,
                sample_col="donor",
                groups_col="nonexistent",
            )

    def test_normalised_X_raises(self, rng):
        """Non-integer .X should be rejected."""
        X = rng.random((20, 5)).astype(np.float32)
        obs = pd.DataFrame(
            {"sample": pd.Categorical(["S1"] * 10 + ["S2"] * 10)},
            index=[f"c{i}" for i in range(20)],
        )
        var = pd.DataFrame(index=[f"g{i}" for i in range(5)])
        adata = AnnData(X=X, obs=obs, var=var)

        with pytest.raises(ValueError, match="non-integer"):
            pseudobulk(adata, sample_col="sample")

    def test_skip_count_check(self, rng):
        """skip_count_check=True bypasses the validation."""
        X = rng.random((20, 5)).astype(np.float32)
        obs = pd.DataFrame(
            {"sample": pd.Categorical(["S1"] * 10 + ["S2"] * 10)},
            index=[f"c{i}" for i in range(20)],
        )
        var = pd.DataFrame(index=[f"g{i}" for i in range(5)])
        adata = AnnData(X=X, obs=obs, var=var)

        pb = pseudobulk(
            adata, sample_col="sample", skip_count_check=True
        )
        assert pb.n_obs == 2
