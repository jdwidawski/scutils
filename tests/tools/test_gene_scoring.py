"""Tests for scutils.tools.gene_scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from scutils.tools.gene_scoring import (
    _DECOUPLER_AVAILABLE,
    _HOTSPOT_AVAILABLE,
    _PYUCELL_AVAILABLE,
    _resolve_genes,
    compute_aucell_scores,
    compute_hotspot_scores,
    compute_ulm_scores,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_CELLS = 80
N_GENES = 20
GENE_NAMES = [f"gene_{i}" for i in range(N_GENES)]


@pytest.fixture
def scoring_adata() -> AnnData:
    """Sparse AnnData (80 cells × 20 genes) with a PCA embedding."""
    rng = np.random.default_rng(0)
    X = rng.negative_binomial(5, 0.3, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(N_CELLS)])
    var = pd.DataFrame(index=GENE_NAMES)
    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)
    adata.obsm["X_pca"] = rng.standard_normal((N_CELLS, 10)).astype(np.float32)
    return adata


# ---------------------------------------------------------------------------
# _resolve_genes
# ---------------------------------------------------------------------------


class TestResolveGenes:
    def test_no_gene_symbols_returns_genes_as_is(
        self, scoring_adata: AnnData
    ) -> None:
        genes = ["gene_0", "gene_1", "gene_2"]
        result = _resolve_genes(scoring_adata, genes, gene_symbols=None)
        assert result == genes

    def test_gene_symbols_maps_to_var_names(
        self, scoring_adata: AnnData
    ) -> None:
        scoring_adata.var["symbol"] = [f"GENE_{i}" for i in range(N_GENES)]
        symbol_genes = ["GENE_0", "GENE_1", "GENE_2"]
        result = _resolve_genes(
            scoring_adata, symbol_genes, gene_symbols="symbol"
        )
        assert result == ["gene_0", "gene_1", "gene_2"]

    def test_gene_symbols_unknown_gene_excluded(
        self, scoring_adata: AnnData
    ) -> None:
        scoring_adata.var["symbol"] = [f"GENE_{i}" for i in range(N_GENES)]
        symbol_genes = ["GENE_0", "NONEXISTENT"]
        result = _resolve_genes(
            scoring_adata, symbol_genes, gene_symbols="symbol"
        )
        assert result == ["gene_0"]
        assert "NONEXISTENT" not in result

    def test_empty_genes_returns_empty(self, scoring_adata: AnnData) -> None:
        result = _resolve_genes(scoring_adata, [], gene_symbols=None)
        assert result == []


# ---------------------------------------------------------------------------
# compute_hotspot_scores
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HOTSPOT_AVAILABLE,
    reason="hotspot-sc / pynndescent not installed",
)
class TestComputeHotspotScores:
    GENES = GENE_NAMES[:5]

    def test_inplace_returns_none(self, scoring_adata: AnnData) -> None:
        result = compute_hotspot_scores(scoring_adata, genes=self.GENES, use_rep=None)
        assert result is None

    def test_inplace_adds_obs_column(self, scoring_adata: AnnData) -> None:
        compute_hotspot_scores(scoring_adata, genes=self.GENES, use_rep=None)
        assert "hotspot_score" in scoring_adata.obs.columns

    def test_copy_returns_adata(self, scoring_adata: AnnData) -> None:
        result = compute_hotspot_scores(
            scoring_adata, genes=self.GENES, use_rep=None, copy=True
        )
        assert isinstance(result, AnnData)
        # original must not be modified
        assert "hotspot_score" not in scoring_adata.obs.columns
        assert "hotspot_score" in result.obs.columns

    def test_custom_score_name(self, scoring_adata: AnnData) -> None:
        compute_hotspot_scores(
            scoring_adata, genes=self.GENES, use_rep=None, score_name="my_score"
        )
        assert "my_score" in scoring_adata.obs.columns

    def test_scores_are_finite_floats(self, scoring_adata: AnnData) -> None:
        compute_hotspot_scores(scoring_adata, genes=self.GENES, use_rep=None)
        scores = scoring_adata.obs["hotspot_score"]
        assert np.isfinite(scores).all()

    def test_score_length_matches_n_obs(self, scoring_adata: AnnData) -> None:
        compute_hotspot_scores(scoring_adata, genes=self.GENES, use_rep=None)
        assert len(scoring_adata.obs["hotspot_score"]) == scoring_adata.n_obs

    def test_with_pca_rep(self, scoring_adata: AnnData) -> None:
        compute_hotspot_scores(
            scoring_adata, genes=self.GENES, use_rep="X_pca"
        )
        assert "hotspot_score" in scoring_adata.obs.columns

    def test_with_layer(self, scoring_adata: AnnData) -> None:
        scoring_adata.layers["counts"] = scoring_adata.X.copy()
        compute_hotspot_scores(
            scoring_adata, genes=self.GENES, use_rep=None, layer="counts"
        )
        assert "hotspot_score" in scoring_adata.obs.columns

    def test_gene_symbols_column(self, scoring_adata: AnnData) -> None:
        scoring_adata.var["symbol"] = [f"GENE_{i}" for i in range(N_GENES)]
        symbol_genes = [f"GENE_{i}" for i in range(5)]
        compute_hotspot_scores(
            scoring_adata,
            genes=symbol_genes,
            use_rep=None,
            gene_symbols="symbol",
        )
        assert "hotspot_score" in scoring_adata.obs.columns

    def test_missing_dep_raises(
        self, scoring_adata: AnnData, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scutils.tools.gene_scoring as gs

        monkeypatch.setattr(gs, "_HOTSPOT_AVAILABLE", False)
        with pytest.raises(ImportError, match="hotspot-sc"):
            compute_hotspot_scores(scoring_adata, genes=self.GENES, use_rep=None)


# ---------------------------------------------------------------------------
# compute_ulm_scores
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DECOUPLER_AVAILABLE,
    reason="decoupler not installed",
)
class TestComputeUlmScores:
    GENES = GENE_NAMES[:8]

    def test_inplace_returns_none(self, scoring_adata: AnnData) -> None:
        result = compute_ulm_scores(scoring_adata, genes=self.GENES)
        assert result is None

    def test_inplace_adds_obs_column(self, scoring_adata: AnnData) -> None:
        compute_ulm_scores(scoring_adata, genes=self.GENES, score_name="tf")
        assert "tf" in scoring_adata.obs.columns

    def test_copy_returns_adata(self, scoring_adata: AnnData) -> None:
        result = compute_ulm_scores(
            scoring_adata, genes=self.GENES, score_name="tf", copy=True
        )
        assert isinstance(result, AnnData)
        assert "tf" not in scoring_adata.obs.columns
        assert "tf" in result.obs.columns

    def test_scores_are_finite(self, scoring_adata: AnnData) -> None:
        compute_ulm_scores(scoring_adata, genes=self.GENES, score_name="tf")
        assert np.isfinite(scoring_adata.obs["tf"]).all()

    def test_score_length_matches_n_obs(self, scoring_adata: AnnData) -> None:
        compute_ulm_scores(scoring_adata, genes=self.GENES, score_name="tf")
        assert len(scoring_adata.obs["tf"]) == scoring_adata.n_obs

    def test_return_pvals_false_no_pval_column(
        self, scoring_adata: AnnData
    ) -> None:
        compute_ulm_scores(
            scoring_adata, genes=self.GENES, score_name="tf", return_pvals=False
        )
        assert "tf_pval" not in scoring_adata.obs.columns

    def test_return_pvals_true_adds_obs_column(
        self, scoring_adata: AnnData
    ) -> None:
        compute_ulm_scores(
            scoring_adata, genes=self.GENES, score_name="tf", return_pvals=True
        )
        assert "tf_pval" in scoring_adata.obs.columns
        assert np.isfinite(scoring_adata.obs["tf_pval"]).all()

    def test_custom_score_name(self, scoring_adata: AnnData) -> None:
        compute_ulm_scores(
            scoring_adata, genes=self.GENES, score_name="activity"
        )
        assert "activity" in scoring_adata.obs.columns

    def test_custom_set_name(self, scoring_adata: AnnData) -> None:
        compute_ulm_scores(
            scoring_adata,
            genes=self.GENES,
            set_name="my_program",
            score_name="prog",
        )
        assert "prog" in scoring_adata.obs.columns

    def test_with_layer(self, scoring_adata: AnnData) -> None:
        scoring_adata.layers["counts"] = scoring_adata.X.copy()
        compute_ulm_scores(
            scoring_adata, genes=self.GENES, score_name="tf", layer="counts"
        )
        assert "tf" in scoring_adata.obs.columns

    def test_no_residual_ulm_keys_in_obsm(self, scoring_adata: AnnData) -> None:
        compute_ulm_scores(scoring_adata, genes=self.GENES, score_name="tf")
        for key in ("ulm_estimate", "ulm_pvals", "score_ulm", "pval_ulm"):
            assert key not in scoring_adata.obsm

    def test_gene_symbols(self, scoring_adata: AnnData) -> None:
        scoring_adata.var["symbol"] = [f"GENE_{i}" for i in range(N_GENES)]
        symbol_genes = [f"GENE_{i}" for i in range(8)]
        compute_ulm_scores(
            scoring_adata,
            genes=symbol_genes,
            gene_symbols="symbol",
            score_name="ulm_sym",
        )
        assert "ulm_sym" in scoring_adata.obs.columns

    def test_missing_dep_raises(
        self, scoring_adata: AnnData, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scutils.tools.gene_scoring as gs

        monkeypatch.setattr(gs, "_DECOUPLER_AVAILABLE", False)
        with pytest.raises(ImportError, match="decoupler"):
            compute_ulm_scores(scoring_adata, genes=self.GENES)


# ---------------------------------------------------------------------------
# compute_aucell_scores
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PYUCELL_AVAILABLE,
    reason="pyUCell not installed",
)
class TestComputeAucellScores:
    GENES = GENE_NAMES[:8]

    def test_inplace_returns_none(self, scoring_adata: AnnData) -> None:
        result = compute_aucell_scores(scoring_adata, genes=self.GENES)
        assert result is None

    def test_inplace_adds_obs_column(self, scoring_adata: AnnData) -> None:
        compute_aucell_scores(scoring_adata, genes=self.GENES)
        assert "aucell_score" in scoring_adata.obs.columns

    def test_copy_returns_adata(self, scoring_adata: AnnData) -> None:
        result = compute_aucell_scores(
            scoring_adata, genes=self.GENES, copy=True
        )
        assert isinstance(result, AnnData)
        assert "aucell_score" not in scoring_adata.obs.columns
        assert "aucell_score" in result.obs.columns

    def test_custom_score_name(self, scoring_adata: AnnData) -> None:
        compute_aucell_scores(
            scoring_adata, genes=self.GENES, score_name="my_score"
        )
        assert "my_score" in scoring_adata.obs.columns

    def test_custom_set_name(self, scoring_adata: AnnData) -> None:
        compute_aucell_scores(
            scoring_adata,
            genes=self.GENES,
            set_name="my_program",
            score_name="prog",
        )
        assert "prog" in scoring_adata.obs.columns

    def test_scores_are_finite_floats(self, scoring_adata: AnnData) -> None:
        compute_aucell_scores(scoring_adata, genes=self.GENES)
        scores = scoring_adata.obs["aucell_score"]
        assert np.isfinite(scores).all()

    def test_score_length_matches_n_obs(self, scoring_adata: AnnData) -> None:
        compute_aucell_scores(scoring_adata, genes=self.GENES)
        assert len(scoring_adata.obs["aucell_score"]) == scoring_adata.n_obs

    def test_scores_in_zero_one_range(self, scoring_adata: AnnData) -> None:
        compute_aucell_scores(scoring_adata, genes=self.GENES)
        scores = scoring_adata.obs["aucell_score"]
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_nonzero_scores(self, scoring_adata: AnnData) -> None:
        """UCell should yield at least some non-zero scores."""
        compute_aucell_scores(scoring_adata, genes=self.GENES)
        scores = scoring_adata.obs["aucell_score"]
        assert (scores > 0).any(), "UCell should produce non-zero scores"

    def test_no_residual_keys_in_obsm(
        self, scoring_adata: AnnData
    ) -> None:
        compute_aucell_scores(scoring_adata, genes=self.GENES)
        # pyUCell writes only to adata.obs; nothing should land in obsm
        assert all("UCell" not in k for k in scoring_adata.obsm)

    def test_with_layer(self, scoring_adata: AnnData) -> None:
        scoring_adata.layers["counts"] = scoring_adata.X.copy()
        compute_aucell_scores(
            scoring_adata, genes=self.GENES, layer="counts"
        )
        assert "aucell_score" in scoring_adata.obs.columns

    def test_gene_symbols(self, scoring_adata: AnnData) -> None:
        scoring_adata.var["symbol"] = [f"GENE_{i}" for i in range(N_GENES)]
        symbol_genes = [f"GENE_{i}" for i in range(8)]
        compute_aucell_scores(
            scoring_adata,
            genes=symbol_genes,
            gene_symbols="symbol",
            score_name="aucell_sym",
        )
        assert "aucell_sym" in scoring_adata.obs.columns

    def test_missing_dep_raises(
        self,
        scoring_adata: AnnData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scutils.tools.gene_scoring as gs

        monkeypatch.setattr(gs, "_PYUCELL_AVAILABLE", False)
        with pytest.raises(ImportError, match="pyUCell"):
            compute_aucell_scores(scoring_adata, genes=self.GENES)
