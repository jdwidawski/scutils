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


@pytest.fixture
def minimal_net() -> pd.DataFrame:
    """Minimal prior-knowledge network: 2 TFs × 6 target genes each."""
    return pd.DataFrame(
        {
            "source": ["TF1"] * 6 + ["TF2"] * 6,
            "target": GENE_NAMES[:6] + GENE_NAMES[6:12],
            "weight": [1.0] * 12,
        }
    )


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
    def test_inplace_returns_none(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        result = compute_ulm_scores(scoring_adata, net=minimal_net)
        assert result is None

    def test_inplace_adds_obs_columns(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        compute_ulm_scores(scoring_adata, net=minimal_net, score_name="tf")
        assert "tf_TF1" in scoring_adata.obs.columns
        assert "tf_TF2" in scoring_adata.obs.columns

    def test_copy_returns_adata(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        result = compute_ulm_scores(
            scoring_adata, net=minimal_net, score_name="tf", copy=True
        )
        assert isinstance(result, AnnData)
        assert "tf_TF1" not in scoring_adata.obs.columns
        assert "tf_TF1" in result.obs.columns

    def test_scores_are_finite(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        compute_ulm_scores(scoring_adata, net=minimal_net, score_name="tf")
        assert np.isfinite(scoring_adata.obs["tf_TF1"]).all()
        assert np.isfinite(scoring_adata.obs["tf_TF2"]).all()

    def test_score_length_matches_n_obs(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        compute_ulm_scores(scoring_adata, net=minimal_net, score_name="tf")
        assert len(scoring_adata.obs["tf_TF1"]) == scoring_adata.n_obs

    def test_return_pvals_false_no_obsm_key(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        compute_ulm_scores(
            scoring_adata, net=minimal_net, score_name="tf", return_pvals=False
        )
        assert "tf_pvals" not in scoring_adata.obsm

    def test_return_pvals_true_adds_obsm_key(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        compute_ulm_scores(
            scoring_adata, net=minimal_net, score_name="tf", return_pvals=True
        )
        assert "tf_pvals" in scoring_adata.obsm
        assert isinstance(scoring_adata.obsm["tf_pvals"], pd.DataFrame)

    def test_custom_score_name_prefix(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        compute_ulm_scores(scoring_adata, net=minimal_net, score_name="activity")
        assert "activity_TF1" in scoring_adata.obs.columns
        assert "activity_TF2" in scoring_adata.obs.columns

    def test_with_layer(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        scoring_adata.layers["counts"] = scoring_adata.X.copy()
        compute_ulm_scores(
            scoring_adata, net=minimal_net, score_name="tf", layer="counts"
        )
        assert "tf_TF1" in scoring_adata.obs.columns

    def test_no_residual_ulm_keys_in_obsm(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame
    ) -> None:
        compute_ulm_scores(scoring_adata, net=minimal_net, score_name="tf")
        assert "ulm_estimate" not in scoring_adata.obsm
        assert "ulm_pvals" not in scoring_adata.obsm

    def test_missing_dep_raises(
        self, scoring_adata: AnnData, minimal_net: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scutils.tools.gene_scoring as gs

        monkeypatch.setattr(gs, "_DECOUPLER_AVAILABLE", False)
        with pytest.raises(ImportError, match="decoupler"):
            compute_ulm_scores(scoring_adata, net=minimal_net)
