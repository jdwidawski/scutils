"""Tests for scutils.tools.score_hubs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scutils.tools.score_hubs import (
    _parse_threshold,
    find_high_score_subclusters,
    find_score_hubs,
    plot_subcluster_score_diagnostics,
    plot_score_hub_diagnostics,
)


# Stable cell type used as the hub target across all tests.
_HUB_CT = "Dendritic"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adata_with_scores() -> AnnData:
    """AnnData with neighbors, UMAP, spatially-coherent cell types, and a bimodal score.

    Uses ``pbmc68k_reduced``'s ``bulk_labels`` annotation so that cell types
    correspond to genuine spatial clusters in the UMAP.  A strong score boost
    (+10) is injected into 1/3 of ``'Dendritic'`` cells — the largest group —
    so that both hub-detection functions reliably flag it.
    """
    rng = np.random.default_rng(0)

    adata = sc.datasets.pbmc68k_reduced()
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    # bulk_labels form real spatial clusters in the UMAP.
    adata.obs["cell_type"] = adata.obs["bulk_labels"].copy()

    n_cells = adata.n_obs
    scores = rng.normal(0.0, 1.0, size=n_cells)

    # Boost ~1/3 of Dendritic cells (n=240) by +10.
    dc_idx = np.where(adata.obs["cell_type"] == _HUB_CT)[0]
    hub_idx = rng.choice(dc_idx, size=len(dc_idx) // 3, replace=False)
    scores[hub_idx] += 10.0

    adata.obs["gene_score"] = scores.astype(np.float32)
    return adata


# ---------------------------------------------------------------------------
# _parse_threshold
# ---------------------------------------------------------------------------


class TestParseThreshold:
    def test_float_passthrough(self) -> None:
        scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert _parse_threshold(1.5, scores) == pytest.approx(1.5)

    def test_int_passthrough(self) -> None:
        scores = np.array([0.0, 1.0, 2.0])
        assert _parse_threshold(2, scores) == pytest.approx(2.0)

    def test_percentile_string(self) -> None:
        scores = np.arange(100.0)
        result = _parse_threshold("p75", scores)
        assert result == pytest.approx(np.percentile(scores, 75))

    def test_percentile_decimal(self) -> None:
        scores = np.arange(100.0)
        result = _parse_threshold("p90.5", scores)
        assert result == pytest.approx(np.percentile(scores, 90.5))

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="pXX"):
            _parse_threshold("90pct", np.array([1.0]))


# ---------------------------------------------------------------------------
# find_high_score_subclusters
# ---------------------------------------------------------------------------


class TestFindHighScoreSubclusters:
    def test_returns_dataframe(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        df = find_high_score_subclusters(
            adata, score_key="gene_score", cell_type_col="cell_type"
        )
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_columns(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        df = find_high_score_subclusters(
            adata, score_key="gene_score", cell_type_col="cell_type"
        )
        expected = {
            "cell_type",
            "subcluster_id",
            "n_cells",
            "mean_score",
            "median_score",
            "fraction_above_threshold",
            "is_hub",
        }
        assert expected.issubset(df.columns)

    def test_obs_columns_written(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        find_high_score_subclusters(
            adata, score_key="gene_score", cell_type_col="cell_type"
        )
        assert "gene_score_subcluster" in adata.obs.columns
        assert "gene_score_subcluster_is_hub" in adata.obs.columns

    def test_custom_key_added(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        find_high_score_subclusters(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            key_added="my_sub",
        )
        assert "my_sub" in adata.obs.columns
        assert "my_sub_is_hub" in adata.obs.columns

    def test_hub_fraction_coverage(self, adata_with_scores: AnnData) -> None:
        """n_cells in summary must sum to adata.n_obs."""
        adata = adata_with_scores.copy()
        df = find_high_score_subclusters(
            adata, score_key="gene_score", cell_type_col="cell_type"
        )
        assert df["n_cells"].sum() == adata.n_obs

    def test_hub_ct_flagged_as_hub(self, adata_with_scores: AnnData) -> None:
        """At least one Dendritic sub-cluster must be flagged as a hub."""
        adata = adata_with_scores.copy()
        df = find_high_score_subclusters(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            score_threshold="p85",
        )
        hubs = df[(df["cell_type"] == _HUB_CT) & df["is_hub"]]
        assert len(hubs) >= 1

    def test_absolute_threshold(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        df = find_high_score_subclusters(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            score_threshold=0.0,
        )
        # With threshold=0 and boosted T cells, at least one sub-cluster is a hub.
        assert df["is_hub"].any()

    def test_per_cell_type_resolution_dict(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        ct_labels = adata.obs["cell_type"].cat.categories.tolist()
        # Assign a higher resolution to the hub cell type, lower to others.
        resolution = {ct: (0.5 if ct == _HUB_CT else 0.1) for ct in ct_labels}
        df = find_high_score_subclusters(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            resolution=resolution,
        )
        assert isinstance(df, pd.DataFrame)
        assert "gene_score_subcluster" in adata.obs.columns

    def test_invalid_score_key_raises(self, adata_with_scores: AnnData) -> None:
        with pytest.raises(ValueError, match="score_key"):
            find_high_score_subclusters(
                adata_with_scores.copy(),
                score_key="nonexistent",
                cell_type_col="cell_type",
            )

    def test_invalid_cell_type_col_raises(self, adata_with_scores: AnnData) -> None:
        with pytest.raises(ValueError, match="cell_type_col"):
            find_high_score_subclusters(
                adata_with_scores.copy(),
                score_key="gene_score",
                cell_type_col="nonexistent",
            )

    def test_invalid_method_raises(self, adata_with_scores: AnnData) -> None:
        with pytest.raises(ValueError, match="method"):
            find_high_score_subclusters(
                adata_with_scores.copy(),
                score_key="gene_score",
                cell_type_col="cell_type",
                method="bad_method",  # type: ignore[arg-type]
            )

    def test_n_clusters_mode(self, adata_with_scores: AnnData) -> None:
        pytest.importorskip("sklearn")
        adata = adata_with_scores.copy()
        k = 3
        n_cell_types = adata.obs["cell_type"].nunique()
        df = find_high_score_subclusters(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            n_clusters=k,
            use_rep="X_pca",
        )
        assert len(df) == n_cell_types * k
        # Each cell type must produce exactly k sub-clusters.
        for ct, grp in df.groupby("cell_type"):
            assert len(grp) == k, f"{ct} has {len(grp)} sub-clusters, expected {k}"

    def test_n_clusters_per_cell_type_dict(self, adata_with_scores: AnnData) -> None:
        pytest.importorskip("sklearn")
        adata = adata_with_scores.copy()
        ct_labels = adata.obs["cell_type"].cat.categories.tolist()
        # Give the hub cell type 4 sub-clusters, all others 2.
        n_clusters_dict = {ct: (4 if ct == _HUB_CT else 2) for ct in ct_labels}
        df = find_high_score_subclusters(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            n_clusters=n_clusters_dict,
            use_rep="X_pca",
        )
        for ct, expected_k in n_clusters_dict.items():
            grp = df[df["cell_type"] == ct]
            assert len(grp) == expected_k, (
                f"{ct}: expected {expected_k} sub-clusters, got {len(grp)}"
            )

    def test_recompute_neighbors(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        df = find_high_score_subclusters(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            resolution=0.3,
            recompute_neighbors=True,
            use_rep="X_pca",
        )
        assert isinstance(df, pd.DataFrame)
        assert "gene_score_subcluster" in adata.obs.columns


# ---------------------------------------------------------------------------
# find_score_hubs
# ---------------------------------------------------------------------------


class TestFindScoreHubs:
    def test_returns_dataframe(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        df = find_score_hubs(
            adata, score_key="gene_score", cell_type_col="cell_type"
        )
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_columns(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        df = find_score_hubs(
            adata, score_key="gene_score", cell_type_col="cell_type"
        )
        expected = {"cell_type", "mean_score", "hub_score", "hub_ratio", "hub_fraction", "is_hub"}
        assert expected.issubset(df.columns)

    def test_local_score_written_to_obs(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        find_score_hubs(adata, score_key="gene_score", cell_type_col="cell_type")
        assert "gene_score_local" in adata.obs.columns

    def test_custom_key_added(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        find_score_hubs(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            key_added="my_local",
        )
        assert "my_local" in adata.obs.columns

    def test_local_scores_finite(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        find_score_hubs(adata, score_key="gene_score", cell_type_col="cell_type")
        assert np.isfinite(adata.obs["gene_score_local"].values).all()

    def test_local_score_length(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        find_score_hubs(adata, score_key="gene_score", cell_type_col="cell_type")
        assert len(adata.obs["gene_score_local"]) == adata.n_obs

    def test_sorted_by_hub_ratio(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        df = find_score_hubs(
            adata, score_key="gene_score", cell_type_col="cell_type"
        )
        ratios = df["hub_ratio"].values
        assert (ratios[:-1] >= ratios[1:]).all()

    def test_one_row_per_cell_type(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        df = find_score_hubs(
            adata, score_key="gene_score", cell_type_col="cell_type"
        )
        n_types = adata.obs["cell_type"].nunique()
        assert len(df) == n_types

    def test_hub_ct_flagged_as_hub(self, adata_with_scores: AnnData) -> None:
        """The boosted Dendritic population must be detected as a hub."""
        adata = adata_with_scores.copy()
        df = find_score_hubs(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            hub_percentile=90,
            min_hub_ratio=1.2,
        )
        dc_row = df[df["cell_type"] == _HUB_CT]
        assert len(dc_row) == 1
        assert bool(dc_row["is_hub"].iloc[0])

    def test_invalid_score_key_raises(self, adata_with_scores: AnnData) -> None:
        with pytest.raises(ValueError, match="score_key"):
            find_score_hubs(
                adata_with_scores.copy(),
                score_key="nonexistent",
                cell_type_col="cell_type",
            )

    def test_invalid_cell_type_col_raises(self, adata_with_scores: AnnData) -> None:
        with pytest.raises(ValueError, match="cell_type_col"):
            find_score_hubs(
                adata_with_scores.copy(),
                score_key="gene_score",
                cell_type_col="nonexistent",
            )

    def test_invalid_method_raises(self, adata_with_scores: AnnData) -> None:
        with pytest.raises(ValueError, match="method"):
            find_score_hubs(
                adata_with_scores.copy(),
                score_key="gene_score",
                cell_type_col="cell_type",
                method="bad_method",  # type: ignore[arg-type]
            )

    def test_missing_neighbors_raises(self, adata_with_scores: AnnData) -> None:
        adata = adata_with_scores.copy()
        del adata.obsp["connectivities"]
        with pytest.raises(KeyError, match="connectivities"):
            find_score_hubs(adata, score_key="gene_score", cell_type_col="cell_type")

    def test_gmm_method(self, adata_with_scores: AnnData) -> None:
        pytest.importorskip("sklearn")
        adata = adata_with_scores.copy()
        df = find_score_hubs(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            method="gmm",
        )
        assert isinstance(df, pd.DataFrame)
        assert "hub_fraction" in df.columns
        # hub_fraction must be in [0, 1].
        assert ((df["hub_fraction"] >= 0) & (df["hub_fraction"] <= 1)).all()


# ---------------------------------------------------------------------------
# plot_subcluster_score_diagnostics
# ---------------------------------------------------------------------------


class TestPlotSubclusterScoreDiagnostics:
    @pytest.fixture
    def adata_subclustered(self, adata_with_scores: AnnData) -> AnnData:
        adata = adata_with_scores.copy()
        find_high_score_subclusters(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            score_threshold="p85",
        )
        return adata

    def test_returns_figure(self, adata_subclustered: AnnData) -> None:
        fig = plot_subcluster_score_diagnostics(
            adata_subclustered,
            score_key="gene_score",
            subcluster_key="gene_score_subcluster",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correct_panel_count(self, adata_subclustered: AnnData) -> None:
        fig = plot_subcluster_score_diagnostics(
            adata_subclustered,
            score_key="gene_score",
            subcluster_key="gene_score_subcluster",
        )
        # Each hub row has 2 subplot axes + 1 colorbar (panel 1 only) = 3 axes.
        is_hub_col = "gene_score_subcluster_is_hub"
        hub_cell_types = (
            adata_subclustered.obs.loc[
                adata_subclustered.obs[is_hub_col], "gene_score_subcluster"
            ]
            .astype(str)
            .str.rsplit("_", n=1)
            .str[0]
            .unique()
        )
        n_rows = len(hub_cell_types)
        assert len(fig.axes) == n_rows * 3
        plt.close(fig)

    def test_missing_subcluster_key_raises(self, adata_with_scores: AnnData) -> None:
        with pytest.raises(ValueError, match="not found in adata.obs"):
            plot_subcluster_score_diagnostics(
                adata_with_scores.copy(),
                score_key="gene_score",
                subcluster_key="nonexistent",
            )

    def test_missing_basis_raises(self, adata_subclustered: AnnData) -> None:
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            plot_subcluster_score_diagnostics(
                adata_subclustered,
                score_key="gene_score",
                subcluster_key="gene_score_subcluster",
                basis="X_nonexistent",
            )

    def test_absolute_vmin_vmax(self, adata_subclustered: AnnData) -> None:
        fig = plot_subcluster_score_diagnostics(
            adata_subclustered,
            score_key="gene_score",
            subcluster_key="gene_score_subcluster",
            vmin=-5.0,
            vmax=10.0,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_score_hub_diagnostics
# ---------------------------------------------------------------------------


class TestPlotScoreHubDiagnostics:
    @pytest.fixture
    def adata_with_local_score(self, adata_with_scores: AnnData) -> tuple[AnnData, pd.DataFrame]:
        adata = adata_with_scores.copy()
        hub_df = find_score_hubs(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            hub_percentile=90,
            min_hub_ratio=1.2,
        )
        return adata, hub_df

    def test_returns_figure(
        self, adata_with_local_score: tuple[AnnData, pd.DataFrame]
    ) -> None:
        adata, hub_df = adata_with_local_score
        fig = plot_score_hub_diagnostics(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            hub_df=hub_df,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correct_panel_count(
        self, adata_with_local_score: tuple[AnnData, pd.DataFrame]
    ) -> None:
        adata, hub_df = adata_with_local_score
        n_hubs = int(hub_df["is_hub"].sum())
        fig = plot_score_hub_diagnostics(
            adata,
            score_key="gene_score",
            cell_type_col="cell_type",
            hub_df=hub_df,
        )
        # 2 subplot axes + 2 colorbar axes per hub row.
        assert len(fig.axes) == n_hubs * 4
        plt.close(fig)

    def test_missing_local_score_raises(
        self, adata_with_local_score: tuple[AnnData, pd.DataFrame]
    ) -> None:
        adata, hub_df = adata_with_local_score
        adata2 = adata.copy()
        del adata2.obs["gene_score_local"]
        with pytest.raises(ValueError, match="not found in adata.obs"):
            plot_score_hub_diagnostics(
                adata2,
                score_key="gene_score",
                cell_type_col="cell_type",
                hub_df=hub_df,
            )

    def test_missing_is_hub_column_raises(
        self, adata_with_local_score: tuple[AnnData, pd.DataFrame]
    ) -> None:
        adata, hub_df = adata_with_local_score
        bad_df = hub_df.drop(columns=["is_hub"])
        with pytest.raises(ValueError, match="is_hub"):
            plot_score_hub_diagnostics(
                adata,
                score_key="gene_score",
                cell_type_col="cell_type",
                hub_df=bad_df,
            )

    def test_no_hubs_raises(
        self, adata_with_local_score: tuple[AnnData, pd.DataFrame]
    ) -> None:
        adata, hub_df = adata_with_local_score
        no_hub_df = hub_df.copy()
        no_hub_df["is_hub"] = False
        with pytest.raises(ValueError, match="No hub cell types"):
            plot_score_hub_diagnostics(
                adata,
                score_key="gene_score",
                cell_type_col="cell_type",
                hub_df=no_hub_df,
            )

    def test_custom_local_score_key(
        self, adata_with_local_score: tuple[AnnData, pd.DataFrame]
    ) -> None:
        adata, hub_df = adata_with_local_score
        # Create a copy of the local score under a custom key.
        adata2 = adata.copy()
        adata2.obs["custom_local"] = adata2.obs["gene_score_local"]
        fig = plot_score_hub_diagnostics(
            adata2,
            score_key="gene_score",
            cell_type_col="cell_type",
            hub_df=hub_df,
            local_score_key="custom_local",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
