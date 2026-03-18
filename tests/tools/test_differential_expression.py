"""Tests for scutils.tools.differential_expression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scutils.preprocessing.pseudobulk import pseudobulk
from scutils.tools.differential_expression import deseq2, format_deseq2_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pb_adata(adata_pseudobulk_ready):
    """Pseudobulked AnnData with donor × condition (12 obs × 30 genes)."""
    return pseudobulk(
        adata_pseudobulk_ready,
        sample_col="donor",
        groups_col="condition",
        min_cells=1,
    )


# ---------------------------------------------------------------------------
# deseq2 — basic functionality
# ---------------------------------------------------------------------------


class TestDeseq2Basic:
    def test_returns_dataframe(self, pb_adata):
        results = deseq2(
            pb_adata,
            design="~condition",
            contrast=["condition", "treated", "control"],
            shrink_lfc=False,
        )
        assert isinstance(results, pd.DataFrame)

    def test_expected_columns(self, pb_adata):
        results = deseq2(
            pb_adata,
            design="~condition",
            contrast=["condition", "treated", "control"],
            shrink_lfc=False,
        )
        expected = {"baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"}
        assert expected.issubset(set(results.columns))

    def test_index_is_gene_names(self, pb_adata):
        results = deseq2(
            pb_adata,
            design="~condition",
            contrast=["condition", "treated", "control"],
            shrink_lfc=False,
        )
        assert list(results.index) == list(pb_adata.var_names)

    def test_lfc_shrinkage_runs(self, pb_adata):
        """shrink_lfc=True should not raise."""
        results = deseq2(
            pb_adata,
            design="~condition",
            contrast=["condition", "treated", "control"],
            shrink_lfc=True,
        )
        assert "log2FoldChange" in results.columns

    def test_injected_genes_are_significant(self, pb_adata):
        """Genes 0-4 had a fold-change injected; at least some should be DE."""
        results = deseq2(
            pb_adata,
            design="~condition",
            contrast=["condition", "treated", "control"],
            shrink_lfc=False,
        )
        sig = results.loc[results["padj"] < 0.05]
        injected = [f"gene_{i}" for i in range(5)]
        overlap = set(sig.index) & set(injected)
        assert len(overlap) >= 2, (
            f"Expected ≥2 of {injected} to be significant, got {overlap}"
        )


# ---------------------------------------------------------------------------
# deseq2 — validation
# ---------------------------------------------------------------------------


class TestDeseq2Validation:
    def test_bad_contrast_length_raises(self, pb_adata):
        with pytest.raises(ValueError, match="three-element list"):
            deseq2(
                pb_adata,
                design="~condition",
                contrast=["condition", "treated"],
            )

    def test_ref_level_accepted(self, pb_adata):
        """ref_level should be passed through without error."""
        results = deseq2(
            pb_adata,
            design="~condition",
            contrast=["condition", "treated", "control"],
            ref_level=["condition", "control"],
            shrink_lfc=False,
        )
        assert isinstance(results, pd.DataFrame)


# ---------------------------------------------------------------------------
# format_deseq2_results
# ---------------------------------------------------------------------------


class TestFormatDeseq2Results:
    @pytest.fixture
    def de_results(self, pb_adata):
        return deseq2(
            pb_adata,
            design="~condition",
            contrast=["condition", "treated", "control"],
            shrink_lfc=False,
        )

    def test_scanpy_columns_present(self, de_results):
        formatted = format_deseq2_results(de_results)
        assert "names" in formatted.columns
        assert "pvals_adj" in formatted.columns
        assert "logfoldchanges" in formatted.columns

    def test_gene_names_from_index(self, de_results):
        formatted = format_deseq2_results(de_results)
        assert formatted["names"].tolist() == list(de_results.index)

    def test_values_preserved(self, de_results):
        formatted = format_deseq2_results(de_results)
        np.testing.assert_array_equal(
            de_results["padj"].values,
            formatted["pvals_adj"].values,
        )
        np.testing.assert_array_equal(
            de_results["log2FoldChange"].values,
            formatted["logfoldchanges"].values,
        )

    def test_original_unchanged(self, de_results):
        """format_deseq2_results should not mutate the input."""
        cols_before = list(de_results.columns)
        format_deseq2_results(de_results)
        assert list(de_results.columns) == cols_before
