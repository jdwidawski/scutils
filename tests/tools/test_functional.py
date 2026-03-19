"""Tests for scutils.tools.functional."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scutils.tools.functional import get_enriched_terms


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_enrich_df() -> pd.DataFrame:
    """Minimal DataFrame that mimics the output of ``sc.queries.enrich``."""
    return pd.DataFrame(
        {
            "source": ["GO:BP", "GO:MF", "REAC", "KEGG", "GO:BP"],
            "name": ["term A", "term B", "term C", "term D", "term E"],
            "p_value": [0.001, 0.01, 0.05, 0.10, 0.001],
            "intersection_size": [5, 3, 10, 2, 4],
            "term_size": [100, 50, 500, 200, 80],
            "parents": [["GO:0001"], ["GO:0002"], [], ["GO:0003"], ["GO:0004"]],
        }
    )


GENE_LIST = ["GENE1", "GENE2", "GENE3"]


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestGetEnrichedTermsBasic:
    def test_returns_dataframe(self):
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            result = get_enriched_terms(GENE_LIST)
        assert isinstance(result, pd.DataFrame)

    def test_adds_intersection_pct_column(self):
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            result = get_enriched_terms(GENE_LIST)
        assert "intersection_pct" in result.columns

    def test_adds_log10_pvalue_column(self):
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            result = get_enriched_terms(GENE_LIST)
        assert "-log10(adj_p_value)" in result.columns

    def test_log10_pvalue_values(self):
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            result = get_enriched_terms(GENE_LIST)
        expected = -np.log10(result["p_value"])
        np.testing.assert_allclose(result["-log10(adj_p_value)"], expected)

    def test_intersection_pct_calculation(self):
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            result = get_enriched_terms(GENE_LIST)
        # First row: 5/100 * 100 = 5.0
        assert result["intersection_pct"].iloc[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Source filtering
# ---------------------------------------------------------------------------


class TestGetEnrichedTermsSourceFilter:
    def test_filters_to_requested_sources(self):
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            result = get_enriched_terms(GENE_LIST, sources=["GO:BP"])
        assert set(result["source"].unique()) == {"GO:BP"}

    def test_empty_sources_returns_all(self):
        df = _make_enrich_df()
        with patch("scanpy.queries.enrich", return_value=df):
            result = get_enriched_terms(GENE_LIST, sources=[])
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Term-size filtering
# ---------------------------------------------------------------------------


class TestGetEnrichedTermsTermSizeFilter:
    def test_min_term_size_filter(self):
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            result = get_enriched_terms(GENE_LIST, sources=[], min_term_size=100)
        assert (result["term_size"] >= 100).all()

    def test_max_term_size_filter(self):
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            result = get_enriched_terms(GENE_LIST, sources=[], max_term_size=200)
        assert (result["term_size"] <= 200).all()

    def test_combined_term_size_filter(self):
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            result = get_enriched_terms(
                GENE_LIST, sources=[], min_term_size=80, max_term_size=200
            )
        assert ((result["term_size"] >= 80) & (result["term_size"] <= 200)).all()


# ---------------------------------------------------------------------------
# Output file handling
# ---------------------------------------------------------------------------


class TestGetEnrichedTermsOutputFile:
    def test_csv_output(self, tmp_path: Path):
        out = tmp_path / "results.csv"
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            get_enriched_terms(GENE_LIST, sources=[], output_file=str(out))
        assert out.exists()
        saved = pd.read_csv(out)
        assert "intersection_pct" in saved.columns

    def test_tsv_output(self, tmp_path: Path):
        out = tmp_path / "results.tsv"
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            get_enriched_terms(GENE_LIST, sources=[], output_file=str(out))
        assert out.exists()
        saved = pd.read_csv(out, sep="\t")
        assert "source" in saved.columns

    def test_xlsx_output(self, tmp_path: Path):
        pytest.importorskip("openpyxl")
        out = tmp_path / "results.xlsx"
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            get_enriched_terms(GENE_LIST, sources=[], output_file=str(out))
        assert out.exists()

    def test_invalid_extension_raises(self, tmp_path: Path):
        out = tmp_path / "results.json"
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            with pytest.raises(ValueError, match="Unsupported output file extension"):
                get_enriched_terms(GENE_LIST, sources=[], output_file=str(out))

    def test_nonexistent_directory_raises(self, tmp_path: Path):
        out = tmp_path / "nonexistent_dir" / "results.csv"
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            with pytest.raises(ValueError, match="does not exist"):
                get_enriched_terms(GENE_LIST, sources=[], output_file=str(out))

    def test_parents_serialised_to_string(self, tmp_path: Path):
        out = tmp_path / "results.csv"
        with patch("scanpy.queries.enrich", return_value=_make_enrich_df()):
            get_enriched_terms(GENE_LIST, sources=[], output_file=str(out))
        saved = pd.read_csv(out)
        # parents should be a pipe-joined string, not a Python list repr
        assert isinstance(saved["parents"].iloc[0], str)
