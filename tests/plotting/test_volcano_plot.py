"""Tests for scutils.plotting.volcano_plot."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from scutils.plotting.volcano_plot import volcano_plot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def de_df() -> pd.DataFrame:
    """Synthetic differential expression DataFrame (50 genes)."""
    rng = np.random.default_rng(42)
    n = 50
    genes = [f"GENE{i}" for i in range(n)]
    lfc = rng.uniform(-4.0, 4.0, size=n)
    pvals = rng.uniform(0.0, 0.1, size=n)
    return pd.DataFrame(
        {"names": genes, "logfoldchanges": lfc, "pvals_adj": pvals}
    )


# ---------------------------------------------------------------------------
# Basic return-type tests
# ---------------------------------------------------------------------------


def test_volcano_plot_returns_figure(de_df: pd.DataFrame) -> None:
    fig = volcano_plot(de_df)
    assert isinstance(fig, Figure)


def test_volcano_plot_custom_figsize(de_df: pd.DataFrame) -> None:
    fig = volcano_plot(de_df, figsize=(10.0, 8.0))
    assert isinstance(fig, Figure)
    w, h = fig.get_size_inches()
    assert abs(w - 10.0) < 1e-6
    assert abs(h - 8.0) < 1e-6


def test_volcano_plot_no_annotations(de_df: pd.DataFrame) -> None:
    fig = volcano_plot(de_df, top_n_up=0, top_n_down=0)
    assert isinstance(fig, Figure)


def test_volcano_plot_extra_genes(de_df: pd.DataFrame) -> None:
    fig = volcano_plot(de_df, extra_genes=["GENE0", "GENE1"])
    assert isinstance(fig, Figure)


def test_volcano_plot_custom_cutoffs(de_df: pd.DataFrame) -> None:
    fig = volcano_plot(de_df, pval_cutoff=0.05, lfc_cutoff=0.5)
    assert isinstance(fig, Figure)


def test_volcano_plot_sort_by_lfc(de_df: pd.DataFrame) -> None:
    fig = volcano_plot(de_df, annot_sort_by="lfc")
    assert isinstance(fig, Figure)


def test_volcano_plot_vertical_mode(de_df: pd.DataFrame) -> None:
    fig = volcano_plot(de_df, annotation_mode="vertical")
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_volcano_plot_invalid_sort_by(de_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        volcano_plot(de_df, annot_sort_by="invalid")


def test_volcano_plot_all_non_significant(de_df: pd.DataFrame) -> None:
    """Should still return a figure even when nothing is significant."""
    fig = volcano_plot(de_df, pval_cutoff=1e-100, lfc_cutoff=100.0)
    assert isinstance(fig, Figure)
