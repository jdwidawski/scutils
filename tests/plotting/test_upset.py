"""Tests for scutils.plotting.upset."""
from __future__ import annotations

import matplotlib
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from scutils.plotting.upset import upset_plot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gene_sets() -> dict[str, set[str]]:
    """Three partially overlapping gene sets."""
    return {
        "Comp_A": {"TP53", "BRCA1", "MYC", "EGFR", "PTEN"},
        "Comp_B": {"TP53", "MYC", "KRAS", "BRAF"},
        "Comp_C": {"TP53", "BRCA1", "KRAS", "PTEN", "APC"},
    }


@pytest.fixture
def disjoint_sets() -> dict[str, set[str]]:
    """Two sets with no overlap."""
    return {
        "Set_X": {"A", "B", "C"},
        "Set_Y": {"D", "E", "F"},
    }


# ---------------------------------------------------------------------------
# Basic return-type tests
# ---------------------------------------------------------------------------


def test_upset_plot_returns_figure(gene_sets: dict[str, set[str]]) -> None:
    fig = upset_plot(gene_sets)
    assert isinstance(fig, Figure)


def test_upset_plot_custom_figsize(gene_sets: dict[str, set[str]]) -> None:
    fig = upset_plot(gene_sets, figsize=(12.0, 8.0))
    assert isinstance(fig, Figure)
    w, h = fig.get_size_inches()
    assert abs(w - 12.0) < 1e-6
    assert abs(h - 8.0) < 1e-6


def test_upset_plot_show_counts_false(gene_sets: dict[str, set[str]]) -> None:
    fig = upset_plot(gene_sets, show_counts=False)
    assert isinstance(fig, Figure)


def test_upset_plot_sort_by_degree(gene_sets: dict[str, set[str]]) -> None:
    fig = upset_plot(gene_sets, sort_by="degree")
    assert isinstance(fig, Figure)


def test_upset_plot_min_subset_size(gene_sets: dict[str, set[str]]) -> None:
    fig = upset_plot(gene_sets, min_subset_size=2)
    assert isinstance(fig, Figure)


def test_upset_plot_show_percentages(gene_sets: dict[str, set[str]]) -> None:
    fig = upset_plot(gene_sets, show_percentages=True)
    assert isinstance(fig, Figure)


def test_upset_plot_vertical(gene_sets: dict[str, set[str]]) -> None:
    fig = upset_plot(gene_sets, orientation="vertical")
    assert isinstance(fig, Figure)


def test_upset_plot_single_set() -> None:
    with pytest.raises(ValueError, match="at least 2 sets"):
        upset_plot({"Only": {"A", "B", "C"}})


def test_upset_plot_disjoint_sets(disjoint_sets: dict[str, set[str]]) -> None:
    fig = upset_plot(disjoint_sets)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_upset_plot_empty_dict() -> None:
    with pytest.raises(ValueError, match="non-empty dict"):
        upset_plot({})


def test_upset_plot_all_empty_sets() -> None:
    with pytest.raises(ValueError, match="empty"):
        upset_plot({"A": set(), "B": set()})
