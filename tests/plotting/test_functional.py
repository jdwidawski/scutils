"""Tests for scutils.plotting.functional."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scutils.plotting.functional import (
    _create_size_legend_handles,
    _create_source_legend_handles,
    create_pathway_dotplot,
)

matplotlib.use("Agg")  # non-interactive backend for tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SOURCE_COLORS = {
    "GO:BP": "#1f77b4",
    "GO:MF": "#ff7f0e",
    "REAC": "#2ca02c",
}


@pytest.fixture
def pathway_df() -> pd.DataFrame:
    """Small synthetic pathway enrichment result DataFrame."""
    rng = np.random.default_rng(0)
    n = 15
    sources = rng.choice(list(SOURCE_COLORS), size=n)
    p_values = rng.uniform(1e-10, 0.05, size=n)
    return pd.DataFrame(
        {
            "source": sources,
            "name": [f"Pathway {i}" for i in range(n)],
            "p_value": p_values,
        }
    )


# ---------------------------------------------------------------------------
# create_pathway_dotplot — type / value guards
# ---------------------------------------------------------------------------


class TestCreatePathwayDotplotValidation:
    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            create_pathway_dotplot({"source": [], "name": [], "p_value": []}, SOURCE_COLORS)

    def test_non_dict_source_colors_raises_type_error(self, pathway_df):
        with pytest.raises(TypeError, match="dict"):
            create_pathway_dotplot(pathway_df, source_colors=["red", "blue"])

    def test_non_bool_variable_size_raises_type_error(self, pathway_df):
        with pytest.raises(TypeError, match="bool"):
            create_pathway_dotplot(pathway_df, SOURCE_COLORS, variable_size="yes")

    def test_missing_required_columns_raises_value_error(self):
        df = pd.DataFrame({"source": ["GO:BP"], "name": ["term"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            create_pathway_dotplot(df, SOURCE_COLORS)

    def test_empty_dataframe_raises_value_error(self):
        df = pd.DataFrame(columns=["source", "name", "p_value"])
        with pytest.raises(ValueError, match="empty"):
            create_pathway_dotplot(df, SOURCE_COLORS)


# ---------------------------------------------------------------------------
# create_pathway_dotplot — return type and basic properties
# ---------------------------------------------------------------------------


class TestCreatePathwayDotplotOutput:
    def test_returns_matplotlib_figure(self, pathway_df):
        fig = create_pathway_dotplot(pathway_df, SOURCE_COLORS)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_variable_size_false_returns_figure(self, pathway_df):
        fig = create_pathway_dotplot(pathway_df, SOURCE_COLORS, variable_size=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_max_pathways_limits_rows(self, pathway_df):
        fig = create_pathway_dotplot(pathway_df, SOURCE_COLORS, max_pathways=5)
        ax = fig.axes[0]
        # y-ticks correspond to displayed pathways
        assert len(ax.get_yticks()) == 5
        plt.close(fig)

    def test_max_pathways_none_shows_all(self, pathway_df):
        fig = create_pathway_dotplot(pathway_df, SOURCE_COLORS, max_pathways=None)
        ax = fig.axes[0]
        assert len(ax.get_yticks()) == len(pathway_df)
        plt.close(fig)

    def test_title_set_on_axes(self, pathway_df):
        fig = create_pathway_dotplot(pathway_df, SOURCE_COLORS, title="My Plot")
        assert fig.axes[0].get_title() == "My Plot"
        plt.close(fig)

    def test_missing_source_color_gets_fallback(self, pathway_df):
        """Sources not in source_colors dict receive default fallback colours."""
        colors = {"GO:BP": "#1f77b4"}  # missing GO:MF and REAC
        fig = create_pathway_dotplot(pathway_df, colors)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path_writes_file(self, pathway_df, tmp_path):
        out = tmp_path / "test_plot.png"
        fig = create_pathway_dotplot(pathway_df, SOURCE_COLORS, save_path=str(out))
        assert out.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestCreateSourceLegendHandles:
    def test_returns_list(self):
        handles = _create_source_legend_handles(SOURCE_COLORS)
        assert isinstance(handles, list)

    def test_length_matches_sources(self):
        handles = _create_source_legend_handles(SOURCE_COLORS)
        assert len(handles) == len(SOURCE_COLORS)

    def test_labels_match_keys(self):
        handles = _create_source_legend_handles(SOURCE_COLORS)
        labels = [h.get_label() for h in handles]
        assert labels == list(SOURCE_COLORS.keys())

    def test_empty_dict_returns_empty_list(self):
        assert _create_source_legend_handles({}) == []


class TestCreateSizeLegendHandles:
    def test_returns_three_handles_when_valid(self):
        handles = _create_size_legend_handles(1e-10, 0.05, 20, 200)
        assert len(handles) == 3

    def test_returns_empty_when_min_equals_max(self):
        handles = _create_size_legend_handles(0.01, 0.01, 20, 200)
        assert handles == []
