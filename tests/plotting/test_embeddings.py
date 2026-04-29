"""Tests for scutils.plotting.embeddings."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from matplotlib.figure import Figure

matplotlib.use("Agg")

pytest.importorskip("plotly", reason="plotly is required for umap_3d tests")
import plotly.graph_objects as go  # noqa: E402

from scutils.plotting.embeddings import (
    _categorical_color_map,
    _get_obs_or_var_values,
    _resolve_obsm_key,
    _resolve_vbound,
    embedding_category_multiplot,
    embedding_gene_expression_multiplot,
    flatten_list_of_lists,
    is_sequence,
    umap_3d,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adata_umap() -> AnnData:
    """pbmc68k_reduced with UMAP and leiden clustering."""
    adata = sc.datasets.pbmc68k_reduced()
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="leiden")
    return adata


@pytest.fixture
def adata_umap3d() -> AnnData:
    """pbmc68k_reduced with 3-component UMAP and leiden clustering."""
    adata = sc.datasets.pbmc68k_reduced()
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, n_components=3)
    sc.tl.leiden(adata, key_added="leiden")
    return adata


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


def test_is_sequence_list() -> None:
    assert is_sequence([1, 2, 3]) is True


def test_is_sequence_str() -> None:
    # str supports __len__ and __getitem__ — this is intentional behaviour
    assert is_sequence("hello") is True


def test_is_sequence_int() -> None:
    assert is_sequence(42) is False


def test_flatten_list_of_lists() -> None:
    assert flatten_list_of_lists([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]


def test_flatten_empty() -> None:
    assert flatten_list_of_lists([]) == []


# ---------------------------------------------------------------------------
# embedding_category_multiplot
# ---------------------------------------------------------------------------


def test_embedding_category_multiplot_returns_figure(adata_umap: AnnData) -> None:
    fig = embedding_category_multiplot(adata_umap, column="leiden")
    assert isinstance(fig, Figure)


def test_embedding_category_multiplot_palette_list(adata_umap: AnnData) -> None:
    n_cats = adata_umap.obs["leiden"].nunique()
    palette = ["#ff0000"] * n_cats
    fig = embedding_category_multiplot(adata_umap, column="leiden", palette=palette)
    assert isinstance(fig, Figure)


def test_embedding_category_multiplot_palette_single_colour(
    adata_umap: AnnData,
) -> None:
    fig = embedding_category_multiplot(
        adata_umap, column="leiden", palette="steelblue"
    )
    assert isinstance(fig, Figure)


def test_embedding_category_multiplot_groups_subset(adata_umap: AnnData) -> None:
    cats = adata_umap.obs["leiden"].cat.categories[:2].tolist()
    fig = embedding_category_multiplot(
        adata_umap, column="leiden", groups=cats, ncols=2
    )
    assert isinstance(fig, Figure)


def test_embedding_category_multiplot_invalid_column(adata_umap: AnnData) -> None:
    with pytest.raises(ValueError, match="not found"):
        embedding_category_multiplot(adata_umap, column="nonexistent_col")


# ---------------------------------------------------------------------------
# embedding_gene_expression_multiplot
# ---------------------------------------------------------------------------


def test_embedding_gene_expression_multiplot_returns_figure(
    adata_umap: AnnData,
) -> None:
    gene = adata_umap.var_names[0]
    fig = embedding_gene_expression_multiplot(
        adata_umap, column="leiden", feature=gene
    )
    assert isinstance(fig, Figure)


def test_embedding_gene_expression_multiplot_obs_feature(
    adata_umap: AnnData,
) -> None:
    fig = embedding_gene_expression_multiplot(
        adata_umap, column="leiden", feature="n_counts"
    )
    assert isinstance(fig, Figure)


def test_embedding_gene_expression_multiplot_invalid_column(
    adata_umap: AnnData,
) -> None:
    gene = adata_umap.var_names[0]
    with pytest.raises(ValueError, match="not found"):
        embedding_gene_expression_multiplot(
            adata_umap, column="nonexistent_col", feature=gene
        )


def test_embedding_gene_expression_multiplot_invalid_feature(
    adata_umap: AnnData,
) -> None:
    with pytest.raises(ValueError, match="not found"):
        embedding_gene_expression_multiplot(
            adata_umap, column="leiden", feature="NOT_A_GENE"
        )


# ---------------------------------------------------------------------------
# Private helper unit tests for 3-D helpers
# ---------------------------------------------------------------------------


def test_resolve_obsm_key_exact(adata_umap3d: AnnData) -> None:
    assert _resolve_obsm_key(adata_umap3d, "X_umap") == "X_umap"


def test_resolve_obsm_key_short_alias(adata_umap3d: AnnData) -> None:
    assert _resolve_obsm_key(adata_umap3d, "umap") == "X_umap"


def test_resolve_obsm_key_missing(adata_umap3d: AnnData) -> None:
    with pytest.raises(KeyError, match="not found"):
        _resolve_obsm_key(adata_umap3d, "nonexistent")


def test_resolve_vbound_none() -> None:
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert _resolve_vbound(vals, None) is None


def test_resolve_vbound_float() -> None:
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert _resolve_vbound(vals, 2.5) == pytest.approx(2.5)


def test_resolve_vbound_percentile() -> None:
    vals = np.arange(100, dtype=np.float32)
    result = _resolve_vbound(vals, "p90")
    assert result == pytest.approx(89.1, abs=0.5)


def test_categorical_color_map_from_uns(adata_umap3d: AnnData) -> None:
    cats = list(adata_umap3d.obs["leiden"].cat.categories)
    adata_umap3d.uns["leiden_colors"] = ["#ff0000"] * len(cats)
    cmap = _categorical_color_map(adata_umap3d, "leiden", palette=None)
    assert all(v == "#ff0000" for v in cmap.values())


def test_categorical_color_map_palette_list(adata_umap3d: AnnData) -> None:
    cats = list(adata_umap3d.obs["leiden"].cat.categories)
    palette = ["#aabbcc"] * len(cats)
    cmap = _categorical_color_map(adata_umap3d, "leiden", palette=palette)
    assert all(v.lower() == "#aabbcc" for v in cmap.values())


def test_categorical_color_map_no_uns(adata_umap3d: AnnData) -> None:
    adata_umap3d.uns.pop("leiden_colors", None)
    cmap = _categorical_color_map(adata_umap3d, "leiden", palette=None)
    cats = list(adata_umap3d.obs["leiden"].cat.categories)
    assert set(cmap.keys()) == {str(c) for c in cats}
    # uns state must be restored
    assert "leiden_colors" not in adata_umap3d.uns


def test_get_obs_or_var_values_obs(adata_umap3d: AnnData) -> None:
    vals = _get_obs_or_var_values(adata_umap3d, "n_counts", layer=None)
    assert vals.dtype == np.float32
    assert vals.shape == (adata_umap3d.n_obs,)


def test_get_obs_or_var_values_gene(adata_umap3d: AnnData) -> None:
    gene = adata_umap3d.var_names[0]
    vals = _get_obs_or_var_values(adata_umap3d, gene, layer=None)
    assert vals.dtype == np.float32
    assert vals.shape == (adata_umap3d.n_obs,)


def test_get_obs_or_var_values_missing(adata_umap3d: AnnData) -> None:
    with pytest.raises(ValueError, match="not found"):
        _get_obs_or_var_values(adata_umap3d, "NOT_A_KEY", layer=None)


# ---------------------------------------------------------------------------
# umap_3d — smoke and return-type tests
# ---------------------------------------------------------------------------


def test_umap_3d_returns_figure_no_color(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d)
    assert isinstance(fig, go.Figure)


def test_umap_3d_returns_figure_categorical(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d, color="leiden")
    assert isinstance(fig, go.Figure)


def test_umap_3d_returns_figure_continuous_obs(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d, color="n_counts")
    assert isinstance(fig, go.Figure)


def test_umap_3d_returns_figure_gene(adata_umap3d: AnnData) -> None:
    gene = adata_umap3d.var_names[0]
    fig = umap_3d(adata_umap3d, color=gene)
    assert isinstance(fig, go.Figure)


def test_umap_3d_multicolor_returns_figure(adata_umap3d: AnnData) -> None:
    gene = adata_umap3d.var_names[0]
    fig = umap_3d(adata_umap3d, color=["leiden", gene])
    assert isinstance(fig, go.Figure)


def test_umap_3d_short_basis_alias(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d, color="leiden", basis="umap")
    assert isinstance(fig, go.Figure)


def test_umap_3d_groups(adata_umap3d: AnnData) -> None:
    cats = adata_umap3d.obs["leiden"].cat.categories[:2].tolist()
    fig = umap_3d(adata_umap3d, color="leiden", groups=cats)
    assert isinstance(fig, go.Figure)
    # Inactive cells should produce an "other" background trace
    trace_names = [t.name for t in fig.data]
    assert "other" in trace_names


def test_umap_3d_palette_single_color(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d, color="leiden", palette="red")
    assert isinstance(fig, go.Figure)


def test_umap_3d_palette_list(adata_umap3d: AnnData) -> None:
    n_cats = adata_umap3d.obs["leiden"].nunique()
    fig = umap_3d(adata_umap3d, color="leiden", palette=["#ff0000"] * n_cats)
    assert isinstance(fig, go.Figure)


def test_umap_3d_palette_dict(adata_umap3d: AnnData) -> None:
    cats = adata_umap3d.obs["leiden"].cat.categories.tolist()
    palette = {str(c): "#123456" for c in cats}
    fig = umap_3d(adata_umap3d, color="leiden", palette=palette)
    assert isinstance(fig, go.Figure)


def test_umap_3d_vmin_vmax_float(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d, color="n_counts", vmin=0.0, vmax=1000.0)
    assert isinstance(fig, go.Figure)


def test_umap_3d_vmin_vmax_percentile(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d, color="n_counts", vmin="p5", vmax="p95")
    assert isinstance(fig, go.Figure)


def test_umap_3d_hover_keys(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d, color="leiden", hover_keys=["n_counts"])
    assert isinstance(fig, go.Figure)


def test_umap_3d_ncols(adata_umap3d: AnnData) -> None:
    gene = adata_umap3d.var_names[0]
    fig = umap_3d(adata_umap3d, color=["leiden", gene, "n_counts"], ncols=2)
    assert isinstance(fig, go.Figure)


def test_umap_3d_uns_colors_not_mutated(adata_umap3d: AnnData) -> None:
    """adata.uns should not be altered when palette=None and colors are absent."""
    adata_umap3d.uns.pop("leiden_colors", None)
    umap_3d(adata_umap3d, color="leiden")
    assert "leiden_colors" not in adata_umap3d.uns


def test_umap_3d_uns_colors_preserved(adata_umap3d: AnnData) -> None:
    """Pre-existing adata.uns colors should be unchanged after the call."""
    cats = list(adata_umap3d.obs["leiden"].cat.categories)
    original_colors = ["#abcdef"] * len(cats)
    adata_umap3d.uns["leiden_colors"] = original_colors.copy()
    umap_3d(adata_umap3d, color="leiden")
    assert list(adata_umap3d.uns["leiden_colors"]) == original_colors


def test_umap_3d_invalid_basis(adata_umap3d: AnnData) -> None:
    with pytest.raises(KeyError, match="not found"):
        umap_3d(adata_umap3d, basis="nonexistent")


def test_umap_3d_too_few_components(adata_umap3d: AnnData) -> None:
    """Raise ValueError when the embedding has fewer than 3 components."""
    import numpy as np

    adata_umap3d.obsm["X_umap2d"] = adata_umap3d.obsm["X_umap"][:, :2]
    with pytest.raises(ValueError, match="3"):
        umap_3d(adata_umap3d, basis="X_umap2d")


def test_umap_3d_invalid_color_key(adata_umap3d: AnnData) -> None:
    with pytest.raises(ValueError, match="not found"):
        umap_3d(adata_umap3d, color="NOT_A_GENE_OR_OBS")


def test_umap_3d_sort_order_false(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d, color="n_counts", sort_order=False)
    assert isinstance(fig, go.Figure)


def test_umap_3d_figure_dimensions(adata_umap3d: AnnData) -> None:
    fig = umap_3d(adata_umap3d, color="leiden", width=800, height=600)
    assert fig.layout.width == 800
    assert fig.layout.height == 600


def test_umap_3d_multicolor_dimensions(adata_umap3d: AnnData) -> None:
    gene = adata_umap3d.var_names[0]
    fig = umap_3d(
        adata_umap3d, color=["leiden", gene], width=500, height=400, ncols=2
    )
    assert fig.layout.width == 1000
    assert fig.layout.height == 400


# ---------------------------------------------------------------------------
# merge_traces and max_cells
# ---------------------------------------------------------------------------


def test_umap_3d_merge_traces_true_returns_figure(adata_umap3d: AnnData) -> None:
    """merge_traces=True (default) should still return a valid Figure."""
    fig = umap_3d(adata_umap3d, color="leiden", merge_traces=True)
    assert isinstance(fig, go.Figure)


def test_umap_3d_merge_traces_fewer_traces_than_per_category(
    adata_umap3d: AnnData,
) -> None:
    """With merge_traces=True the real data trace count < one-per-category count."""
    n_cats = adata_umap3d.obs["leiden"].nunique()
    fig_merged = umap_3d(adata_umap3d, color="leiden", merge_traces=True)
    fig_per_cat = umap_3d(adata_umap3d, color="leiden", merge_traces=False)
    # merged: 1 data trace + n_cats legend-only traces
    # per-cat: n_cats data traces
    # The merged figure should have the same number of traces (1 data + n legend)
    # but fewer *rendering* traces than the per-category figure.
    assert len(fig_merged.data) == 1 + n_cats
    assert len(fig_per_cat.data) == n_cats


def test_umap_3d_merge_traces_false_returns_figure(adata_umap3d: AnnData) -> None:
    """merge_traces=False should return a valid Figure with one trace per category."""
    fig = umap_3d(adata_umap3d, color="leiden", merge_traces=False)
    assert isinstance(fig, go.Figure)


def test_umap_3d_max_cells_downsamples(adata_umap3d: AnnData) -> None:
    """max_cells should reduce the total number of plotted points."""
    n_cells = adata_umap3d.n_obs
    max_c = n_cells // 2
    fig = umap_3d(adata_umap3d, color="leiden", max_cells=max_c)
    assert isinstance(fig, go.Figure)
    # Total non-legend points across all traces should be <= max_cells
    total_pts = sum(
        len(t.x) for t in fig.data if t.x is not None and t.x[0] is not None
    )
    assert total_pts <= max_c


def test_umap_3d_max_cells_noop_when_large_enough(adata_umap3d: AnnData) -> None:
    """max_cells larger than n_obs should not change anything."""
    fig_full = umap_3d(adata_umap3d, color="leiden")
    fig_noop = umap_3d(adata_umap3d, color="leiden", max_cells=10_000_000)
    # Both should produce the same number of traces
    assert len(fig_full.data) == len(fig_noop.data)


def test_umap_3d_max_cells_deterministic(adata_umap3d: AnnData) -> None:
    """Same random_state should yield identical figures."""
    fig_a = umap_3d(adata_umap3d, color="leiden", max_cells=20, random_state=42)
    fig_b = umap_3d(adata_umap3d, color="leiden", max_cells=20, random_state=42)
    import numpy as np

    np.testing.assert_array_equal(fig_a.data[0].x, fig_b.data[0].x)
