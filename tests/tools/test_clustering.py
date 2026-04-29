"""Tests for scutils.tools.clustering."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from matplotlib.figure import Figure

matplotlib.use("Agg")

from scutils.tools.clustering import (
    iterative_subcluster,
    plot_spatial_split_diagnostics,
    plot_spatial_split_reassign_subclusters_diagnostics,
    rename_subcluster_labels,
    spatial_split_clusters,
    spatial_split_reassign_subclusters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adata_clustered() -> AnnData:
    """pbmc68k_reduced with neighbors graph and leiden clustering."""
    adata = sc.datasets.pbmc68k_reduced()
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="leiden", flavor="igraph", n_iterations=2, directed=False)
    return adata


# ---------------------------------------------------------------------------
# iterative_subcluster
# ---------------------------------------------------------------------------


def test_iterative_subcluster_creates_column(adata_clustered: AnnData) -> None:
    first_cluster = adata_clustered.obs["leiden"].cat.categories[0]
    iterative_subcluster(
        adata_clustered,
        cluster_col="leiden",
        subcluster_resolutions={first_cluster: 0.3},
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    assert "leiden_subclustered" in adata_clustered.obs.columns


def test_iterative_subcluster_clean_labels_are_integers(
    adata_clustered: AnnData,
) -> None:
    first_cluster = adata_clustered.obs["leiden"].cat.categories[0]
    iterative_subcluster(
        adata_clustered,
        cluster_col="leiden",
        subcluster_resolutions={first_cluster: 0.3},
        clean=True,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    cats = adata_clustered.obs["leiden_subclustered"].cat.categories.tolist()
    # All labels should be parseable as integers when clean=True
    for label in cats:
        int(label)


def test_iterative_subcluster_invalid_col(adata_clustered: AnnData) -> None:
    with pytest.raises(ValueError, match="not found"):
        iterative_subcluster(
            adata_clustered,
            cluster_col="nonexistent",
            subcluster_resolutions={},
        )


def test_iterative_subcluster_invalid_method(adata_clustered: AnnData) -> None:
    with pytest.raises(ValueError):
        iterative_subcluster(
            adata_clustered,
            cluster_col="leiden",
            subcluster_resolutions={},
            method="unknown_method",
        )


# ---------------------------------------------------------------------------
# rename_subcluster_labels
# ---------------------------------------------------------------------------


def test_rename_subcluster_labels_basic(adata_clustered: AnnData) -> None:
    cats = adata_clustered.obs["leiden"].cat.categories.tolist()
    new_name = "group_A"
    label_map = {new_name: [cats[0]]}
    rename_subcluster_labels(adata_clustered, col="leiden", label_map=label_map)
    assert new_name in adata_clustered.obs["leiden"].cat.categories


def test_rename_subcluster_labels_invalid_col(adata_clustered: AnnData) -> None:
    with pytest.raises(ValueError, match="not found"):
        rename_subcluster_labels(
            adata_clustered,
            col="nonexistent",
            label_map={"A": ["0"]},
        )


def test_rename_subcluster_labels_invalid_label(adata_clustered: AnnData) -> None:
    with pytest.raises(ValueError):
        rename_subcluster_labels(
            adata_clustered,
            col="leiden",
            label_map={"A": ["NONEXISTENT_LABEL_XYZ"]},
        )


# ---------------------------------------------------------------------------
# spatial_split_clusters
# ---------------------------------------------------------------------------


def test_spatial_split_clusters_creates_column(adata_clustered: AnnData) -> None:
    first_cluster = adata_clustered.obs["leiden"].cat.categories[0]
    spatial_split_clusters(
        adata_clustered,
        cluster_col="leiden",
        categories=[first_cluster],
        eps=2.0,
    )
    assert "leiden_spatial_split" in adata_clustered.obs.columns


def test_spatial_split_clusters_custom_key(adata_clustered: AnnData) -> None:
    first_cluster = adata_clustered.obs["leiden"].cat.categories[0]
    spatial_split_clusters(
        adata_clustered,
        cluster_col="leiden",
        categories=[first_cluster],
        key_added="my_split",
        eps=2.0,
    )
    assert "my_split" in adata_clustered.obs.columns


def test_spatial_split_clusters_hdbscan(adata_clustered: AnnData) -> None:
    first_cluster = adata_clustered.obs["leiden"].cat.categories[0]
    spatial_split_clusters(
        adata_clustered,
        cluster_col="leiden",
        categories=[first_cluster],
        method="hdbscan",
        min_cluster_size=5,
    )
    assert "leiden_spatial_split" in adata_clustered.obs.columns


# ---------------------------------------------------------------------------
# plot_spatial_split_diagnostics
# ---------------------------------------------------------------------------


def test_plot_spatial_split_diagnostics_returns_figure(
    adata_clustered: AnnData,
) -> None:
    first_cluster = adata_clustered.obs["leiden"].cat.categories[0]
    fig = plot_spatial_split_diagnostics(
        adata_clustered,
        cluster_col="leiden",
        categories=[first_cluster],
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# spatial_split_reassign_subclusters
# ---------------------------------------------------------------------------


def test_spatial_split_reassign_subclusters_creates_column(
    adata_clustered: AnnData,
) -> None:
    first_cluster = adata_clustered.obs["leiden"].cat.categories[0]
    spatial_split_clusters(
        adata_clustered,
        cluster_col="leiden",
        categories=[first_cluster],
        clean=False,
        eps=2.0,
    )
    first_split_cat = adata_clustered.obs["leiden_spatial_split"].cat.categories[0]
    spatial_split_reassign_subclusters(
        adata_clustered,
        split_col="leiden_spatial_split",
        subclusters=[first_split_cat],
    )
    assert "leiden_spatial_split_reassigned" in adata_clustered.obs.columns


def test_spatial_split_reassign_subclusters_invalid_split_col(
    adata_clustered: AnnData,
) -> None:
    with pytest.raises(ValueError, match="not found"):
        spatial_split_reassign_subclusters(
            adata_clustered,
            split_col="nonexistent",
            subclusters=["0"],
        )


# ---------------------------------------------------------------------------
# plot_spatial_split_reassign_subclusters_diagnostics
# ---------------------------------------------------------------------------


def test_plot_spatial_split_reassign_subclusters_diagnostics_returns_figure(
    adata_clustered: AnnData,
) -> None:
    first_cluster = adata_clustered.obs["leiden"].cat.categories[0]
    spatial_split_clusters(
        adata_clustered,
        cluster_col="leiden",
        categories=[first_cluster],
        clean=False,
        eps=2.0,
    )
    first_split_cat = adata_clustered.obs["leiden_spatial_split"].cat.categories[0]
    fig = plot_spatial_split_reassign_subclusters_diagnostics(
        adata_clustered,
        split_col="leiden_spatial_split",
        subclusters=[first_split_cat],
        cluster_col="leiden",
    )
    assert isinstance(fig, Figure)
