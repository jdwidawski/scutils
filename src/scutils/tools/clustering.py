from __future__ import annotations
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def iterative_subcluster(
    adata: ad.AnnData,
    cluster_col: str,
    subcluster_resolutions: Dict[Union[str, int], float],
    method: Literal["scanpy", "rapids"] = "scanpy",
    clean: bool = True,
    **kwargs,
) -> None:
    """Iteratively subcluster a clustering column into finer groups.

    For each cluster specified in *subcluster_resolutions*, Leiden community
    detection is re-run on only the cells belonging to that cluster using the
    provided resolution.  Results are accumulated in a new column
    ``{cluster_col}_subclustered`` in ``adata.obs``.  Clusters not listed in
    *subcluster_resolutions* are left unchanged.

    Args:
        adata: Annotated data matrix.
        cluster_col: Key in ``adata.obs`` that holds the original cluster
            assignments.
        subcluster_resolutions: Mapping of cluster label → Leiden resolution
            to use when subclustering that cluster.  Dict iteration order is
            preserved, so hierarchical subclustering is supported by passing
            an already-subclustered label (e.g. ``"1,1"``) as a subsequent key.
        method: Backend to use for Leiden clustering.  ``"scanpy"`` calls
            ``sc.tl.leiden``; ``"rapids"`` calls ``rsc.tl.leiden``.
            Defaults to ``"scanpy"``.
        clean: When ``True``, all resulting category labels are renamed to
            consecutive integer strings (``"0"``, ``"1"``, ``"2"``, …)
            sorted by the original cluster ordering.  When ``False``, the raw
            comma-separated labels produced by scanpy / rapids are kept.
            Defaults to ``True``.
        **kwargs: Additional keyword arguments forwarded verbatim to
            ``sc.tl.leiden`` / ``rsc.tl.leiden`` (e.g. ``n_iterations``,
            ``random_state``, ``neighbors_key``).

    Returns:
        None: Modifies *adata* in place.  The result is stored at
        ``adata.obs[f"{cluster_col}_subclustered"]``.

    Raises:
        ValueError: If *cluster_col* is not found in ``adata.obs``.
        ValueError: If *method* is not ``"scanpy"`` or ``"rapids"``.

    Example:
        >>> iterative_subcluster(
        ...     adata,
        ...     cluster_col="leiden",
        ...     subcluster_resolutions={"1": 0.5, "3": 0.3},
        ... )
        >>> adata.obs["leiden_subclustered"].value_counts()
    """
    if cluster_col not in adata.obs.columns:
        raise ValueError(
            f"cluster_col '{cluster_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    if method == "scanpy":
        import scanpy as sc

        tl = sc.tl
    elif method == "rapids":
        import rapids_singlecell as rsc

        tl = rsc.tl
    else:
        raise ValueError(
            f"method must be 'scanpy' or 'rapids', got '{method!r}'"
        )

    out_col = f"{cluster_col}_subclustered"

    # Initialise the output column as a copy of the original clustering.
    adata.obs[out_col] = adata.obs[cluster_col].copy()

    for cluster, resolution in subcluster_resolutions.items():
        cluster = str(cluster)
        tl.leiden(
            adata,
            restrict_to=(out_col, [cluster]),
            resolution=resolution,
            key_added=out_col,
            **kwargs,
        )

    if clean:
        _clean_subcluster_labels(adata, out_col)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sort_key(cat: str) -> list:
    """Sort key for comma-separated cluster labels.

    Splits the label on commas and compares each part numerically when
    possible.  This guarantees an order like ``"0" < "1,1" < "1,2" <
    "2"``.
    Args:
        cat: A cluster label string, possibly comma-separated.

    Returns:
        A list of ``(type_flag, value)`` tuples suitable for
        :func:`sorted`.    """
    parts = str(cat).split(",")
    key = []
    for part in parts:
        part = part.strip()
        try:
            key.append((0, int(part)))
        except ValueError:
            key.append((1, part))
    return key


def _clean_subcluster_labels(adata: ad.AnnData, col: str) -> None:
    """Rename subclustered categories to consecutive integer strings.

    Categories are sorted by their comma-separated components so that the
    original cluster ordering is preserved.  Numeric parts are compared
    numerically; non-numeric parts fall back to lexicographic comparison.

    Args:
        adata: Annotated data matrix to modify in place.
        col: Column in ``adata.obs`` whose categories should be relabelled.
    """
    current_cats: list = adata.obs[col].cat.categories.tolist()
    sorted_cats = sorted(current_cats, key=_sort_key)
    rename_map: dict[str, str] = {old: str(i) for i, old in enumerate(sorted_cats)}

    new_labels = pd.Categorical(
        adata.obs[col].map(rename_map),
        categories=[str(i) for i in range(len(rename_map))],
        ordered=False,
    )
    adata.obs[col] = new_labels


def rename_subcluster_labels(
    adata: ad.AnnData,
    col: str,
    label_map: Dict[str, List[str]],
) -> None:
    """Rename subclustered category labels based on a user-supplied mapping.

    Groups one or more existing category labels under a single new name.
    Labels not mentioned in *label_map* are left unchanged.  The result
    is written back to the same column in place.

    Args:
        adata: Annotated data matrix to modify in place.
        col: Key in ``adata.obs`` that holds the categorical clustering
            column to rename.
        label_map: Mapping of ``new_label → [old_label, ...]``.  Every label
            that appears in the value list will be renamed to the corresponding
            key. Example::

                {
                    "T cells":    ["3", "7", "12"],
                    "B cells":    ["1"],
                    "NK cells":   ["5", "6"],
                }

    Returns:
        None: Modifies *adata* in place.

    Raises:
        ValueError: If *col* is not found in ``adata.obs``.
        ValueError: If any label listed in *label_map* values is not present
            in the column's categories.

    Example:
        >>> rename_subcluster_labels(
        ...     adata,
        ...     col="leiden_subclustered",
        ...     label_map={
        ...         "T cells":  ["3", "7"],
        ...         "B cells":  ["1"],
        ...     },
        ... )
    """
    if col not in adata.obs.columns:
        raise ValueError(
            f"Column '{col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Build a flat old→new translation dict and validate all source labels.
    current_cats: set = set(adata.obs[col].cat.categories.tolist())
    rename: dict[str, str] = {}
    for new_label, old_labels in label_map.items():
        for old in old_labels:
            old = str(old)
            if old not in current_cats:
                raise ValueError(
                    f"Label '{old}' (mapped to '{new_label}') is not present "
                    f"in column '{col}'. Available categories: "
                    f"{sorted(current_cats)}"
                )
            rename[old] = new_label

    # Derive the new ordered category list, preserving original order for
    # untouched labels and inserting new names at the position of their first
    # source label.
    seen_new: dict[str, None] = {}  # ordered set via dict keys
    new_cats: list[str] = []
    for cat in adata.obs[col].cat.categories.tolist():
        target = rename.get(str(cat), str(cat))
        if target not in seen_new:
            seen_new[target] = None
            new_cats.append(target)

    new_values = adata.obs[col].map(lambda x: rename.get(str(x), str(x)))
    adata.obs[col] = pd.Categorical(new_values, categories=new_cats, ordered=False)


def spatial_split_clusters(
    adata: ad.AnnData,
    cluster_col: str,
    categories: Sequence[Union[str, int]],
    basis: str = "X_umap",
    method: Literal["dbscan", "hdbscan"] = "dbscan",
    eps: float = 2.0,
    min_samples: int = 15,
    min_cluster_size: int = 50,
    assign_noise: bool = True,
    key_added: Optional[str] = None,
    clean: bool = True,
) -> None:
    """Split cluster categories that occupy disjoint regions on an embedding.

    For each category listed in *categories*, a density-based clustering
    algorithm (DBSCAN or HDBSCAN) is applied to the embedding coordinates
    of cells belonging to that category.  When more than one spatial group
    is detected the category is split into numbered sub-categories following
    the ``sc.tl.leiden`` convention (e.g. ``"3"`` → ``"3,1"``, ``"3,2"``).
    Categories that remain spatially contiguous are left untouched.

    Args:
        adata: Annotated data matrix.  Must contain the embedding stored at
            ``adata.obsm[basis]``.
        cluster_col: Key in ``adata.obs`` with the cluster assignments to
            split.
        categories: Cluster labels to evaluate for spatial separation.
        basis: Key in ``adata.obsm`` with the 2-D (or n-D) embedding
            coordinates. Defaults to ``"X_umap"``.
        method: Density-based algorithm to use.  ``"dbscan"`` uses
            ``sklearn.cluster.DBSCAN`` with *eps* and *min_samples*;
            ``"hdbscan"`` uses ``sklearn.cluster.HDBSCAN`` with
            *min_cluster_size* and *min_samples*. Defaults to ``"dbscan"``.
        eps: Maximum distance between two samples for DBSCAN to consider them
            in the same neighbourhood.  Only used when
            ``method="dbscan"``. Defaults to ``2.0``.
        min_samples: Core-point neighbourhood size for DBSCAN / HDBSCAN.
            Defaults to ``15``.
        min_cluster_size: Minimum number of cells to form a cluster.  Only
            used when ``method="hdbscan"``. Defaults to ``50``.
        assign_noise: When ``True``, cells labelled as noise are re-assigned
            to the nearest spatial subcluster.  When ``False``, noise cells
            keep the original category label. Defaults to ``True``.
        key_added: Column name for the result.  When ``None``, defaults to
            ``"{cluster_col}_spatial_split"``.
        clean: When ``True``, all categories in the result column are renamed
            to consecutive integer strings.  When ``False``, the raw
            comma-separated labels are kept. Defaults to ``True``.

    Returns:
        None: Modifies *adata* in place.

    Raises:
        ValueError: If *cluster_col* is not in ``adata.obs`` or *basis* is
            not in ``adata.obsm``.

    Example:
        >>> spatial_split_clusters(
        ...     adata,
        ...     cluster_col="leiden",
        ...     categories=["3", "7"],
        ...     method="dbscan",
        ...     eps=2.0,
        ... )
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if cluster_col not in adata.obs.columns:
        raise ValueError(
            f"cluster_col '{cluster_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if basis not in adata.obsm:
        raise ValueError(
            f"basis '{basis}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    out_col = key_added if key_added is not None else f"{cluster_col}_spatial_split"
    coords = adata.obsm[basis]

    # Start from a string copy of the original column.
    labels = adata.obs[cluster_col].astype(str).copy()

    categories = [str(c) for c in categories]

    for cat in categories:
        mask = labels == cat
        n_cells = mask.sum()
        if n_cells == 0:
            continue

        cat_coords = coords[mask.values]

        # ---- density-based clustering --------------------------------
        spatial_labels = _density_cluster(
            cat_coords,
            method=method,
            eps=eps,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        )

        n_spatial = len(set(spatial_labels) - {-1})

        # Nothing to split – single contiguous island (or all noise).
        if n_spatial <= 1:
            continue

        # ---- assign noise to nearest subcluster ----------------------
        if assign_noise and -1 in spatial_labels:
            spatial_labels = _assign_noise_to_nearest(
                cat_coords, spatial_labels
            )

        # ---- build new labels ----------------------------------------
        # Map the 0-based density labels to 1-based sub-indices so the
        # output matches the sc.tl.leiden convention ("3,1", "3,2", …).
        unique_spatial = sorted(set(spatial_labels) - {-1})
        remap = {old: new for new, old in enumerate(unique_spatial, start=1)}

        new_cat_labels = np.array(
            [
                f"{cat},{remap[sl]}" if sl != -1 else cat
                for sl in spatial_labels
            ]
        )
        labels.values[mask.values] = new_cat_labels

    # ------------------------------------------------------------------
    # Store result
    # ------------------------------------------------------------------
    sorted_cats = sorted(set(labels), key=_sort_key)
    adata.obs[out_col] = pd.Categorical(labels, categories=sorted_cats)

    if clean:
        _clean_subcluster_labels(adata, out_col)


# ---------------------------------------------------------------------------
# Spatial-split helpers
# ---------------------------------------------------------------------------


def _density_cluster(
    coords: np.ndarray,
    method: str,
    eps: float,
    min_samples: int,
    min_cluster_size: int,
) -> np.ndarray:
    """Run DBSCAN or HDBSCAN on embedding coordinates.

    Args:
        coords: Embedding coordinates, shape ``(n_cells, n_dims)``.
        method: Algorithm choice — ``"dbscan"`` or ``"hdbscan"``.
        eps: DBSCAN neighbourhood radius.
        min_samples: Core-point neighbourhood size.
        min_cluster_size: HDBSCAN minimum cluster size.

    Returns:
        Array of integer cluster labels, shape ``(n_cells,)``.
        ``-1`` denotes noise.
    """
    if method == "dbscan":
        from sklearn.cluster import DBSCAN

        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
    else:
        raise ValueError(
            f"method must be 'dbscan' or 'hdbscan', got '{method!r}'"
        )

    return model.fit_predict(coords)


def _assign_noise_to_nearest(
    coords: np.ndarray,
    spatial_labels: np.ndarray,
) -> np.ndarray:
    """Re-assign noise points (``-1``) to the nearest subcluster centroid.

    Args:
        coords: Embedding coordinates for cells in one original cluster,
            shape ``(n_cells, n_dims)``.
        spatial_labels: Labels from density-based clustering, shape
            ``(n_cells,)``.  ``-1`` indicates noise.

    Returns:
        Updated labels array with no ``-1`` entries.
    """
    spatial_labels = spatial_labels.copy()
    noise_mask = spatial_labels == -1
    if not noise_mask.any():
        return spatial_labels

    # Compute centroids of each non-noise cluster.
    unique_labels = sorted(set(spatial_labels) - {-1})
    centroids = np.array(
        [coords[spatial_labels == lab].mean(axis=0) for lab in unique_labels]
    )

    # Assign each noise point to the closest centroid.
    noise_coords = coords[noise_mask]
    # shape: (n_noise, n_clusters)
    dists = np.linalg.norm(
        noise_coords[:, None, :] - centroids[None, :, :], axis=2
    )
    nearest = np.array(unique_labels)[dists.argmin(axis=1)]
    spatial_labels[noise_mask] = nearest

    return spatial_labels



# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_spatial_split_diagnostics(
    adata: ad.AnnData,
    cluster_col: str,
    categories: Sequence[Union[str, int]],
    basis: str = "X_umap",
    method: Literal["dbscan", "hdbscan"] = "dbscan",
    eps: float = 2.0,
    min_samples: int = 15,
    min_cluster_size: int = 50,
    point_size: float = 3.0,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.5, 4.0),
) -> plt.Figure:
    """Visualise the effect of density-based spatial splitting before committing.

    For every requested category three side-by-side panels are drawn:

    1. **Original cluster** – all cells of the category coloured uniformly
       on the full embedding (greyed-out background).
    2. **Density split** – the same cells coloured by the DBSCAN / HDBSCAN
       label (noise shown in grey).
    3. **Pairwise-distance histogram** – distribution of pairwise
       distances between cells in the cluster on the embedding, with the
       current *eps* value shown as a vertical line (DBSCAN only).

    Args:
        adata: Annotated data matrix.
        cluster_col: Key in ``adata.obs`` with the cluster assignments.
        categories: Cluster labels to inspect.
        basis: Key in ``adata.obsm`` with the embedding coordinates.
            Defaults to ``"X_umap"``.
        method: Density-based algorithm to preview.  Defaults to
            ``"dbscan"``.
        eps: DBSCAN neighbourhood radius. Defaults to ``2.0``.
        min_samples: Core-point neighbourhood size. Defaults to ``15``.
        min_cluster_size: HDBSCAN minimum cluster size. Defaults to ``50``.
        point_size: Scatter-plot point size. Defaults to ``3.0``.
        ncols: Number of panel columns per category row (normally 3).
            Defaults to ``3``.
        figsize_per_panel: ``(width, height)`` of each individual panel in
            inches. Defaults to ``(4.5, 4.0)``.

    Returns:
        The diagnostic matplotlib figure.

    Example:
        >>> fig = plot_spatial_split_diagnostics(
        ...     adata,
        ...     cluster_col="leiden",
        ...     categories=["3", "7"],
        ...     eps=1.5,
        ... )
        >>> fig.savefig("spatial_split_diagnostics.png", dpi=150)
    """
    from sklearn.metrics import pairwise_distances

    if cluster_col not in adata.obs.columns:
        raise ValueError(
            f"cluster_col '{cluster_col}' not found in adata.obs."
        )
    if basis not in adata.obsm:
        raise ValueError(
            f"basis '{basis}' not found in adata.obsm."
        )

    coords = adata.obsm[basis]
    all_labels = adata.obs[cluster_col].astype(str).values
    categories = [str(c) for c in categories]

    n_rows = len(categories)
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols,
                 figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for row, cat in enumerate(categories):
        mask = all_labels == cat
        cat_coords = coords[mask]

        # ---- density clustering --------------------------------------
        spatial_labels = _density_cluster(
            cat_coords,
            method=method,
            eps=eps,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        )
        unique_spatial = sorted(set(spatial_labels) - {-1})
        n_spatial = len(unique_spatial)

        # ---- Panel 1: original cluster on full embedding -------------
        ax1 = axs[row, 0]
        ax1.scatter(
            coords[:, 0], coords[:, 1],
            c="#e0e0e0", s=point_size * 0.3, rasterized=True,
        )
        ax1.scatter(
            cat_coords[:, 0], cat_coords[:, 1],
            c="#1f77b4", s=point_size, rasterized=True,
        )
        ax1.set_title(f"Cluster {cat} – Original  (n={mask.sum():,})", fontsize=10)
        ax1.set_xlabel(f"{basis.replace('X_', '')}1", fontsize=8)
        ax1.set_ylabel(f"{basis.replace('X_', '')}2", fontsize=8)
        ax1.tick_params(labelsize=7)
        ax1.grid(False)
        for spine in ax1.spines.values():
            spine.set_visible(False)

        # ---- Panel 2: density split coloured -------------------------
        ax2 = axs[row, 1]
        ax2.scatter(
            coords[:, 0], coords[:, 1],
            c="#e0e0e0", s=point_size * 0.3, rasterized=True,
        )

        # Colour palette for subclusters (skip -1 = noise).
        cmap = matplotlib.colormaps["tab10"]
        for i, lab in enumerate(unique_spatial):
            sub_mask = spatial_labels == lab
            ax2.scatter(
                cat_coords[sub_mask, 0], cat_coords[sub_mask, 1],
                c=[cmap(i % 10)], s=point_size, label=f"{cat},{i + 1}",
                rasterized=True,
            )
        # Noise in grey.
        noise_mask = spatial_labels == -1
        if noise_mask.any():
            ax2.scatter(
                cat_coords[noise_mask, 0], cat_coords[noise_mask, 1],
                c="#999999", s=point_size * 0.6, label="noise",
                rasterized=True, alpha=0.5,
            )
        ax2.legend(fontsize=7, markerscale=2, frameon=True, loc="best")
        title_method = method.upper()
        ax2.set_title(
            f"Cluster {cat} – {title_method} → {n_spatial} group(s)",
            fontsize=10,
        )
        ax2.set_xlabel(f"{basis.replace('X_', '')}1", fontsize=8)
        ax2.set_ylabel(f"{basis.replace('X_', '')}2", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        # ---- Panel 3: pairwise distance histogram --------------------
        ax3 = axs[row, 2]
        # Sub-sample for performance (pairwise is O(n²)).
        max_sample = 2000
        if len(cat_coords) > max_sample:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(cat_coords), max_sample, replace=False)
            sample_coords = cat_coords[idx]
        else:
            sample_coords = cat_coords

        dists = pairwise_distances(sample_coords).ravel()
        # Remove self-distances (zeros).
        dists = dists[dists > 0]

        ax3.hist(dists, bins=80, color="#5a9bd5", edgecolor="none", alpha=0.8)
        if method == "dbscan":
            ax3.axvline(
                eps, color="red", linestyle="--", linewidth=1.2,
                label=f"eps = {eps}",
            )
            ax3.legend(fontsize=8)
        elif method == "hdbscan":
            # Show min_cluster_size as text annotation.
            ax3.text(
                0.97, 0.95,
                f"min_cluster_size = {min_cluster_size}",
                transform=ax3.transAxes,
                fontsize=7, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax3.set_title("Pairwise distances", fontsize=10)
        ax3.set_xlabel("Distance", fontsize=8)
        ax3.set_ylabel("Count", fontsize=8)
        ax3.tick_params(labelsize=7)

    # Add a bold row title on the left side of each row of panels.
    left_margin = 0.06
    plt.tight_layout(rect=[left_margin, 0, 1, 1])
    fig.subplots_adjust(hspace=0.55)
    for row, cat in enumerate(categories):
        # y position: vertical centre of the row.
        y = 1.0 - (row + 0.5) / n_rows
        fig.text(
            left_margin - 0.01, y,
            f"Cluster {cat}",
            ha="right", va="center",
            fontsize=16, fontweight="bold",
        )
    return fig
