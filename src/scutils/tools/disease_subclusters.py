"""
detect_disease_enriched_subclusters
===================================
Identify spatially contiguous regions on a UMAP embedding that are
statistically enriched for specific diseases or conditions.

Pre-requisites
--------------
Before calling this function, the user must:

1. Compute a kNN graph (stored in ``adata.obsp['connectivities']`` /
   ``adata.obsp['distances']``)::

       import scanpy as sc
       sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')

Pipeline
--------
1.  Over-cluster the *entire* dataset with high-resolution Leiden.
    The resolution is chosen automatically from the dataset size:
    ``max(2.0, sqrt(n_cells / 1000))``.
2.  **Fisher exact test screening** — select micro-clusters enriched
    for each disease (fold ≥ ``min_enrichment_fold``).  Optionally
    tests against a user-specified reference group.
3.  **(Optional) Combine diseases** — pool disease labels into one
    binary label, or take the union of per-disease enrichments.
4.  **kNN-graph connected components** — merge enriched cells via the
    pre-computed neighbour graph.
5.  **GMM splitting** — fit a Gaussian Mixture Model (BIC-selected *k*)
    on 2-D UMAP coordinates to split multimodal components.
6.  **kNN graph re-validation** — confirm each piece is a connected
    subgraph.
7.  **DBSCAN spatial enforcement** — final pass ensuring every piece
    occupies exactly one contiguous hub in UMAP space.  ``eps`` is
    estimated automatically per piece from the local k-NN distance
    distribution, scaled by ``spatial_sensitivity``.  Fragments
    smaller than ``min_subcluster_size`` are reassigned to their
    nearest valid DBSCAN cluster.
8.  **(Optional) Cell-type splitting** — intersect each subcluster
    with each cell type; every piece must independently satisfy
    ``min_subcluster_size``, ``min_enrichment_fold``, and
    ``enrichment_fdr``.
9.  Final Fisher exact test + Benjamini–Hochberg FDR correction.
"""

from __future__ import annotations

import re
import warnings
from typing import Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.figure import Figure
from scipy.sparse.csgraph import connected_components as sparse_connected_components
from scipy.stats import fisher_exact
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leiden(
    adata_sub: AnnData,
    resolution: float,
    method: str,
    key_added: str,
) -> None:
    """Run Leiden clustering on *adata_sub* using a pre-computed kNN
    graph.

    Raises
    ------
    KeyError
        If the kNN graph has not been computed.
    """
    if "connectivities" not in adata_sub.obsp:
        raise KeyError(
            "Pre-computed kNN graph not found in "
            "adata.obsp['connectivities']. Please run:\n"
            "    import scanpy as sc\n"
            "    sc.pp.neighbors(adata, n_neighbors=15, "
            "use_rep='X_pca')\n"
            "before calling this function."
        )

    if method == "rapids":
        try:
            import rapids_singlecell as rsc

            rsc.tl.leiden(
                adata_sub,
                resolution=resolution,
                key_added=key_added,
            )
        except ImportError:
            raise ImportError(
                "rapids_singlecell is required for method='rapids'. "
                "Install it or use method='scanpy'."
            )
    else:
        import scanpy as _sc

        _sc.tl.leiden(
            adata_sub,
            resolution=resolution,
            key_added=key_added,
        )


# ---- Fisher exact test ----------------------------------------------------


def _enrichment_test(
    cluster_mask: np.ndarray,
    disease_mask: np.ndarray,
    total_cells: int,
    reference_mask: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """One-sided Fisher exact test for enrichment of *disease_mask*
    cells inside *cluster_mask*.

    When *reference_mask* is provided the 2×2 contingency table is
    restricted to cells that are either disease-positive **or** in
    the reference group.  All other cells (e.g. other diseases) are
    excluded from the test.

    Returns ``(fold_enrichment, p_value)``.
    """
    if reference_mask is not None:
        universe = disease_mask | reference_mask
        cm = cluster_mask & universe
        dm = disease_mask & universe
        n = int(universe.sum())

        a = int(np.sum(cm & dm))
        b = int(np.sum(cm & ~dm))
        c = int(np.sum(~cm & dm))
        d = int(np.sum(~cm & ~dm))
    else:
        n = total_cells
        a = int(np.sum(cluster_mask & disease_mask))
        b = int(np.sum(cluster_mask & ~disease_mask))
        c = int(np.sum(~cluster_mask & disease_mask))
        d = int(np.sum(~cluster_mask & ~disease_mask))

    total_disease = a + c
    total_cluster = a + b
    if total_disease == 0 or total_cluster == 0 or n == 0:
        return 0.0, 1.0

    expected_prop = total_disease / n
    observed_prop = a / total_cluster if total_cluster > 0 else 0.0
    fold = observed_prop / expected_prop if expected_prop > 0 else 0.0

    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return fold, p


# ---- Reference helpers ----------------------------------------------------


def _resolve_reference_mask(
    ref_mask_full: Optional[np.ndarray],
    n_ref: int,
    min_reference_cells: int,
) -> Optional[np.ndarray]:
    """Return the reference mask if enough cells exist, else ``None``
    (signalling a fallback to the full dataset).
    """
    if ref_mask_full is not None and n_ref >= min_reference_cells:
        return ref_mask_full
    return None


def _reference_used_string(
    ref_labels: list[str],
    ref_mask_full: Optional[np.ndarray],
    n_ref: int,
    min_reference_cells: int,
) -> str:
    """Human-readable string describing which reference was used.

    * ``"all"`` — no reference group requested; full dataset used.
    * Comma-separated list of reference labels — when the reference
      group had enough cells.
    * ``"all (fallback)"`` — reference group was too small; fell
      back to the full dataset.
    """
    if ref_mask_full is None:
        return "all"
    if n_ref >= min_reference_cells:
        return ", ".join(sorted(ref_labels))
    return "all (fallback)"


# ---- Auto resolution ------------------------------------------------------


def _auto_resolution(n_cells: int) -> float:
    """Compute a Leiden resolution that scales with dataset size.

    Uses ``max(2.0, sqrt(n_cells / 1000))``, which gives:

    ==========  ==========
    n_cells     resolution
    ==========  ==========
    10 000      3.2
    30 000      5.5
    50 000      7.1
    100 000     10.0
    150 000     12.2
    300 000     17.3
    500 000     22.4
    ==========  ==========
    """
    return max(2.0, np.sqrt(n_cells / 1000))


# ---- Spatial sensitivity ---------------------------------------------------


_SPATIAL_MULTIPLIERS: dict[str, float] = {
    "low": 2.0,
    "medium": 1.5,
    "high": 1.0,
    "very_high": 0.7,
}


def _resolve_spatial_multiplier(
    spatial_sensitivity: str,
) -> float:
    """Map a *spatial_sensitivity* name to a DBSCAN eps multiplier."""
    key = spatial_sensitivity.lower().strip()
    if key not in _SPATIAL_MULTIPLIERS:
        valid = ", ".join(
            f"'{k}'" for k in _SPATIAL_MULTIPLIERS
        )
        raise ValueError(
            f"spatial_sensitivity must be one of {valid}, "
            f"got '{spatial_sensitivity}'"
        )
    return _SPATIAL_MULTIPLIERS[key]


# ---- kNN-graph connected components ---------------------------------------


def _graph_connected_components(
    adata_sub: AnnData,
    cell_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Extract the kNN sub-graph for *cell_mask* cells and return its
    connected components.

    Returns
    -------
    n_components : int
    comp_labels : np.ndarray
        Integer component label for every ``True`` cell in *cell_mask*.
    """
    indices = np.where(cell_mask)[0]
    if len(indices) == 0:
        return 0, np.array([], dtype=int)

    sub_graph = adata_sub.obsp["connectivities"][
        np.ix_(indices, indices)
    ]
    n_comp, comp_labels = sparse_connected_components(
        sub_graph, directed=False, return_labels=True
    )
    return n_comp, comp_labels


# ---- GMM splitting with graph re-validation ------------------------------


def _gmm_split(
    umap_coords: np.ndarray,
    max_k: int = 5,
    covariance_type: str = "full",
    random_state: int = 0,
) -> np.ndarray:
    """Fit a GMM to 2-D *umap_coords*, choosing *k* via BIC.

    Returns an integer label array of length ``len(umap_coords)``.
    """
    n = len(umap_coords)
    if n < 2:
        return np.zeros(n, dtype=int)

    upper_k = min(max_k, n)
    best_k = 1
    best_bic = np.inf

    for k in range(1, upper_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=300,
        )
        try:
            gmm.fit(umap_coords)
            bic = gmm.bic(umap_coords)
        except Exception:
            continue
        if bic < best_bic:
            best_bic = bic
            best_k = k

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=300,
    )
    gmm.fit(umap_coords)
    return gmm.predict(umap_coords)


def _split_and_revalidate(
    adata_sub: AnnData,
    cell_indices: np.ndarray,
    umap_key: str,
    gmm_max_components: int,
    gmm_covariance_type: str,
) -> list[np.ndarray]:
    """GMM split on UMAP coordinates → kNN re-validation.

    Returns
    -------
    list[np.ndarray]
        Index arrays (into *adata_sub*), one per final contiguous
        piece.
    """
    umap_coords = adata_sub.obsm[umap_key]
    coords = umap_coords[cell_indices]

    gmm_labels = _gmm_split(
        coords,
        max_k=gmm_max_components,
        covariance_type=gmm_covariance_type,
    )

    unique_gmm = np.unique(gmm_labels)
    if len(unique_gmm) <= 1:
        return _verify_connectivity(adata_sub, cell_indices)

    pieces: list[np.ndarray] = []
    for g in unique_gmm:
        g_idx = cell_indices[gmm_labels == g]
        pieces.extend(_verify_connectivity(adata_sub, g_idx))

    return pieces


def _verify_connectivity(
    adata_sub: AnnData,
    cell_indices: np.ndarray,
) -> list[np.ndarray]:
    """Split *cell_indices* into kNN-graph connected components."""
    if len(cell_indices) <= 1:
        return [cell_indices]

    sub_graph = adata_sub.obsp["connectivities"][
        np.ix_(cell_indices, cell_indices)
    ]
    n_comp, comp_labels = sparse_connected_components(
        sub_graph, directed=False, return_labels=True
    )

    if n_comp == 1:
        return [cell_indices]

    return [cell_indices[comp_labels == c] for c in range(n_comp)]


# ---- DBSCAN spatial enforcement -------------------------------------------


def _estimate_dbscan_eps(
    umap_coords: np.ndarray,
    k: int = 15,
    multiplier: float = 1.5,
) -> float:
    """Estimate a reasonable DBSCAN ``eps`` from the data.

    Computes the mean k-th nearest-neighbour distance across all
    points and multiplies by *multiplier*.
    """
    k_use = min(k, len(umap_coords) - 1)
    if k_use < 1:
        return 1.0

    nn = NearestNeighbors(n_neighbors=k_use + 1)
    nn.fit(umap_coords)
    distances, _ = nn.kneighbors(umap_coords)
    kth_distances = distances[:, k_use]
    return float(np.mean(kth_distances) * multiplier)


def _dbscan_enforce_contiguity(
    umap_coords: np.ndarray,
    cell_indices: np.ndarray,
    min_subcluster_size: int,
    eps_multiplier: float = 1.5,
    dbscan_min_samples: int = 15,
) -> list[np.ndarray]:
    """Split *cell_indices* into spatially contiguous UMAP hubs using
    DBSCAN with an automatically estimated ``eps``.

    Fragments smaller than *min_subcluster_size* are reassigned to
    their nearest valid (large enough) DBSCAN cluster.  If no valid
    cluster exists the piece is returned as-is.
    """
    coords = umap_coords[cell_indices]

    if len(cell_indices) < dbscan_min_samples:
        return [cell_indices]

    eps = _estimate_dbscan_eps(
        coords,
        k=min(dbscan_min_samples, len(coords) - 1),
        multiplier=eps_multiplier,
    )

    db = DBSCAN(eps=eps, min_samples=dbscan_min_samples)
    db_labels = db.fit_predict(coords)

    unique_labels = set(db_labels)
    unique_labels.discard(-1)

    if len(unique_labels) <= 1:
        return [cell_indices]

    noise_mask = db_labels == -1
    if noise_mask.any() and not noise_mask.all():
        non_noise_idx = np.where(~noise_mask)[0]
        noise_idx = np.where(noise_mask)[0]
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(coords[non_noise_idx])
        _, nearest = nn.kneighbors(coords[noise_idx])
        for i, ni in enumerate(noise_idx):
            db_labels[ni] = db_labels[non_noise_idx[nearest[i, 0]]]

    cluster_map: dict[int, np.ndarray] = {}
    for lbl in sorted(set(db_labels)):
        if lbl == -1:
            continue
        cluster_map[lbl] = cell_indices[db_labels == lbl]

    valid_labels = [
        lbl
        for lbl, idx in cluster_map.items()
        if len(idx) >= min_subcluster_size
    ]
    small_labels = [
        lbl
        for lbl, idx in cluster_map.items()
        if len(idx) < min_subcluster_size
    ]

    if not valid_labels:
        return [cell_indices]

    if small_labels:
        valid_centroids = np.array([
            umap_coords[cluster_map[lbl]].mean(axis=0)
            for lbl in valid_labels
        ])
        for s_lbl in small_labels:
            s_centroid = umap_coords[cluster_map[s_lbl]].mean(
                axis=0
            )
            dists = np.linalg.norm(
                valid_centroids - s_centroid, axis=1
            )
            nearest_valid = valid_labels[np.argmin(dists)]
            cluster_map[nearest_valid] = np.concatenate([
                cluster_map[nearest_valid],
                cluster_map[s_lbl],
            ])
            del cluster_map[s_lbl]

    return [
        cluster_map[lbl]
        for lbl in sorted(cluster_map.keys())
    ]


# ---- Label sort key -------------------------------------------------------


def _subcluster_sort_key(label: str) -> tuple:
    """Sort key for ``"{subset}|{disease}|sub{id}"`` labels."""
    if label == "background":
        return ("", "", -1)

    parts = label.split("|")
    if len(parts) == 3:
        subset = parts[0]
        disease = parts[1]
        match = re.search(r"(\d+)$", parts[2])
        num = int(match.group(1)) if match else 0
        return (subset, disease, num)

    return (label, "", 0)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def detect_disease_enriched_subclusters(
    adata: AnnData,
    *,
    celltype_key: Optional[str] = None,
    groups_celltype: Optional[Sequence[str]] = None,
    groups_disease: Optional[Sequence[str]] = None,
    disease_key: str = "disease",
    umap_key: str = "X_umap",
    method: Literal["scanpy", "rapids"] = "scanpy",
    min_enrichment_fold: float = 2.0,
    min_subcluster_size: int = 100,
    enrichment_fdr: float = 0.05,
    gmm_max_components: int = 5,
    gmm_covariance_type: str = "full",
    spatial_sensitivity: Literal[
        "low", "medium", "high", "very_high"
    ] = "medium",
    combine_diseases: Optional[Literal["pool", "union"]] = None,
    reference_group: Optional[Union[str, Sequence[str]]] = None,
    min_reference_cells: int = 50,
    result_key: str = "disease_subcluster",
    verbose: bool = True,
) -> None:
    """Detect disease-enriched subclusters on a UMAP embedding.

    The pipeline always runs on the **entire dataset**.  When
    ``celltype_key`` is provided subclusters are intersected with
    each cell type post-hoc and every piece must independently pass
    size and enrichment filters.

    The Leiden resolution for over-clustering is determined
    automatically from the dataset size using
    ``max(2.0, sqrt(n_cells / 1000))``.

    Modifies *adata* in place:

    * ``adata.obs[result_key]`` — categorical labels
      (``"background"`` or ``"{celltype}|{disease}|sub{id}"``).
    * ``adata.uns[f'{result_key}_info']`` — :class:`pandas.DataFrame`
      with one row per (cell-type, subcluster, disease) triple.

    Pre-requisites
    --------------
    A kNN graph must be pre-computed::

        import scanpy as sc
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')

    Parameters
    ----------
    adata
        Annotated data matrix.
    celltype_key
        Column in ``adata.obs`` with cell-type labels.  ``None`` →
        no cell-type splitting (subset name is ``"all_cells"``).
    groups_celltype
        Subset of cell types to report.  ``None`` = all.
    groups_disease
        Subset of diseases to test.  ``None`` = all.  If
        *reference_group* is specified, those labels are
        automatically excluded from the tested diseases.
    disease_key
        Column in ``adata.obs`` with disease/condition labels.
    umap_key
        Key in ``adata.obsm`` for 2-D UMAP coordinates.
    method
        Backend for Leiden: ``"scanpy"`` (CPU) or ``"rapids"`` (GPU).
    min_enrichment_fold
        Minimum fold-enrichment for micro-cluster screening **and**
        final subcluster reporting.
    min_subcluster_size
        Minimum cells in a reported subcluster.
    enrichment_fdr
        Benjamini–Hochberg FDR threshold.
    gmm_max_components
        Maximum *k* for GMM splitting.
    gmm_covariance_type
        Covariance type for
        :class:`~sklearn.mixture.GaussianMixture`.
    spatial_sensitivity
        Controls how aggressively the DBSCAN spatial-enforcement
        step splits pieces that span multiple UMAP hubs:

        ``"low"``
            Only splits clearly separated hubs (eps multiplier 2.0).
        ``"medium"`` (default)
            Balanced (eps multiplier 1.5).
        ``"high"``
            Splits smaller protrusions from main clusters
            (eps multiplier 1.0).
        ``"very_high"``
            Very aggressive — splits anything that sticks out
            (eps multiplier 0.7).
    combine_diseases
        How to handle multiple diseases:

        ``None`` (default)
            Independent per-disease pipeline.
        ``"pool"``
            Merge all diseases in *groups_disease* into a single
            binary label ``"combined"`` before the pipeline.
        ``"union"``
            Screen micro-clusters per-disease; a micro-cluster is
            enriched if it passes for **any** disease.  Merge/split
            uses the union of enriched micro-clusters and the
            combined disease mask.
    reference_group
        If ``None`` (default), the enrichment test uses the full
        dataset as background.  If a string or list of strings,
        the 2×2 contingency table is restricted to disease cells
        plus reference-group cells only — other diseases are
        excluded.  Falls back to the full dataset when fewer than
        *min_reference_cells* reference cells are available.

        The reference labels are automatically removed from the
        set of diseases to test.
    min_reference_cells
        Minimum number of reference-group cells required; below
        this the test falls back to the full dataset.
    result_key
        Column name written to ``adata.obs``.
    verbose
        Print progress.
    """

    # ------------------------------------------------------------------
    # 0.  Validate inputs
    # ------------------------------------------------------------------
    if umap_key not in adata.obsm:
        raise KeyError(
            f"UMAP coordinates not found at adata.obsm['{umap_key}']"
        )
    if disease_key not in adata.obs.columns:
        raise KeyError(
            f"Disease labels not found at adata.obs['{disease_key}']"
        )
    if "connectivities" not in adata.obsp:
        raise KeyError(
            "Pre-computed kNN graph not found in "
            "adata.obsp['connectivities']. Please run:\n"
            "    import scanpy as sc\n"
            "    sc.pp.neighbors(adata, n_neighbors=15, "
            "use_rep='X_pca')\n"
            "before calling this function."
        )
    if celltype_key is not None and celltype_key not in adata.obs.columns:
        raise KeyError(
            f"Cell-type column not found at "
            f"adata.obs['{celltype_key}']"
        )
    if adata.obsm[umap_key].shape[1] < 2:
        raise ValueError(
            "UMAP coordinates must have at least 2 dimensions."
        )

    eps_multiplier = _resolve_spatial_multiplier(
        spatial_sensitivity
    )

    # --- Reference group -----------------------------------------------
    if reference_group is not None:
        if isinstance(reference_group, str):
            ref_labels: list[str] = [reference_group]
        else:
            ref_labels = list(reference_group)
        ref_mask_full: Optional[np.ndarray] = (
            adata.obs[disease_key].isin(ref_labels).values
        )
        n_ref = int(ref_mask_full.sum())
    else:
        ref_labels = []
        ref_mask_full = None
        n_ref = 0

    # --- Disease groups (excluding reference) --------------------------
    diseases_all = adata.obs[disease_key].unique().tolist()
    if groups_disease is not None:
        diseases_all = [d for d in diseases_all if d in groups_disease]
    if ref_labels:
        diseases_all = [
            d for d in diseases_all if d not in ref_labels
        ]

    if not diseases_all:
        raise ValueError(
            "No diseases left to test after excluding the "
            "reference group. Check 'groups_disease' and "
            "'reference_group'."
        )

    # Cell types
    if celltype_key is not None:
        all_celltypes = adata.obs[celltype_key].unique().tolist()
        if groups_celltype is not None:
            all_celltypes = [
                c for c in all_celltypes if c in groups_celltype
            ]
    else:
        all_celltypes = None

    umap_coords = adata.obsm[umap_key]
    disease_col = adata.obs[disease_key].values
    n_total = adata.n_obs

    # Reference status string
    ref_m = _resolve_reference_mask(
        ref_mask_full, n_ref, min_reference_cells
    )
    ref_used_str = _reference_used_string(
        ref_labels, ref_mask_full, n_ref, min_reference_cells
    )

    if verbose:
        print(
            f"[detect_disease_enriched_subclusters] "
            f"Processing {n_total} cells, "
            f"{len(diseases_all)} disease(s): {diseases_all}"
        )
        if all_celltypes is not None:
            print(
                f"  Cell types for post-hoc splitting: "
                f"{all_celltypes}"
            )
        if ref_mask_full is not None:
            print(
                f"  Reference group: {ref_labels} "
                f"({n_ref} cells) — using: '{ref_used_str}'"
            )
        if combine_diseases is not None:
            print(f"  Combine diseases mode: '{combine_diseases}'")
        print(
            f"  Spatial sensitivity: '{spatial_sensitivity}' "
            f"(DBSCAN eps multiplier={eps_multiplier})"
        )

    # ------------------------------------------------------------------
    # 1.  Over-cluster the ENTIRE dataset
    # ------------------------------------------------------------------
    micro_key = "_micro_global"
    resolution = _auto_resolution(n_total)

    _leiden(
        adata,
        resolution=resolution,
        method=method,
        key_added=micro_key,
    )

    micro_labels = adata.obs[micro_key].values.astype(str)
    n_micros = len(np.unique(micro_labels))

    if verbose:
        print(
            f"  Global over-clustering: {n_total} cells → "
            f"{n_micros} micro-clusters "
            f"(auto resolution={resolution:.1f})"
        )

    # ------------------------------------------------------------------
    # 2.  Determine disease masks and screening strategy
    # ------------------------------------------------------------------
    raw_pieces: list[tuple[str, np.ndarray]] = []

    if combine_diseases == "pool":
        combined_mask = np.isin(disease_col, diseases_all)
        if verbose:
            print(
                f"  Pooling {len(diseases_all)} diseases into "
                f"'combined' ({int(combined_mask.sum())} cells)"
            )

        pieces = _screen_merge_split(
            adata=adata,
            micro_labels=micro_labels,
            disease_mask=combined_mask,
            disease_label="combined",
            n_total=n_total,
            n_micros=n_micros,
            umap_coords=umap_coords,
            ref_mask=ref_m,
            min_enrichment_fold=min_enrichment_fold,
            min_subcluster_size=min_subcluster_size,
            umap_key=umap_key,
            gmm_max_components=gmm_max_components,
            gmm_covariance_type=gmm_covariance_type,
            eps_multiplier=eps_multiplier,
            verbose=verbose,
        )
        for p in pieces:
            raw_pieces.append(("combined", p))

    elif combine_diseases == "union":
        unique_micros = np.unique(micro_labels)
        all_enriched: set[str] = set()

        for disease in diseases_all:
            d_mask = disease_col == disease
            n_d = int(d_mask.sum())

            for mc in unique_micros:
                mc_mask = micro_labels == mc
                fold, _ = _enrichment_test(
                    mc_mask, d_mask, n_total, ref_m
                )
                if fold >= min_enrichment_fold:
                    all_enriched.add(mc)

            if verbose:
                print(
                    f"    [{disease}] {n_d} disease cells — "
                    f"running total enriched: "
                    f"{len(all_enriched)}/{n_micros}"
                )

        combined_mask = np.isin(disease_col, diseases_all)
        enriched_cell_mask = np.isin(
            micro_labels, list(all_enriched)
        )
        enriched_indices = np.where(enriched_cell_mask)[0]

        if verbose:
            print(
                f"  Union mode: {len(all_enriched)} enriched "
                f"micro-clusters, {len(enriched_indices)} cells"
            )

        if len(enriched_indices) >= min_subcluster_size:
            pieces = _merge_split_pipeline(
                adata=adata,
                enriched_cell_mask=enriched_cell_mask,
                enriched_indices=enriched_indices,
                umap_coords=umap_coords,
                min_subcluster_size=min_subcluster_size,
                umap_key=umap_key,
                gmm_max_components=gmm_max_components,
                gmm_covariance_type=gmm_covariance_type,
                eps_multiplier=eps_multiplier,
                verbose=verbose,
                disease_label="combined",
            )
            for p in pieces:
                raw_pieces.append(("combined", p))

    else:
        for disease in diseases_all:
            d_mask = disease_col == disease

            pieces = _screen_merge_split(
                adata=adata,
                micro_labels=micro_labels,
                disease_mask=d_mask,
                disease_label=disease,
                n_total=n_total,
                n_micros=n_micros,
                umap_coords=umap_coords,
                ref_mask=ref_m,
                min_enrichment_fold=min_enrichment_fold,
                min_subcluster_size=min_subcluster_size,
                umap_key=umap_key,
                gmm_max_components=gmm_max_components,
                gmm_covariance_type=gmm_covariance_type,
                eps_multiplier=eps_multiplier,
                verbose=verbose,
            )
            for p in pieces:
                raw_pieces.append((disease, p))

    if verbose:
        print(
            f"  Total raw pieces across all diseases: "
            f"{len(raw_pieces)}"
        )

    # ------------------------------------------------------------------
    # 3.  Cell-type splitting & final enrichment filtering
    # ------------------------------------------------------------------
    labels = pd.Series(
        "background", index=adata.obs_names, dtype=object
    )
    info_rows: list[dict] = []
    sub_counter = 0

    for disease_label, piece_idx in raw_pieces:
        if disease_label == "combined":
            d_mask = np.isin(disease_col, diseases_all)
        else:
            d_mask = disease_col == disease_label

        if all_celltypes is not None:
            ct_col = adata.obs[celltype_key].values
            piece_ct = ct_col[piece_idx]

            for ct in all_celltypes:
                ct_local_mask = piece_ct == ct
                ct_piece_idx = piece_idx[ct_local_mask]

                n_cells = len(ct_piece_idx)
                if n_cells < min_subcluster_size:
                    continue

                ct_piece_mask = np.zeros(n_total, dtype=bool)
                ct_piece_mask[ct_piece_idx] = True

                fold, pval = _enrichment_test(
                    ct_piece_mask, d_mask, n_total, ref_m
                )
                if fold < min_enrichment_fold:
                    continue

                n_disease = int(
                    (ct_piece_mask & d_mask).sum()
                )

                diseases_contrib = _diseases_contributing(
                    disease_col, ct_piece_idx, d_mask,
                    combine_diseases,
                )

                sub_counter += 1
                label_str = (
                    f"{ct}|{disease_label}|sub{sub_counter}"
                )
                labels.iloc[ct_piece_idx] = label_str

                info_rows.append(
                    {
                        "subset": ct,
                        "subcluster": f"sub{sub_counter}",
                        "disease": disease_label,
                        "n_cells": n_cells,
                        "n_disease_cells": n_disease,
                        "fold_enrichment": fold,
                        "pvalue": pval,
                        "pvalue_adj": np.nan,
                        "label": label_str,
                        "reference_used": ref_used_str,
                        "diseases_contributing":
                            diseases_contrib,
                    }
                )
        else:
            piece_mask = np.zeros(n_total, dtype=bool)
            piece_mask[piece_idx] = True

            fold, pval = _enrichment_test(
                piece_mask, d_mask, n_total, ref_m
            )
            if fold < min_enrichment_fold:
                continue

            n_cells = len(piece_idx)
            n_disease = int((piece_mask & d_mask).sum())

            diseases_contrib = _diseases_contributing(
                disease_col, piece_idx, d_mask,
                combine_diseases,
            )

            sub_counter += 1
            label_str = (
                f"all_cells|{disease_label}|sub{sub_counter}"
            )
            labels.iloc[piece_idx] = label_str

            info_rows.append(
                {
                    "subset": "all_cells",
                    "subcluster": f"sub{sub_counter}",
                    "disease": disease_label,
                    "n_cells": n_cells,
                    "n_disease_cells": n_disease,
                    "fold_enrichment": fold,
                    "pvalue": pval,
                    "pvalue_adj": np.nan,
                    "label": label_str,
                    "reference_used": ref_used_str,
                    "diseases_contributing":
                        diseases_contrib,
                }
            )

    if verbose:
        print(
            f"  After cell-type splitting & fold filter: "
            f"{len(info_rows)} candidate(s)"
        )

    # ------------------------------------------------------------------
    # 4.  Multiple-testing correction
    # ------------------------------------------------------------------
    if info_rows:
        info_df = pd.DataFrame(info_rows)
        _, pvals_adj, _, _ = multipletests(
            info_df["pvalue"].values,
            alpha=enrichment_fdr,
            method="fdr_bh",
        )
        info_df["pvalue_adj"] = pvals_adj

        keep = info_df["pvalue_adj"] < enrichment_fdr
        n_before = len(info_df)
        removed_labels = set(info_df.loc[~keep, "label"])
        info_df = info_df.loc[keep].reset_index(drop=True)

        if removed_labels:
            labels[labels.isin(removed_labels)] = "background"

        if verbose:
            n_removed = n_before - len(info_df)
            print(
                f"  FDR correction (α={enrichment_fdr}): "
                f"{n_before} candidate(s) → {len(info_df)} "
                f"retained, {n_removed} removed"
            )

        if combine_diseases is None:
            info_df = info_df.drop(
                columns=["diseases_contributing"],
                errors="ignore",
            )
    else:
        cols = [
            "subset", "subcluster", "disease", "n_cells",
            "n_disease_cells", "fold_enrichment", "pvalue",
            "pvalue_adj", "label", "reference_used",
        ]
        if combine_diseases is not None:
            cols.append("diseases_contributing")
        info_df = pd.DataFrame(columns=cols)
        if verbose:
            print("  No enriched subclusters found.")

    # ------------------------------------------------------------------
    # 5.  Store results
    # ------------------------------------------------------------------
    if micro_key in adata.obs.columns:
        del adata.obs[micro_key]

    unique_labels = labels.unique().tolist()
    sorted_labels = sorted(unique_labels, key=_subcluster_sort_key)
    adata.obs[result_key] = pd.Categorical(
        labels, categories=sorted_labels, ordered=True
    )
    adata.uns[f"{result_key}_info"] = info_df

    if verbose:
        n_labelled = int((labels != "background").sum())
        n_subclusters = len(info_df)
        print(
            f"\n  Done. {n_subclusters} disease-enriched "
            f"subcluster(s) covering {n_labelled} cells stored "
            f"in adata.obs['{result_key}']."
        )
        if n_subclusters > 0:
            summary = info_df.groupby("subset").agg(
                subclusters=("subcluster", "nunique"),
                diseases=("disease", "nunique"),
                total_cells=("n_cells", "sum"),
            )
            print("  Summary per subset:")
            for row_name, row in summary.iterrows():
                print(
                    f"    {row_name}: {row['subclusters']} "
                    f"subcluster(s), "
                    f"{row['diseases']} disease(s), "
                    f"{row['total_cells']} cells"
                )


# ---------------------------------------------------------------------------
# Internal pipeline helpers
# ---------------------------------------------------------------------------


def _screen_merge_split(
    adata: AnnData,
    micro_labels: np.ndarray,
    disease_mask: np.ndarray,
    disease_label: str,
    n_total: int,
    n_micros: int,
    umap_coords: np.ndarray,
    ref_mask: Optional[np.ndarray],
    min_enrichment_fold: float,
    min_subcluster_size: int,
    umap_key: str,
    gmm_max_components: int,
    gmm_covariance_type: str,
    eps_multiplier: float,
    verbose: bool,
) -> list[np.ndarray]:
    """Screen micro-clusters → kNN connected components → GMM split →
    kNN re-validation → DBSCAN enforcement for a single disease
    (or combined) mask.

    Returns a list of cell-index arrays into *adata*.
    """
    n_disease_total = int(disease_mask.sum())

    unique_micros = np.unique(micro_labels)
    enriched_micros: set[str] = set()

    for mc in unique_micros:
        mc_mask = micro_labels == mc
        fold, _ = _enrichment_test(
            mc_mask, disease_mask, n_total, ref_mask
        )
        if fold >= min_enrichment_fold:
            enriched_micros.add(mc)

    if verbose:
        print(
            f"  [{disease_label}] {n_disease_total} disease cells, "
            f"{len(enriched_micros)}/{n_micros} enriched "
            f"micro-clusters (fold ≥ {min_enrichment_fold})"
        )

    if not enriched_micros:
        return []

    enriched_cell_mask = np.isin(
        micro_labels, list(enriched_micros)
    )
    enriched_indices = np.where(enriched_cell_mask)[0]

    return _merge_split_pipeline(
        adata=adata,
        enriched_cell_mask=enriched_cell_mask,
        enriched_indices=enriched_indices,
        umap_coords=umap_coords,
        min_subcluster_size=min_subcluster_size,
        umap_key=umap_key,
        gmm_max_components=gmm_max_components,
        gmm_covariance_type=gmm_covariance_type,
        eps_multiplier=eps_multiplier,
        verbose=verbose,
        disease_label=disease_label,
    )


def _merge_split_pipeline(
    adata: AnnData,
    enriched_cell_mask: np.ndarray,
    enriched_indices: np.ndarray,
    umap_coords: np.ndarray,
    min_subcluster_size: int,
    umap_key: str,
    gmm_max_components: int,
    gmm_covariance_type: str,
    eps_multiplier: float,
    verbose: bool,
    disease_label: str,
) -> list[np.ndarray]:
    """kNN connected components → GMM split → kNN re-validation →
    DBSCAN spatial enforcement.

    Returns a list of cell-index arrays into *adata*.
    """
    n_comp, comp_labels = _graph_connected_components(
        adata, enriched_cell_mask
    )

    if verbose:
        print(
            f"  [{disease_label}] kNN connected components → "
            f"{n_comp} component(s)"
        )

    gmm_pieces: list[np.ndarray] = []

    for comp_id in range(n_comp):
        comp_cell_indices = enriched_indices[
            comp_labels == comp_id
        ]
        if len(comp_cell_indices) < min_subcluster_size:
            continue

        pieces = _split_and_revalidate(
            adata,
            comp_cell_indices,
            umap_key=umap_key,
            gmm_max_components=gmm_max_components,
            gmm_covariance_type=gmm_covariance_type,
        )

        for piece_idx in pieces:
            if len(piece_idx) >= min_subcluster_size:
                gmm_pieces.append(piece_idx)

    if verbose:
        print(
            f"  [{disease_label}] After GMM split + kNN "
            f"re-validation: {len(gmm_pieces)} piece(s)"
        )

    result: list[np.ndarray] = []

    for piece_idx in gmm_pieces:
        dbscan_pieces = _dbscan_enforce_contiguity(
            umap_coords=umap_coords,
            cell_indices=piece_idx,
            min_subcluster_size=min_subcluster_size,
            eps_multiplier=eps_multiplier,
        )
        for dp in dbscan_pieces:
            if len(dp) >= min_subcluster_size:
                result.append(dp)

    if verbose and len(result) != len(gmm_pieces):
        print(
            f"  [{disease_label}] After DBSCAN enforcement: "
            f"{len(result)} piece(s)"
        )

    return result


def _diseases_contributing(
    disease_col: np.ndarray,
    piece_idx: np.ndarray,
    disease_mask: np.ndarray,
    combine_diseases: Optional[str],
) -> str:
    """Return a comma-separated string of original disease labels
    present among the disease-positive cells in this piece.

    Returns ``""`` when *combine_diseases* is ``None``.
    """
    if combine_diseases is None:
        return ""

    disease_cells_in_piece = disease_col[piece_idx][
        disease_mask[piece_idx]
    ]
    return ", ".join(sorted(set(disease_cells_in_piece)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_disease_enriched_subclusters(
    adata: AnnData,
    *,
    celltype_key: Optional[str] = None,
    disease_key: str = "disease",
    result_key: str = "disease_subcluster",
    umap_key: str = "X_umap",
    split_by: Literal["celltype", "disease"] = "celltype",
    groups: Optional[Sequence[str]] = None,
    ncols: int = 2,
    figsize_panel: Tuple[float, float] = (5, 5),
    bg_dotsize: float = 2,
    fg_dotsize: float = 6,
    bg_alpha: float = 0.3,
    group_bg_alpha: float = 0.5,
    group_bg_color: str = "#808080",
    show: bool = True,
    **umap_kwargs,
) -> Figure:
    """Generate a grid of UMAP panels for disease-enriched subclusters,
    split by cell type **or** by disease.

    Each panel shows one group with three visual layers:

    -  Cells *not* in the current group: light grey.
    -  Cells in the group labelled ``"background"``: darker grey.
    -  Disease-enriched subcluster cells: coloured by label.

    Enrichment statistics are available in
    ``adata.uns[f'{result_key}_info']``.

    Parameters
    ----------
    adata
        Must contain ``adata.obs[result_key]`` and
        ``adata.uns[f'{result_key}_info']``.
    celltype_key
        Column in ``adata.obs`` with cell-type annotations.
    disease_key
        Column in ``adata.obs`` with disease labels.
    result_key
        Column produced by
        :func:`detect_disease_enriched_subclusters`.
    umap_key
        Key in ``adata.obsm`` for UMAP coordinates.
    split_by
        ``"celltype"`` — one panel per cell type.
        ``"disease"`` — one panel per disease.
    groups
        Which groups to plot (cell types or diseases depending on
        *split_by*).  ``None`` → all in the info table.
    ncols
        Number of columns in the panel grid.
    figsize_panel
        ``(width, height)`` of each individual panel.
    bg_dotsize
        Dot size for background cells.
    fg_dotsize
        Dot size for enriched cells.
    bg_alpha
        Alpha for cells outside the current group.
    group_bg_alpha
        Alpha for non-enriched cells inside the current group.
    group_bg_color
        Colour for non-enriched cells inside the current group.
    show
        Call ``plt.show()``.
    **umap_kwargs
        Forwarded to ``sc.pl.umap`` (except reserved keys).

    Returns
    -------
    Figure
    """
    info_key = f"{result_key}_info"
    if result_key not in adata.obs.columns:
        raise KeyError(
            f"Result column '{result_key}' not found in adata.obs. "
            f"Run detect_disease_enriched_subclusters first."
        )
    info = adata.uns.get(info_key)
    if info is None or info.empty:
        print(
            "No disease-enriched subclusters detected — nothing "
            "to plot."
        )
        return plt.figure()

    if umap_key not in adata.obsm:
        raise KeyError(
            f"UMAP key '{umap_key}' not found in adata.obsm."
        )

    if split_by not in ("celltype", "disease"):
        raise ValueError(
            f"split_by must be 'celltype' or 'disease', "
            f"got '{split_by}'"
        )

    _reserved = {"color", "ax", "show", "title", "size", "save"}
    for key in list(umap_kwargs.keys()):
        if key in _reserved:
            warnings.warn(
                f"'{key}' is set internally and will be ignored.",
                stacklevel=2,
            )
            umap_kwargs.pop(key)

    umap = adata.obsm[umap_key]
    result_vals = adata.obs[result_key].values.astype(str)

    info_col = "subset" if split_by == "celltype" else "disease"

    if groups is not None:
        groups_to_plot = [
            g for g in groups if g in info[info_col].values
        ]
    else:
        groups_to_plot = info[info_col].unique().tolist()

    if not groups_to_plot:
        print("No groups to plot.")
        return plt.figure()

    # Grid layout
    n_panels = len(groups_to_plot)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_panel[0] * ncols, figsize_panel[1] * nrows),
        squeeze=False,
    )

    for idx, group_name in enumerate(groups_to_plot):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        # Group mask
        if split_by == "celltype":
            if (
                celltype_key is not None
                and celltype_key in adata.obs.columns
            ):
                group_mask = (
                    adata.obs[celltype_key] == group_name
                ).values
            elif group_name == "all_cells":
                group_mask = np.ones(adata.n_obs, dtype=bool)
            else:
                group_mask = np.array(
                    [
                        str(v).startswith(f"{group_name}|")
                        or v == "background"
                        for v in result_vals
                    ]
                )
        else:
            if disease_key in adata.obs.columns:
                group_mask = (
                    adata.obs[disease_key] == group_name
                ).values
            else:
                group_mask = np.ones(adata.n_obs, dtype=bool)

        group_info = info[info[info_col] == group_name]
        fg_label_set = set(group_info["label"].values)
        fg_mask = np.array(
            [v in fg_label_set for v in result_vals], dtype=bool
        )

        if split_by == "celltype":
            fg_mask = fg_mask & group_mask

        bg_in_group = group_mask & ~fg_mask
        other_mask = ~group_mask & ~fg_mask

        if other_mask.any():
            ax.scatter(
                umap[other_mask, 0],
                umap[other_mask, 1],
                s=bg_dotsize,
                c="lightgrey",
                alpha=bg_alpha,
                rasterized=True,
                edgecolors="none",
            )

        if bg_in_group.any():
            ax.scatter(
                umap[bg_in_group, 0],
                umap[bg_in_group, 1],
                s=bg_dotsize,
                c=group_bg_color,
                alpha=group_bg_alpha,
                rasterized=True,
                edgecolors="none",
            )

        if fg_mask.any():
            sub_fg = adata[fg_mask].copy()
            present_labels = sorted(
                sub_fg.obs[result_key].unique().tolist(),
                key=_subcluster_sort_key,
            )
            sub_fg.obs[result_key] = pd.Categorical(
                sub_fg.obs[result_key].astype(str),
                categories=present_labels,
                ordered=True,
            )
            sc.pl.umap(
                sub_fg,
                color=result_key,
                title=group_name,
                ax=ax,
                show=False,
                save=False,
                size=fg_dotsize,
                **umap_kwargs,
            )
        else:
            ax.set_title(
                f"{group_name}\n(no enriched subclusters)",
                fontsize=10,
            )

        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)

    # Turn off unused axes
    for idx in range(n_panels, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig