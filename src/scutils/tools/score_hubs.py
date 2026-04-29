"""Score-hub detection utilities for single-cell RNA-seq data.

Provides two complementary approaches to identify cell types that contain
localised subpopulations with a high gene-set score:

* :func:`find_high_score_subclusters` – clustering-based: splits each cell
  type into sub-clusters and flags those whose mean score exceeds a threshold.
* :func:`find_score_hubs` – score-driven: smooths the score over the
  precomputed kNN graph and detects per-cell-type concentration via a
  percentile ratio or a Gaussian mixture model.

Diagnostic plots are provided by :func:`plot_subcluster_score_diagnostics`
and :func:`plot_score_hub_diagnostics`.
"""

from __future__ import annotations

import re
from typing import Dict, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

try:
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_threshold(
    score_threshold: Union[float, str],
    scores: np.ndarray,
) -> float:
    """Parse *score_threshold* into an absolute float cutoff.

    Args:
        score_threshold: Either a plain ``float`` (used as-is) or a string
            of the form ``"pXX"`` (e.g. ``"p75"``) interpreted as the
            *XX*-th percentile of *scores*.
        scores: 1-D array of score values used to resolve percentile
            thresholds.

    Returns:
        Absolute threshold value.

    Raises:
        ValueError: If *score_threshold* is a string but does not match the
            pattern ``r'^p(\\d+\\.?\\d*)$'``.
    """
    if isinstance(score_threshold, (int, float)):
        return float(score_threshold)
    m = re.match(r"^p(\d+\.?\d*)$", str(score_threshold))
    if m is None:
        raise ValueError(
            f"score_threshold string must match 'pXX' (e.g. 'p75'), "
            f"got '{score_threshold!r}'."
        )
    return float(np.percentile(scores, float(m.group(1))))


def _parse_vbound(
    bound: Union[str, float],
    values: np.ndarray,
) -> float:
    """Parse a colour-scale bound (absolute float or percentile string).

    Args:
        bound: Either a plain ``float`` or a string of the form ``"pXX"``
            (e.g. ``"p95"``).
        values: 1-D array used to resolve percentile bounds.

    Returns:
        Absolute bound value.

    Raises:
        ValueError: If *bound* is a string but does not match ``'pXX'``.
    """
    if isinstance(bound, (int, float)):
        return float(bound)
    m = re.match(r"^p(\d+\.?\d*)$", str(bound))
    if m is None:
        raise ValueError(
            f"vmin/vmax string must match 'pXX' (e.g. 'p95'), "
            f"got '{bound!r}'."
        )
    return float(np.percentile(values, float(m.group(1))))


def _resolve_connectivities_key(
    adata: AnnData,
    neighbors_key: str,
) -> str:
    """Return the ``obsp`` key that holds the kNN connectivities matrix.

    Follows the Scanpy convention:
    ``adata.uns[neighbors_key]['connectivities_key']``, falling back to
    ``"connectivities"`` if the key is absent.

    Args:
        adata: Annotated data matrix.
        neighbors_key: Key in ``adata.uns`` for the neighbors dict (usually
            ``"neighbors"``).

    Returns:
        The ``obsp`` key string.

    Raises:
        KeyError: If the resolved key is not present in ``adata.obsp``.
    """
    conn_key = adata.uns.get(neighbors_key, {}).get(
        "connectivities_key", "connectivities"
    )
    if conn_key not in adata.obsp:
        raise KeyError(
            f"Connectivities matrix '{conn_key}' not found in adata.obsp. "
            f"Run sc.pp.neighbors() first, or set neighbors_key correctly."
        )
    return conn_key


def _style_ax(ax: plt.Axes, basis: str) -> None:
    """Apply minimal spine-free styling to an embedding axis.

    Args:
        ax: Matplotlib axes to style.
        basis: ``adata.obsm`` key (e.g. ``"X_umap"``); used to label axes.
    """
    ax.tick_params(labelsize=7)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    label_base = basis.replace("X_", "")
    ax.set_xlabel(f"{label_base}1", fontsize=8)
    ax.set_ylabel(f"{label_base}2", fontsize=8)


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def find_high_score_subclusters(
    adata: AnnData,
    score_key: str,
    cell_type_col: str,
    resolution: Union[float, Dict[str, float]] = 0.3,
    n_clusters: Optional[Union[int, Dict[str, int]]] = None,
    score_threshold: Union[float, str] = "p75",
    recompute_neighbors: bool = False,
    use_rep: str = "X_pca",
    neighbors_key: str = "neighbors",
    key_added: Optional[str] = None,
    method: Literal["scanpy", "rapids"] = "scanpy",
) -> pd.DataFrame:
    """Sub-cluster each cell type and flag sub-clusters with a high gene-set score.

    For every category in *cell_type_col*, cells are partitioned into
    sub-clusters using either Leiden community detection (*resolution* mode)
    or agglomerative clustering (*n_clusters* mode).  Each sub-cluster is
    then scored by its mean *score_key* value and flagged as a hub when that
    mean exceeds *score_threshold*.

    Results are written back to *adata* in place:

    * ``adata.obs[key_added]`` – composite sub-cluster label
      (e.g. ``"T cell_0"``, ``"T cell_1"``).
    * ``adata.obs[f"{key_added}_is_hub"]`` – boolean, ``True`` for cells
      that belong to a hub sub-cluster.

    Args:
        adata: Annotated data matrix.
        score_key: Column in ``adata.obs`` containing the gene-set score.
        cell_type_col: Column in ``adata.obs`` containing cell-type labels.
        resolution: Leiden resolution used when *n_clusters* is ``None``.
            A ``float`` applies the same resolution to all cell types; a
            ``dict`` maps cell-type label → resolution for per-type control.
            Defaults to ``0.3``.
        n_clusters: Target number of sub-clusters.  When provided,
            agglomerative clustering is used instead of Leiden (requires
            *scikit-learn*).  An ``int`` applies to all cell types; a
            ``dict`` maps cell-type label → k.  Takes precedence over
            *resolution* when set.  Defaults to ``None``.
        score_threshold: Cutoff for flagging a sub-cluster as a hub.  A
            ``float`` is used as an absolute threshold; a string of the
            form ``"pXX"`` (e.g. ``"p75"``) is resolved to the *XX*-th
            percentile of *score_key* across **all** cells.
            Defaults to ``"p75"``.
        recompute_neighbors: When ``True``, run ``sc.pp.neighbors`` on each
            cell-type subset before Leiden clustering (uses *use_rep*).
            When ``False``, the precomputed neighbor graph is sliced to the
            subset.  Note that slicing removes inter-cell-type edges, which
            may produce disconnected components; this is generally acceptable
            for Leiden.  Only relevant in resolution mode.
            Defaults to ``False``.
        use_rep: Representation in ``adata.obsm`` used when
            *recompute_neighbors* is ``True`` or when *n_clusters* is set.
            Defaults to ``"X_pca"``.
        neighbors_key: Key in ``adata.uns`` for the precomputed neighbor
            graph used when *recompute_neighbors* is ``False``.
            Defaults to ``"neighbors"``.
        key_added: Base name for the obs columns written by this function.
            Defaults to ``"{score_key}_subcluster"``.
        method: Backend to use for Leiden clustering and neighbor
            computation.  ``"scanpy"`` calls ``sc.tl.leiden`` /
            ``sc.pp.neighbors``; ``"rapids"`` calls ``rsc.tl.leiden`` /
            ``rsc.pp.neighbors``.  Defaults to ``"scanpy"``.

    Returns:
        Summary ``DataFrame`` with one row per sub-cluster and columns:
        ``cell_type``, ``subcluster_id``, ``n_cells``, ``mean_score``,
        ``median_score``, ``fraction_above_threshold``, ``is_hub``.

    Raises:
        ValueError: If *score_key* or *cell_type_col* is not found in
            ``adata.obs``.
        ValueError: If *method* is not ``"scanpy"`` or ``"rapids"``.
        ValueError: If *use_rep* is not found in ``adata.obsm`` when needed.
        ImportError: If *n_clusters* is provided and scikit-learn is not
            installed.

    Example:
        >>> summary = find_high_score_subclusters(
        ...     adata,
        ...     score_key="my_score",
        ...     cell_type_col="cell_type",
        ...     resolution=0.5,
        ...     score_threshold="p80",
        ... )
        >>> summary[summary.is_hub]
    """
    if method == "scanpy":
        import scanpy as sc

        pp, tl = sc.pp, sc.tl
    elif method == "rapids":
        import rapids_singlecell as rsc

        pp, tl = rsc.pp, rsc.tl
    else:
        raise ValueError(
            f"method must be 'scanpy' or 'rapids', got '{method!r}'"
        )
    # igraph kwargs are scanpy-specific; rapids has its own GPU implementation.
    leiden_kwargs: dict = (
        {"flavor": "igraph", "n_iterations": 2, "directed": False}
        if method == "scanpy"
        else {}
    )

    if score_key not in adata.obs.columns:
        raise ValueError(
            f"score_key '{score_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if cell_type_col not in adata.obs.columns:
        raise ValueError(
            f"cell_type_col '{cell_type_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if n_clusters is not None and not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for n_clusters mode. "
            "Install it with: pip install scikit-learn"
        )

    if key_added is None:
        key_added = f"{score_key}_subcluster"

    scores = adata.obs[score_key].values.astype(float)
    threshold = _parse_threshold(score_threshold, scores)

    cell_types = (
        adata.obs[cell_type_col].cat.categories
        if hasattr(adata.obs[cell_type_col], "cat")
        else adata.obs[cell_type_col].unique()
    )

    # Initialise output columns.
    subcluster_labels = np.full(adata.n_obs, "", dtype=object)
    is_hub_arr = np.zeros(adata.n_obs, dtype=bool)
    records = []

    for ct in cell_types:
        mask = (adata.obs[cell_type_col] == ct).values
        if mask.sum() == 0:
            continue

        ct_str = str(ct)
        idx_in_adata = np.where(mask)[0]
        ct_scores = scores[mask]

        # --- Determine sub-cluster labels --------------------------------
        if n_clusters is not None:
            # Agglomerative clustering mode.
            k = (
                n_clusters
                if isinstance(n_clusters, int)
                else n_clusters.get(ct_str, 2)
            )
            if use_rep not in adata.obsm:
                raise ValueError(
                    f"use_rep '{use_rep}' not found in adata.obsm."
                )
            from sklearn.cluster import AgglomerativeClustering

            coords = adata.obsm[use_rep][mask]
            sub_labels = AgglomerativeClustering(n_clusters=k).fit_predict(coords)
        else:
            # Leiden resolution mode.
            res = (
                resolution
                if isinstance(resolution, (int, float))
                else resolution.get(ct_str, 0.3)
            )
            adata_sub = adata[mask].copy()
            if recompute_neighbors:
                if use_rep not in adata.obsm:
                    raise ValueError(
                        f"use_rep '{use_rep}' not found in adata.obsm."
                    )
                pp.neighbors(adata_sub, use_rep=use_rep)
                tl.leiden(
                    adata_sub,
                    resolution=res,
                    key_added="_leiden_tmp",
                    **leiden_kwargs,
                )
            else:
                # Validate that the connectivity matrix exists before calling leiden.
                _resolve_connectivities_key(adata, neighbors_key)
                tl.leiden(
                    adata_sub,
                    resolution=res,
                    key_added="_leiden_tmp",
                    neighbors_key=neighbors_key,
                    **leiden_kwargs,
                )
            sub_labels = adata_sub.obs["_leiden_tmp"].astype(int).values

        # --- Score each sub-cluster --------------------------------------
        unique_subs = np.unique(sub_labels)
        for sub_id in unique_subs:
            sub_mask_local = sub_labels == sub_id
            sub_idx = idx_in_adata[sub_mask_local]
            sub_scores = ct_scores[sub_mask_local]

            composite = f"{ct_str}_{sub_id}"
            subcluster_labels[sub_idx] = composite

            mean_s = float(np.mean(sub_scores))
            median_s = float(np.median(sub_scores))
            frac = float((sub_scores > threshold).mean())
            hub = mean_s >= threshold

            if hub:
                is_hub_arr[sub_idx] = True

            records.append(
                {
                    "cell_type": ct_str,
                    "subcluster_id": composite,
                    "n_cells": int(sub_mask_local.sum()),
                    "mean_score": mean_s,
                    "median_score": median_s,
                    "fraction_above_threshold": frac,
                    "is_hub": hub,
                }
            )

    adata.obs[key_added] = pd.Categorical(subcluster_labels)
    adata.obs[f"{key_added}_is_hub"] = is_hub_arr

    return pd.DataFrame(records)


def find_score_hubs(
    adata: AnnData,
    score_key: str,
    cell_type_col: str,
    hub_percentile: float = 90.0,
    min_hub_ratio: float = 1.5,
    method: Literal["percentile", "gmm"] = "percentile",
    neighbors_key: str = "neighbors",
    key_added: Optional[str] = None,
) -> pd.DataFrame:
    """Detect cell types with localised high-score hubs via kNN smoothing.

    The gene-set score is first smoothed over the precomputed kNN
    connectivity graph (row-normalised sparse matrix multiply) to produce a
    per-cell *local score*.  For each cell type the local scores are then
    summarised into a *hub ratio* (ratio of the within-cell-type top
    percentile to the global top percentile) or analysed with a two-component
    Gaussian mixture model to identify a concentrated high-score subpopulation.

    The smoothed local score is written to ``adata.obs`` in place.

    Args:
        adata: Annotated data matrix with a precomputed kNN graph in
            ``adata.obsp``.
        score_key: Column in ``adata.obs`` containing the gene-set score.
        cell_type_col: Column in ``adata.obs`` containing cell-type labels.
        hub_percentile: Percentile used to characterise the high end of the
            score distribution.  Defaults to ``90.0``.
        min_hub_ratio: Minimum ``hub_ratio`` for a cell type to be flagged
            as a hub.  Only used with ``method="percentile"``.
            Defaults to ``1.5``.
        method: Hub-detection statistic.  ``"percentile"`` uses the ratio of
            the within-cell-type *hub_percentile* to the global
            *hub_percentile*.  ``"gmm"`` fits a two-component Gaussian
            mixture to the local scores within each cell type and flags cell
            types whose high-mean component's mean exceeds the global hub
            percentile.  Defaults to ``"percentile"``.
        neighbors_key: Key in ``adata.uns`` for the precomputed kNN graph.
            Defaults to ``"neighbors"``.
        key_added: Name of the ``adata.obs`` column for the smoothed local
            score.  Defaults to ``"{score_key}_local"``.

    Returns:
        Summary ``DataFrame`` with one row per cell type and columns:
        ``cell_type``, ``mean_score``, ``hub_score``, ``hub_ratio``,
        ``hub_fraction``, ``is_hub``.  Sorted by ``hub_ratio`` descending.

    Raises:
        ValueError: If *score_key* or *cell_type_col* is not found in
            ``adata.obs``.
        ValueError: If *method* is not ``"percentile"`` or ``"gmm"``.
        ImportError: If *method* is ``"gmm"`` and scikit-learn is not
            installed.
        KeyError: If no precomputed neighbors graph is found in
            ``adata.obsp``.

    Example:
        >>> hub_df = find_score_hubs(
        ...     adata,
        ...     score_key="my_score",
        ...     cell_type_col="cell_type",
        ...     hub_percentile=90,
        ...     min_hub_ratio=1.5,
        ... )
        >>> hub_df[hub_df.is_hub]
    """
    if score_key not in adata.obs.columns:
        raise ValueError(
            f"score_key '{score_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if cell_type_col not in adata.obs.columns:
        raise ValueError(
            f"cell_type_col '{cell_type_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if method not in ("percentile", "gmm"):
        raise ValueError(
            f"method must be 'percentile' or 'gmm', got '{method!r}'."
        )
    if method == "gmm" and not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for method='gmm'. "
            "Install it with: pip install scikit-learn"
        )

    if key_added is None:
        key_added = f"{score_key}_local"

    # ---- Smooth scores over the kNN connectivity graph -------------------
    conn_key = _resolve_connectivities_key(adata, neighbors_key)
    C = adata.obsp[conn_key]

    # Row-normalise: each cell's local score = weighted average of neighbours.
    row_sums = np.asarray(C.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = scipy.sparse.diags(1.0 / row_sums)
    C_norm = D_inv @ C

    raw_scores = adata.obs[score_key].values.astype(float)
    local_scores = np.asarray(C_norm @ raw_scores).ravel()

    adata.obs[key_added] = local_scores

    # ---- Global hub-percentile reference --------------------------------
    global_hub_score = float(np.percentile(local_scores, hub_percentile))
    # Guard against zero division; use a small epsilon to avoid inf ratios.
    global_hub_ref = global_hub_score if abs(global_hub_score) > 1e-12 else 1e-12

    # ---- Per-cell-type statistics ----------------------------------------
    cell_types = (
        adata.obs[cell_type_col].cat.categories
        if hasattr(adata.obs[cell_type_col], "cat")
        else adata.obs[cell_type_col].unique()
    )

    records = []
    for ct in cell_types:
        mask = (adata.obs[cell_type_col] == ct).values
        ct_local = local_scores[mask]
        ct_raw = raw_scores[mask]
        mean_s = float(np.mean(ct_raw))

        if method == "percentile":
            hub_score = float(np.percentile(ct_local, hub_percentile))
            hub_ratio = hub_score / global_hub_ref
            hub_fraction = float((ct_local > global_hub_score).mean())
            is_hub = hub_ratio >= min_hub_ratio

        else:  # gmm
            from sklearn.mixture import GaussianMixture

            if len(ct_local) < 4:
                # Too few cells to fit a GMM reliably.
                hub_score = float(np.percentile(ct_local, hub_percentile))
                hub_ratio = hub_score / global_hub_ref
                hub_fraction = 0.0
                is_hub = False
            else:
                gm = GaussianMixture(n_components=2, random_state=0)
                gm.fit(ct_local.reshape(-1, 1))
                means = gm.means_.ravel()
                weights = gm.weights_.ravel()
                high_idx = int(np.argmax(means))
                hub_score = float(means[high_idx])
                hub_ratio = hub_score / global_hub_ref
                hub_fraction = float(weights[high_idx])
                is_hub = hub_score >= global_hub_score and hub_ratio >= min_hub_ratio

        records.append(
            {
                "cell_type": str(ct),
                "mean_score": mean_s,
                "hub_score": hub_score,
                "hub_ratio": hub_ratio,
                "hub_fraction": hub_fraction,
                "is_hub": is_hub,
            }
        )

    summary = (
        pd.DataFrame(records)
        .sort_values("hub_ratio", ascending=False)
        .reset_index(drop=True)
    )
    return summary


# ---------------------------------------------------------------------------
# Diagnostic visualisation
# ---------------------------------------------------------------------------


def plot_subcluster_score_diagnostics(
    adata: AnnData,
    score_key: str,
    subcluster_key: str,
    basis: str = "X_umap",
    ncols: int = 2,
    figsize_per_panel: Tuple[float, float] = (4.5, 4.0),
    point_size: float = 3.0,
    vmin: Union[str, float] = "p5",
    vmax: Union[str, float] = "p95",
) -> plt.Figure:
    """Visualise sub-cluster assignments and scores for hub detection results.

    For every cell type that contains at least one hub sub-cluster (as
    determined by ``adata.obs[f"{subcluster_key}_is_hub"]``), two panels are
    drawn side-by-side:

    1. **Sub-cluster map** – cells of that type coloured by sub-cluster ID
       (``tab10``); hub sub-clusters are drawn with full opacity and larger
       points, non-hub sub-clusters are dimmed.
    2. **Score overlay** – same cells coloured by the continuous *score_key*
       value.

    Args:
        adata: Annotated data matrix with ``subcluster_key`` and
            ``f"{subcluster_key}_is_hub"`` columns in ``adata.obs``.
        score_key: Column in ``adata.obs`` containing the gene-set score
            (used for the second panel).
        subcluster_key: Base key written by
            :func:`find_high_score_subclusters` (the *key_added* argument
            passed to that function).
        basis: Key in ``adata.obsm`` with 2-D embedding coordinates.
            Defaults to ``"X_umap"``.
        ncols: Number of panel columns per row.  Normally ``2``.
            Defaults to ``2``.
        figsize_per_panel: ``(width, height)`` of each individual panel in
            inches.  Defaults to ``(4.5, 4.0)``.
        point_size: Scatter-plot point size.  Defaults to ``3.0``.
        vmin: Lower bound of the score colour scale.  Either a ``float`` or
            a percentile string (e.g. ``"p5"``).  Defaults to ``"p5"``.
        vmax: Upper bound of the score colour scale.  Either a ``float`` or
            a percentile string (e.g. ``"p95"``).  Defaults to ``"p95"``.

    Returns:
        The diagnostic matplotlib figure.

    Raises:
        ValueError: If required keys are missing from ``adata.obs`` or
            ``adata.obsm``.
        ValueError: If no hub sub-clusters are found.

    Example:
        >>> summary = find_high_score_subclusters(
        ...     adata, score_key="my_score", cell_type_col="cell_type"
        ... )
        >>> fig = plot_subcluster_score_diagnostics(
        ...     adata,
        ...     score_key="my_score",
        ...     subcluster_key="my_score_subcluster",
        ... )
        >>> fig.savefig("subcluster_diagnostics.png", dpi=150)
    """
    is_hub_col = f"{subcluster_key}_is_hub"
    for col in (score_key, subcluster_key, is_hub_col):
        if col not in adata.obs.columns:
            raise ValueError(
                f"Column '{col}' not found in adata.obs. "
                "Run find_high_score_subclusters() first."
            )
    if basis not in adata.obsm:
        raise ValueError(f"basis '{basis}' not found in adata.obsm.")

    all_scores = adata.obs[score_key].values.astype(float)
    _vmin = _parse_vbound(vmin, all_scores)
    _vmax = _parse_vbound(vmax, all_scores)

    # Cell types that have at least one hub sub-cluster.
    hub_cell_types = (
        adata.obs.loc[adata.obs[is_hub_col], subcluster_key]
        .astype(str)
        .str.rsplit("_", n=1)
        .str[0]
        .unique()
        .tolist()
    )
    if not hub_cell_types:
        raise ValueError(
            "No hub sub-clusters found. "
            "Check that find_high_score_subclusters() found any hubs."
        )

    coords = adata.obsm[basis]
    all_sub_labels = adata.obs[subcluster_key].astype(str).values
    is_hub_all = adata.obs[is_hub_col].values
    cmap_tab10 = matplotlib.colormaps["tab10"]

    n_rows = len(hub_cell_types)
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for row, ct in enumerate(hub_cell_types):
        ct_mask = np.array(
            [s.rsplit("_", 1)[0] == str(ct) for s in all_sub_labels]
        )
        ct_coords = coords[ct_mask]
        ct_scores = all_scores[ct_mask]
        ct_subs = all_sub_labels[ct_mask]
        ct_is_hub = is_hub_all[ct_mask]

        # Preserve encounter order so colours are consistent.
        unique_subs = list(dict.fromkeys(ct_subs))

        # ---- Panel 0: sub-cluster colouring ----------------------------
        ax0 = axs[row, 0]
        ax0.scatter(
            coords[:, 0], coords[:, 1],
            c="#e0e0e0", s=point_size * 0.3, rasterized=True, zorder=1,
        )
        for i, sub in enumerate(unique_subs):
            sub_local = ct_subs == sub
            is_hub_sub = ct_is_hub[sub_local].any()
            alpha = 1.0 if is_hub_sub else 0.45
            size = point_size * 1.5 if is_hub_sub else point_size
            label = f"{sub} ★" if is_hub_sub else sub
            ax0.scatter(
                ct_coords[sub_local, 0], ct_coords[sub_local, 1],
                c=[cmap_tab10(i % 10)], s=size, alpha=alpha,
                label=label, rasterized=True, zorder=2,
            )
        ax0.legend(fontsize=6, markerscale=2, frameon=True, loc="best")
        ax0.set_title(f"{ct} – sub-clusters  (n={ct_mask.sum():,})", fontsize=10)
        _style_ax(ax0, basis)

        # ---- Panel 1: score colouring ----------------------------------
        ax1 = axs[row, 1]
        ax1.scatter(
            coords[:, 0], coords[:, 1],
            c="#e0e0e0", s=point_size * 0.3, rasterized=True, zorder=1,
        )
        sc_plot = ax1.scatter(
            ct_coords[:, 0], ct_coords[:, 1],
            c=ct_scores, cmap="viridis",
            vmin=_vmin, vmax=_vmax,
            s=point_size, rasterized=True, zorder=2,
        )
        plt.colorbar(sc_plot, ax=ax1, shrink=0.7, label=score_key)
        ax1.set_title(f"{ct} – {score_key}", fontsize=10)
        _style_ax(ax1, basis)

    # Row labels on left margin (mirrors plot_spatial_split_diagnostics).
    left_margin = 0.06
    plt.tight_layout(rect=[left_margin, 0, 1, 1])
    fig.subplots_adjust(hspace=0.55)
    for row, ct in enumerate(hub_cell_types):
        y = 1.0 - (row + 0.5) / n_rows
        fig.text(
            left_margin - 0.01, y,
            str(ct),
            ha="right", va="center",
            fontsize=14, fontweight="bold",
        )
    return fig


def plot_score_hub_diagnostics(
    adata: AnnData,
    score_key: str,
    cell_type_col: str,
    hub_df: pd.DataFrame,
    local_score_key: Optional[str] = None,
    basis: str = "X_umap",
    ncols: int = 2,
    figsize_per_panel: Tuple[float, float] = (4.5, 4.0),
    point_size: float = 3.0,
    vmin: Union[str, float] = "p5",
    vmax: Union[str, float] = "p95",
) -> plt.Figure:
    """Visualise raw vs. smoothed scores for cell types flagged as hubs.

    For every cell type where ``hub_df.is_hub == True``, two panels are
    drawn side-by-side:

    1. **Raw score** – cells of that type coloured by *score_key*.
    2. **Smoothed (local) score** – same cells coloured by *local_score_key*
       (the column written to ``adata.obs`` by :func:`find_score_hubs`).

    Args:
        adata: Annotated data matrix.
        score_key: Column in ``adata.obs`` containing the raw gene-set score.
        cell_type_col: Column in ``adata.obs`` containing cell-type labels.
        hub_df: DataFrame returned by :func:`find_score_hubs`.  Must contain
            columns ``cell_type`` and ``is_hub``.
        local_score_key: Column in ``adata.obs`` containing the smoothed
            local score.  Defaults to ``"{score_key}_local"``.
        basis: Key in ``adata.obsm`` with 2-D embedding coordinates.
            Defaults to ``"X_umap"``.
        ncols: Number of panel columns per row.  Normally ``2``.
            Defaults to ``2``.
        figsize_per_panel: ``(width, height)`` of each individual panel in
            inches.  Defaults to ``(4.5, 4.0)``.
        point_size: Scatter-plot point size.  Defaults to ``3.0``.
        vmin: Lower bound of the colour scale.  Either a ``float`` or a
            percentile string (e.g. ``"p5"``).  Defaults to ``"p5"``.
        vmax: Upper bound of the colour scale.  Either a ``float`` or a
            percentile string (e.g. ``"p95"``).  Defaults to ``"p95"``.

    Returns:
        The diagnostic matplotlib figure.

    Raises:
        ValueError: If required keys are missing from ``adata.obs`` or
            ``adata.obsm``.
        ValueError: If *hub_df* contains no rows with ``is_hub == True``.
        ValueError: If *hub_df* is missing required columns.

    Example:
        >>> hub_df = find_score_hubs(
        ...     adata, score_key="my_score", cell_type_col="cell_type"
        ... )
        >>> fig = plot_score_hub_diagnostics(
        ...     adata,
        ...     score_key="my_score",
        ...     cell_type_col="cell_type",
        ...     hub_df=hub_df,
        ... )
        >>> fig.savefig("score_hub_diagnostics.png", dpi=150)
    """
    if local_score_key is None:
        local_score_key = f"{score_key}_local"

    for col in (score_key, local_score_key):
        if col not in adata.obs.columns:
            raise ValueError(
                f"Column '{col}' not found in adata.obs. "
                "Run find_score_hubs() first."
            )
    if cell_type_col not in adata.obs.columns:
        raise ValueError(
            f"cell_type_col '{cell_type_col}' not found in adata.obs."
        )
    if basis not in adata.obsm:
        raise ValueError(f"basis '{basis}' not found in adata.obsm.")
    for required in ("cell_type", "is_hub"):
        if required not in hub_df.columns:
            raise ValueError(
                f"hub_df must contain a '{required}' column."
            )

    hub_types = hub_df.loc[hub_df["is_hub"], "cell_type"].tolist()
    if not hub_types:
        raise ValueError(
            "No hub cell types found in hub_df. "
            "Check that find_score_hubs() produced any hubs."
        )

    coords = adata.obsm[basis]
    raw_scores = adata.obs[score_key].values.astype(float)
    local_scores = adata.obs[local_score_key].values.astype(float)

    _vmin_raw = _parse_vbound(vmin, raw_scores)
    _vmax_raw = _parse_vbound(vmax, raw_scores)
    _vmin_loc = _parse_vbound(vmin, local_scores)
    _vmax_loc = _parse_vbound(vmax, local_scores)

    n_rows = len(hub_types)
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for row, ct in enumerate(hub_types):
        mask = (adata.obs[cell_type_col].astype(str) == str(ct)).values
        ct_coords = coords[mask]
        ct_raw = raw_scores[mask]
        ct_local = local_scores[mask]

        # ---- Panel 0: raw score ----------------------------------------
        ax0 = axs[row, 0]
        ax0.scatter(
            coords[:, 0], coords[:, 1],
            c="#e0e0e0", s=point_size * 0.3, rasterized=True, zorder=1,
        )
        sc0 = ax0.scatter(
            ct_coords[:, 0], ct_coords[:, 1],
            c=ct_raw, cmap="viridis",
            vmin=_vmin_raw, vmax=_vmax_raw,
            s=point_size, rasterized=True, zorder=2,
        )
        plt.colorbar(sc0, ax=ax0, shrink=0.7, label=score_key)
        ax0.set_title(f"{ct} – raw score", fontsize=10)
        _style_ax(ax0, basis)

        # ---- Panel 1: local (smoothed) score ---------------------------
        ax1 = axs[row, 1]
        ax1.scatter(
            coords[:, 0], coords[:, 1],
            c="#e0e0e0", s=point_size * 0.3, rasterized=True, zorder=1,
        )
        sc1 = ax1.scatter(
            ct_coords[:, 0], ct_coords[:, 1],
            c=ct_local, cmap="viridis",
            vmin=_vmin_loc, vmax=_vmax_loc,
            s=point_size, rasterized=True, zorder=2,
        )
        plt.colorbar(sc1, ax=ax1, shrink=0.7, label=local_score_key)
        ax1.set_title(f"{ct} – local score", fontsize=10)
        _style_ax(ax1, basis)

    # Row labels on left margin.
    left_margin = 0.06
    plt.tight_layout(rect=[left_margin, 0, 1, 1])
    fig.subplots_adjust(hspace=0.55)
    for row, ct in enumerate(hub_types):
        y = 1.0 - (row + 0.5) / n_rows
        fig.text(
            left_margin - 0.01, y,
            str(ct),
            ha="right", va="center",
            fontsize=14, fontweight="bold",
        )
    return fig
