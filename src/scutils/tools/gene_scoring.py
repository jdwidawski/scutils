"""Gene-set scoring utilities for single-cell RNA-seq data."""

from __future__ import annotations

from typing import List, Optional
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

try:
    import pynndescent
    from hotspot import modules
    from hotspot.knn import compute_weights
    from hotspot.hotspot import Hotspot

    _HOTSPOT_AVAILABLE = True
except ImportError:
    _HOTSPOT_AVAILABLE = False

try:
    import decoupler as dc

    _DECOUPLER_AVAILABLE = True
except ImportError:
    _DECOUPLER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_genes(
    adata: AnnData,
    genes: List[str],
    gene_symbols: Optional[str] = None,
) -> List[str]:
    """Map gene names to ``adata.var_names``.

    When ``gene_symbols`` is ``None``, returns ``genes`` as-is (assumed to
    be entries in ``adata.var_names``).  When set, finds rows whose
    ``adata.var[gene_symbols]`` value is in ``genes`` and returns the
    corresponding ``adata.var_names`` entries.

    Args:
        adata: Annotated data matrix.
        genes: List of gene identifiers to resolve.
        gene_symbols: Column in ``adata.var`` containing gene symbols.
            When ``None``, ``genes`` is returned unchanged.

    Returns:
        List of ``adata.var_names`` entries that correspond to ``genes``.
    """
    if gene_symbols is None:
        return genes
    mask = adata.var[gene_symbols].isin(genes)
    return adata.var_names[mask].tolist()


def _get_obsm_key(adata: AnnData, *candidates: str) -> str:
    """Return the first obsm key from *candidates* that exists in *adata*.

    Args:
        adata: Annotated data matrix.
        *candidates: obsm key names to try, in order of preference.

    Returns:
        The first key from *candidates* found in ``adata.obsm``.

    Raises:
        KeyError: If none of *candidates* are present in ``adata.obsm``.
    """
    for key in candidates:
        if key in adata.obsm:
            return key
    raise KeyError(
        f"None of the expected keys {candidates} found in adata.obsm. "
        f"Available keys: {list(adata.obsm.keys())}"
    )


def _warn_missing_genes(
    genes: List[str],
    var_names: pd.Index,
    min_n: int,
    func_name: str = "scoring",
) -> List[str]:
    """Warn about genes absent from ``adata.var_names`` and return those present.

    Args:
        genes: Resolved gene names (already mapped to var_names space).
        var_names: ``adata.var_names`` index.
        min_n: Minimum number of genes required for scoring.
        func_name: Name of the calling function, used in messages.

    Returns:
        Subset of ``genes`` present in ``var_names``.

    Raises:
        ValueError: If fewer than ``min_n`` genes are found in
            ``var_names``.
    """
    var_set = set(var_names)
    missing = [g for g in genes if g not in var_set]
    found = [g for g in genes if g in var_set]
    if missing:
        warnings.warn(
            f"{func_name}: {len(missing)} of {len(genes)} input gene(s) were not "
            f"found in adata.var_names and will be ignored.\n"
            f"Missing genes: {missing}",
            UserWarning,
            stacklevel=3,
        )
    if len(found) < min_n:
        raise ValueError(
            f"{func_name}: only {len(found)} input gene(s) are present in "
            f"adata.var_names, but min_n={min_n} is required. "
            f"Provide more genes that exist in the data, or reduce min_n.\n"
            f"Present: {found}\n"
            f"Missing: {missing}"
        )
    return found


# ---------------------------------------------------------------------------
# Hotspot scoring
# ---------------------------------------------------------------------------


def compute_hotspot_scores(
    adata: AnnData,
    genes: List[str],
    score_name: str = "hotspot_score",
    layer: Optional[str] = None,
    n_neighbors: int = 30,
    neighborhood_factor: int = 3,
    gene_symbols: Optional[str] = None,
    use_rep: Optional[str] = "X_scVI",
    model: str = "danb",
    copy: bool = False,
) -> Optional[AnnData]:
    """Compute Hotspot gene-module scores and store them in ``adata.obs``.

    Uses the Hotspot algorithm to score a set of genes based on their
    spatial auto-correlation in a neighbourhood graph.  The graph is
    constructed from a low-dimensional representation (``use_rep``) or
    directly from ``adata.X`` / a layer when ``use_rep`` is ``None``.

    Args:
        adata: Annotated data matrix.
        genes: List of gene names to include in the module score.
        score_name: Key added to ``adata.obs`` for the computed score.
            Defaults to ``"hotspot_score"``.
        layer: Layer to use for expression counts and for building the
            neighbour graph when ``use_rep`` is ``None``.  Defaults to
            ``None`` (uses ``adata.X``).
        n_neighbors: Number of neighbours for the neighbourhood graph.
            Defaults to ``30``.
        neighborhood_factor: Bandwidth factor used when converting
            distances to weights.  Defaults to ``3``.
        gene_symbols: Column in ``adata.var`` that contains gene symbols.
            When provided, ``genes`` is matched against this column instead
            of ``adata.var_names``.  Defaults to ``None``.
        use_rep: Key in ``adata.obsm`` used to build the neighbourhood
            graph.  Set to ``None`` to build from ``adata.X`` (or
            ``layer``).  Defaults to ``"X_scVI"``.
        model: Count model used by Hotspot.  One of ``"danb"`` (default),
            ``"bernoulli"``, ``"normal"``, or ``"none"``.
        copy: If ``True``, operate on a copy of ``adata`` and return it.
            If ``False`` (default), modify ``adata`` in place and return
            ``None``.

    Returns:
        A modified copy of ``adata`` when ``copy=True``, otherwise
        ``None``.

    Raises:
        ImportError: If ``hotspot-sc`` or ``pynndescent`` are not
            installed.

    Example:
        >>> scutils.tl.compute_hotspot_scores(
        ...     adata,
        ...     genes=["SELL", "CCR7", "TCF7"],
        ...     score_name="naive_T_score",
        ...     use_rep="X_pca",
        ... )
        >>> adata.obs["naive_T_score"]
    """
    if not _HOTSPOT_AVAILABLE:
        raise ImportError(
            "hotspot-sc and pynndescent are required for compute_hotspot_scores. "
            "Install them with: pip install 'scutils[scoring]'"
        )

    adata = adata.copy() if copy else adata

    if use_rep is None:
        if layer is None:
            index = pynndescent.NNDescent(adata.X, n_neighbors=n_neighbors + 1)
        else:
            index = pynndescent.NNDescent(
                adata.layers[layer], n_neighbors=n_neighbors + 1
            )
    else:
        index = pynndescent.NNDescent(
            adata.obsm[use_rep], n_neighbors=n_neighbors + 1
        )

    ind, dist = index.neighbor_graph
    ind, dist = ind[:, 1:], dist[:, 1:]

    neighbors = pd.DataFrame(ind, index=list(range(adata.n_obs)))
    weights = compute_weights(dist, neighborhood_factor=neighborhood_factor)
    weights = pd.DataFrame(
        weights,
        index=neighbors.index,
        columns=neighbors.columns,
    )

    if layer is None:
        if scipy.sparse.issparse(adata.X):
            umi_counts = adata.X.sum(axis=1).A1
        else:
            umi_counts = adata.X.sum(axis=1)
    else:
        if scipy.sparse.issparse(adata.layers[layer]):
            umi_counts = adata.layers[layer].sum(axis=1).A1
        else:
            umi_counts = adata.layers[layer].sum(axis=1)

    if gene_symbols is None:
        counts_dense = Hotspot._counts_from_anndata(
            adata[:, adata.var_names.isin(genes)],
            layer,
            dense=True,
        )
    else:
        counts_dense = Hotspot._counts_from_anndata(
            adata[:, adata.var[gene_symbols].isin(genes)],
            layer,
            dense=True,
        )

    scores = modules.compute_scores(
        counts_dense,
        model,
        umi_counts,
        neighbors.values,
        weights.values,
    )

    adata.obs[score_name] = scores
    return adata if copy else None


# ---------------------------------------------------------------------------
# ULM scoring (private runner + public gene-list interface)
# ---------------------------------------------------------------------------


def _run_ulm_network(
    adata: AnnData,
    net: pd.DataFrame,
    source_name: str,
    score_name: str = "ulm_score",
    layer: Optional[str] = None,
    min_n: int = 5,
    batch_size: int = 10000,
    return_pvals: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    """(Private) Run ULM on *net* and store the score for *source_name*.

    Handles both the new ``decoupler.mt.ulm`` API and the legacy
    ``decoupler.run_ulm`` API, and resolves obsm key naming differences
    between decoupler versions.

    Args:
        adata: Annotated data matrix.
        net: Prior-knowledge network with ``source``, ``target``, and
            ``weight`` columns.
        source_name: Name of the source in ``net`` whose score is
            extracted and stored in ``adata.obs``.
        score_name: Key added to ``adata.obs``.  Defaults to
            ``"ulm_score"``.
        layer: Layer to use for gene expression.  Defaults to ``None``.
        min_n: Minimum target overlap required to score a source.
            Defaults to ``5``.
        batch_size: Sources processed per batch (legacy API only).
            Defaults to ``10000``.
        return_pvals: If ``True``, also store the p-value as
            ``adata.obs["{score_name}_pval"]``.
        copy: Return a modified copy when ``True``.

    Returns:
        Modified copy when ``copy=True``, otherwise ``None``.
    """
    adata = adata.copy() if copy else adata

    common_kwargs: dict = dict(verbose=False)
    if layer is not None:
        common_kwargs["layer"] = layer

    if hasattr(dc, "mt") and hasattr(dc.mt, "ulm"):
        # New decoupler API (>= 2.x)
        dc.mt.ulm(adata, net, tmin=min_n, **common_kwargs)
    else:
        # Legacy decoupler API
        dc.run_ulm(
            adata,
            net=net,
            source="source",
            target="target",
            weight="weight",
            min_n=min_n,
            batch_size=batch_size,
            **common_kwargs,
        )

    est_key = _get_obsm_key(adata, "ulm_estimate", "score_ulm")
    adata.obs[score_name] = adata.obsm[est_key][source_name].values
    del adata.obsm[est_key]

    if return_pvals:
        pval_key = _get_obsm_key(adata, "ulm_pvals", "pval_ulm")
        adata.obs[f"{score_name}_pval"] = adata.obsm[pval_key][source_name].values
        del adata.obsm[pval_key]
    else:
        for pval_key in ("ulm_pvals", "pval_ulm"):
            if pval_key in adata.obsm:
                del adata.obsm[pval_key]

    return adata if copy else None


def compute_ulm_scores(
    adata: AnnData,
    genes: List[str],
    set_name: str = "gene_set",
    score_name: str = "ulm_score",
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    min_n: int = 5,
    batch_size: int = 10000,
    return_pvals: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    """Compute a ULM enrichment score for a gene list using decoupler.

    Constructs a one-source, uniform-weight network from ``genes`` and
    runs Univariate Linear Model (ULM).  ULM fits a linear model per cell
    using the binary gene-membership vector as predictor, yielding a
    t-statistic (activity estimate) and an optional p-value.

    Args:
        adata: Annotated data matrix.
        genes: Gene identifiers that define the gene set.  Interpreted as
            ``adata.var_names`` entries unless ``gene_symbols`` is set.
        set_name: Label assigned to the gene set; becomes the source name
            in the internal network.  Defaults to ``"gene_set"``.
        score_name: Key added to ``adata.obs`` for the ULM activity
            estimate.  Defaults to ``"ulm_score"``.
        layer: Layer to use for gene expression.  Defaults to ``None``
            (uses ``adata.X``).
        gene_symbols: Column in ``adata.var`` that contains gene symbols.
            When provided, ``genes`` is matched against this column and
            mapped to ``adata.var_names``.  Defaults to ``None``.
        min_n: Minimum number of genes from the list that must be present
            in ``adata.var_names`` for scoring to proceed.  Defaults to
            ``5``.
        batch_size: Number of sources processed per batch (legacy API
            only).  Defaults to ``10000``.
        return_pvals: If ``True``, also store the p-value as
            ``adata.obs["{score_name}_pval"]``.  Defaults to ``False``.
        copy: If ``True``, operate on a copy of ``adata`` and return it.
            If ``False`` (default), modify ``adata`` in place and return
            ``None``.

    Returns:
        A modified copy of ``adata`` when ``copy=True``, otherwise
        ``None``.

    Raises:
        ImportError: If ``decoupler`` is not installed.
        ValueError: If fewer than ``min_n`` input genes are found in
            ``adata.var_names``.

    Example:
        >>> scutils.tl.compute_ulm_scores(
        ...     adata,
        ...     genes=["SELL", "CCR7", "TCF7", "LEF1", "KLF2"],
        ...     score_name="naive_T_ulm",
        ...     return_pvals=True,
        ... )
        >>> adata.obs["naive_T_ulm"]       # ULM activity estimate
        >>> adata.obs["naive_T_ulm_pval"]  # associated p-value
    """
    if not _DECOUPLER_AVAILABLE:
        raise ImportError(
            "decoupler is required for compute_ulm_scores. "
            "Install it with: pip install 'scutils[decoupler]'"
        )

    resolved = _resolve_genes(adata, genes, gene_symbols)
    found = _warn_missing_genes(
        resolved, adata.var_names, min_n, func_name="compute_ulm_scores"
    )
    net = pd.DataFrame(
        {
            "source": set_name,
            "target": found,
            "weight": 1.0,
        }
    )
    return _run_ulm_network(
        adata,
        net=net,
        source_name=set_name,
        score_name=score_name,
        layer=layer,
        min_n=min_n,
        batch_size=batch_size,
        return_pvals=return_pvals,
        copy=copy,
    )


# ---------------------------------------------------------------------------
# AUCell scoring
# ---------------------------------------------------------------------------


def compute_aucell_scores(
    adata: AnnData,
    genes: List[str],
    set_name: str = "gene_set",
    score_name: str = "aucell_score",
    layer: Optional[str] = None,
    gene_symbols: Optional[str] = None,
    min_n: int = 5,
    n_up: Optional[int] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """Compute an AUCell enrichment score for a gene list using decoupler.

    Builds a one-source network from ``genes``, runs the AUCell algorithm,
    and stores the resulting enrichment score as a single column in
    ``adata.obs``.  AUCell is rank-based and requires no gene weights,
    making it the natural choice for plain gene lists.

    Args:
        adata: Annotated data matrix.  Expression values should be
            normalised (e.g. library-size normalised + log1p); raw integer
            counts are not recommended.
        genes: Gene identifiers that define the gene set.  Interpreted as
            ``adata.var_names`` entries unless ``gene_symbols`` is set.
        set_name: Label assigned to the gene set; used as the source name
            in the internal network.  Defaults to ``"gene_set"``.
        score_name: Key added to ``adata.obs`` for the AUCell score.
            Defaults to ``"aucell_score"``.
        layer: Layer to use for gene expression.  Defaults to ``None``
            (uses ``adata.X``).
        gene_symbols: Column in ``adata.var`` that contains gene symbols.
            When provided, ``genes`` is matched against this column and
            mapped to ``adata.var_names``.  Defaults to ``None``.
        min_n: Minimum number of genes from the list that must be present
            in ``adata.var_names`` for scoring to proceed.  Defaults to
            ``5``.
        n_up: Number of top-ranked features used to compute the AUC.
            When ``None`` (default), an adaptive value is calculated as
            ``max(5% × n_vars, n_vars ÷ k)`` where ``k`` is the number of
            found genes.  This ensures ~63 %% of cells are expected to
            receive a non-zero score regardless of gene-set size.  Pass an
            explicit integer to override.
        copy: If ``True``, operate on a copy of ``adata`` and return it.
            If ``False`` (default), modify ``adata`` in place and return
            ``None``.

    Returns:
        A modified copy of ``adata`` when ``copy=True``, otherwise
        ``None``.

    Raises:
        ImportError: If ``decoupler`` is not installed.
        ValueError: If fewer than ``min_n`` input genes are found in
            ``adata.var_names``.

    Example:
        >>> scutils.tl.compute_aucell_scores(
        ...     adata,
        ...     genes=["SELL", "CCR7", "TCF7", "LEF1", "KLF2"],
        ...     score_name="naive_T_aucell",
        ... )
        >>> adata.obs["naive_T_aucell"]
    """
    if not _DECOUPLER_AVAILABLE:
        raise ImportError(
            "decoupler is required for compute_aucell_scores. "
            "Install it with: pip install 'scutils[decoupler]'"
        )

    adata = adata.copy() if copy else adata

    resolved = _resolve_genes(adata, genes, gene_symbols)
    found = _warn_missing_genes(
        resolved, adata.var_names, min_n, func_name="compute_aucell_scores"
    )
    net = pd.DataFrame({"source": set_name, "target": found})

    # Adaptive n_up: use the larger of 5 % of all features and
    # n_vars / k (ensures ~63 % of cells are expected to score non-zero
    # even for small gene sets on large datasets).
    if n_up is None:
        n_up = max(
            int(np.ceil(0.05 * adata.n_vars)),
            adata.n_vars // len(found),
        )

    run_kwargs: dict = dict(verbose=False, n_up=n_up)
    if layer is not None:
        run_kwargs["layer"] = layer

    if hasattr(dc, "mt") and hasattr(dc.mt, "aucell"):
        # New decoupler API (>= 2.x)
        dc.mt.aucell(adata, net, tmin=min_n, **run_kwargs)
    else:
        # Legacy decoupler API
        dc.run_aucell(
            adata,
            net=net,
            source="source",
            target="target",
            min_n=min_n,
            **run_kwargs,
        )

    est_key = _get_obsm_key(adata, "aucell_estimate", "score_aucell")
    adata.obs[score_name] = adata.obsm[est_key][set_name].values
    del adata.obsm[est_key]

    return adata if copy else None