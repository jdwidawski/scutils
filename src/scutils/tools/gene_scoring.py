"""Gene-set scoring utilities for single-cell RNA-seq data."""

from __future__ import annotations

from typing import List, Optional

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


def compute_ulm_scores(
    adata: AnnData,
    net: pd.DataFrame,
    source: str = "source",
    target: str = "target",
    weight: str = "weight",
    score_name: str = "ulm_score",
    layer: Optional[str] = None,
    min_n: int = 5,
    batch_size: int = 10000,
    return_pvals: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    """Compute ULM transcription-factor activity scores using decoupler.

    Runs Univariate Linear Model (ULM) from the :mod:`decoupler` package
    on each source in ``net``.  Estimated activity scores for every source
    are added as new columns in ``adata.obs`` using the prefix
    ``score_name``.  Optionally, a DataFrame of p-values can be stored in
    ``adata.obsm``.

    Args:
        adata: Annotated data matrix.
        net: Prior knowledge network with at least three columns
            identifying sources (e.g. transcription factors), targets
            (e.g. genes), and edge weights.  Column names are controlled
            by ``source``, ``target``, and ``weight``.
        source: Column in ``net`` containing source names.
            Defaults to ``"source"``.
        target: Column in ``net`` containing target gene names.
            Defaults to ``"target"``.
        weight: Column in ``net`` containing interaction weights.
            Defaults to ``"weight"``.
        score_name: Prefix for the columns added to ``adata.obs``.  Each
            source in the network gets a column named
            ``"{score_name}_{source_name}"``.  Defaults to
            ``"ulm_score"``.
        layer: Layer to use for gene expression.  Defaults to ``None``
            (uses ``adata.X``).
        min_n: Minimum number of targets that must be present in
            ``adata.var_names`` for a source to be scored.  Sources with
            fewer matches are silently dropped.  Defaults to ``5``.
        batch_size: Number of sources processed per batch.  Increase for
            speed at the cost of memory.  Defaults to ``10000``.
        return_pvals: If ``True``, store the p-value matrix as
            ``adata.obsm["{score_name}_pvals"]`` in addition to the
            activity estimates.  Defaults to ``False``.
        copy: If ``True``, operate on a copy of ``adata`` and return it.
            If ``False`` (default), modify ``adata`` in place and return
            ``None``.

    Returns:
        A modified copy of ``adata`` when ``copy=True``, otherwise
        ``None``.

    Raises:
        ImportError: If ``decoupler`` is not installed.

    Example:
        >>> import pandas as pd
        >>> net = pd.read_csv("collectri.csv")  # source / target / weight columns
        >>> scutils.tl.compute_ulm_scores(
        ...     adata,
        ...     net=net,
        ...     score_name="tf_activity",
        ...     return_pvals=True,
        ... )
        >>> # Each TF becomes a column in adata.obs:
        >>> adata.obs.filter(like="tf_activity_")
        >>> # P-value matrix stored in adata.obsm:
        >>> adata.obsm["tf_activity_pvals"]
    """
    if not _DECOUPLER_AVAILABLE:
        raise ImportError(
            "decoupler is required for compute_ulm_scores. "
            "Install it with: pip install 'scutils[decoupler]'"
        )

    adata = adata.copy() if copy else adata

    run_kwargs: dict = dict(
        source=source,
        target=target,
        weight=weight,
        min_n=min_n,
        batch_size=batch_size,
        verbose=False,
        use_raw=False,
    )
    if layer is not None:
        run_kwargs["layer"] = layer

    dc.run_ulm(adata, net=net, **run_kwargs)

    # Move estimates from obsm to obs with prefixed column names
    estimates: pd.DataFrame = adata.obsm["ulm_estimate"]
    for col in estimates.columns:
        adata.obs[f"{score_name}_{col}"] = estimates[col].values
    del adata.obsm["ulm_estimate"]

    # Handle p-values
    if return_pvals:
        adata.obsm[f"{score_name}_pvals"] = adata.obsm["ulm_pvals"]
    del adata.obsm["ulm_pvals"]

    return adata if copy else None