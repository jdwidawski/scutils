"""Differential expression analysis using pyDESeq2."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deseq2(
    adata: AnnData,
    design: str,
    contrast: List[str],
    *,
    alpha: float = 0.05,
    shrink_lfc: bool = True,
    lfc_shrink_coeff: Optional[str] = None,
    cooks_filter: bool = True,
    independent_filter: bool = True,
    refit_cooks: bool = True,
    fit_type: Literal["parametric", "mean"] = "parametric",
    ref_level: Optional[List[str]] = None,
    min_replicates: int = 7,
    n_cpus: Optional[int] = None,
    quiet: bool = True,
) -> pd.DataFrame:
    """Run a DESeq2 differential expression analysis on pseudobulk data.

    Wraps :class:`pydeseq2.dds.DeseqDataSet` and
    :class:`pydeseq2.ds.DeseqStats` to perform a complete Wald-test-based
    DE pipeline.  The input should be a pseudobulked
    :class:`~anndata.AnnData` (e.g. the output of
    :func:`scutils.pp.pseudobulk`) whose ``.X`` contains raw integer
    counts.

    Args:
        adata: Pseudobulk annotated data matrix.  ``.X`` must contain raw
            integer counts (samples × genes) and ``.obs`` must contain the
            columns referenced in *design*.
        design: A formulaic-style design formula (e.g. ``"~condition"`` or
            ``"~condition + batch"``).  Passed directly to
            :class:`~pydeseq2.dds.DeseqDataSet`.
        contrast: A three-element list ``[factor, level_test, level_ref]``
            that specifies the comparison.  For example,
            ``["condition", "treated", "control"]`` tests *treated* vs
            *control*.
        alpha: Significance threshold for adjusted p-values.  Defaults to
            ``0.05``.
        shrink_lfc: Whether to apply empirical-Bayes LFC shrinkage after
            the Wald test.  Highly recommended for ranking and
            visualisation.  Defaults to ``True``.
        lfc_shrink_coeff: Name of the LFC coefficient to shrink.  When
            ``None`` (default), the coefficient is automatically inferred
            from *contrast* using the pyDESeq2 naming convention
            (``"factor[T.level_test]"``).
        cooks_filter: Whether to filter outlier counts using Cook's
            distance.  Defaults to ``True``.
        independent_filter: Whether to apply independent filtering for
            p-value adjustment.  Defaults to ``True``.
        refit_cooks: Whether to refit the model for samples flagged as
            Cook's outliers.  Defaults to ``True``.
        fit_type: Dispersion-estimation strategy.  ``"parametric"``
            (default) fits a parametric curve; ``"mean"`` uses a
            mean-based estimate.
        ref_level: Optional reference level for the contrast factor, as
            ``[factor, level]`` (e.g. ``["condition", "control"]``).  When
            provided, the category is reordered so that *level* is the
            first category (reference).  When ``None``, pyDESeq2 uses the
            first category in alphabetical order.  Defaults to ``None``.
        min_replicates: Minimum number of replicates required to trigger
            Cook's filtering.  Defaults to ``7``.
        n_cpus: Number of CPUs for parallelised steps.  ``None`` uses
            pyDESeq2's default (single-threaded).  Defaults to ``None``.
        quiet: Suppress pyDESeq2 progress messages.  Defaults to ``True``.

    Returns:
        A :class:`pandas.DataFrame` indexed by gene name with columns
        ``baseMean``, ``log2FoldChange``, ``lfcSE``, ``stat``,
        ``pvalue`` and ``padj``.

    Raises:
        ImportError: If *pydeseq2* is not installed.
        ValueError: If *contrast* does not have exactly three elements.
        KeyError: If the design-formula columns are missing from
            ``adata.obs``.

    Example:
        >>> import scutils
        >>> pb = scutils.pp.pseudobulk(adata, sample_col="patient_id")
        >>> results = scutils.tl.deseq2(
        ...     pb,
        ...     design="~condition",
        ...     contrast=["condition", "treated", "control"],
        ... )
        >>> results.head()
    """
    # Lazy import so the package doesn't hard-depend on pydeseq2
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        raise ImportError(
            "pydeseq2 is required for differential expression analysis.  "
            "Install it with:  pip install 'scutils[de]'  or  "
            "pip install pydeseq2"
        ) from exc

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if len(contrast) != 3:
        raise ValueError(
            f"contrast must be a three-element list "
            f"[factor, level_test, level_ref], got {contrast!r}."
        )

    # ------------------------------------------------------------------
    # Prepare counts DataFrame and metadata
    # ------------------------------------------------------------------
    counts_df = pd.DataFrame(
        adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(),
        index=adata.obs_names,
        columns=adata.var_names,
    ).astype(int)

    metadata = adata.obs.copy()

    # Ensure the factor column is categorical
    factor_col = contrast[0]
    if factor_col in metadata.columns:
        metadata[factor_col] = metadata[factor_col].astype("category")

    # Apply reference level by reordering the categorical so that the
    # reference is the first category.  This replaces the deprecated
    # ref_level parameter of DeseqDataSet.
    if ref_level is not None:
        ref_factor, ref_val = ref_level
        if ref_factor in metadata.columns:
            cats = metadata[ref_factor].cat.categories.tolist()
            if ref_val in cats:
                cats.remove(ref_val)
                cats.insert(0, ref_val)
                metadata[ref_factor] = metadata[ref_factor].cat.reorder_categories(
                    cats
                )

    # ------------------------------------------------------------------
    # Fit DESeq2
    # ------------------------------------------------------------------
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design=design,
        refit_cooks=refit_cooks,
        fit_type=fit_type,
        min_replicates=min_replicates,
        n_cpus=n_cpus,
        quiet=quiet,
    )
    dds.deseq2()

    # ------------------------------------------------------------------
    # Statistical testing
    # ------------------------------------------------------------------
    stat_res = DeseqStats(
        dds,
        contrast=contrast,
        alpha=alpha,
        cooks_filter=cooks_filter,
        independent_filter=independent_filter,
        quiet=quiet,
        n_cpus=n_cpus,
    )
    stat_res.summary()

    # ------------------------------------------------------------------
    # Optional LFC shrinkage
    # ------------------------------------------------------------------
    if shrink_lfc:
        if lfc_shrink_coeff is None:
            # Auto-infer from the pyDESeq2 naming convention
            lfc_shrink_coeff = f"{contrast[0]}[T.{contrast[1]}]"
        stat_res.lfc_shrink(coeff=lfc_shrink_coeff)

    return stat_res.results_df


def format_deseq2_results(
    results_df: pd.DataFrame,
    *,
    pval_col: str = "padj",
    lfc_col: str = "log2FoldChange",
    gene_col: Optional[str] = None,
) -> pd.DataFrame:
    """Rename pyDESeq2 result columns to the Scanpy convention.

    Converts a :func:`deseq2` results DataFrame so that it can be passed
    directly to :func:`scutils.pl.volcano_plot` without specifying custom
    column names.

    The Scanpy convention (used by ``sc.tl.rank_genes_groups``) expects
    columns ``"names"``, ``"pvals_adj"`` and ``"logfoldchanges"``.

    Args:
        results_df: DataFrame returned by :func:`deseq2` (or any
            pyDESeq2 ``results_df``).
        pval_col: Column name in *results_df* that contains adjusted
            p-values.  Defaults to ``"padj"``.
        lfc_col: Column name in *results_df* that contains log-fold
            changes.  Defaults to ``"log2FoldChange"``.
        gene_col: Column name containing gene names.  When ``None``
            (default), the DataFrame index is used.

    Returns:
        A new :class:`pandas.DataFrame` with columns ``"names"``,
        ``"pvals_adj"`` and ``"logfoldchanges"``, plus all other columns
        from the input carried through unchanged.

    Example:
        >>> results = scutils.tl.deseq2(pb, design="~condition", ...)
        >>> df = scutils.tl.format_deseq2_results(results)
        >>> fig = scutils.pl.volcano_plot(df)
    """
    df = results_df.copy()

    # Gene names
    if gene_col is not None:
        df["names"] = df[gene_col]
    else:
        df["names"] = df.index

    # Rename columns
    rename_map: Dict[str, str] = {}
    if pval_col in df.columns and pval_col != "pvals_adj":
        rename_map[pval_col] = "pvals_adj"
    if lfc_col in df.columns and lfc_col != "logfoldchanges":
        rename_map[lfc_col] = "logfoldchanges"

    df = df.rename(columns=rename_map)
    return df
