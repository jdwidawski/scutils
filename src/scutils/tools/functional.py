"""Functional / pathway-enrichment analysis tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fraction_pct(subset_size: int, base_size: int) -> Optional[float]:
    """Return ``100 * subset_size / base_size``, or ``None`` if *base_size* is 0."""
    if base_size == 0:
        return None
    return round(100 * (subset_size / base_size), 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_enriched_terms(
    gene_list: List[str],
    sources: List[str] = (
        "GO:BP",
        "GO:CC",
        "GO:MF",
        "REAC",
        "WP",
        "KEGG",
    ),
    pval_adjust_method: str = "fdr",
    pval_adjust_sign: float = 0.05,
    gene_background: Optional[List[str]] = None,
    min_term_size: int = 20,
    max_term_size: int = 1000,
    output_file: Optional[str] = None,
    organism: str = "hsapiens",
) -> pd.DataFrame:
    """Run gene-set enrichment analysis via g:Profiler.

    Wraps :func:`scanpy.queries.enrich` to query g:Profiler and returns a
    filtered, annotated results table.  Optionally writes the results to a
    file.

    Args:
        gene_list: Gene symbols or Ensembl IDs to test for enrichment.
        sources: Pathway / ontology databases to query.  Accepted values are
            ``"GO:BP"``, ``"GO:CC"``, ``"GO:MF"``, ``"REAC"``, ``"WP"``,
            ``"KEGG"``, ``"TF"``, ``"MIRNA"``, ``"HP"``, ``"HPA"``,
            ``"CORUM"``.  Pass an empty list to include all available sources.
            Defaults to the six most common databases.
        pval_adjust_method: Multiple-testing correction method passed to
            g:Profiler (e.g. ``"fdr"``).  Defaults to ``"fdr"``.
        pval_adjust_sign: Significance threshold for adjusted p-values.
            Defaults to ``0.05``.
        gene_background: Custom background gene list.  When ``None``, g:Profiler
            uses its default annotated background.  Defaults to ``None``.
        min_term_size: Minimum number of genes in a term for it to be
            returned.  Defaults to ``20``.
        max_term_size: Maximum number of genes in a term for it to be
            returned.  Defaults to ``1000``.
        output_file: Path to save the results.  Supported extensions are
            ``.csv``, ``.tsv``, and ``.xlsx``.  When ``None``, no file is
            written.  Defaults to ``None``.
        organism: g:Profiler organism identifier (e.g. ``"hsapiens"``,
            ``"mmusculus"``).  Defaults to ``"hsapiens"``.

    Returns:
        DataFrame of significant enriched terms with additional columns:

        - ``intersection_pct``: percentage of the term's genes present in
          *gene_list*.
        - ``-log10(adj_p_value)``: transformed p-value for downstream
          plotting.

    Raises:
        ValueError: If *output_file* has an unsupported extension.
        ValueError: If the directory of *output_file* does not exist.

    Example:
        >>> markers = adata.uns["rank_genes_groups"]["names"]["T"].tolist()
        >>> enrich_df = get_enriched_terms(
        ...     markers,
        ...     sources=["GO:BP", "REAC"],
        ...     pval_adjust_sign=0.01,
        ... )
        >>> enrich_df.head()
    """
    sources = list(sources)

    terms_enrich = sc.queries.enrich(
        gene_list,
        org=organism,
        gprofiler_kwargs={
            "ordered": False,
            "no_iea": False,
            "measure_underrepresentation": False,
            "significance_threshold_method": pval_adjust_method,
            "user_threshold": pval_adjust_sign,
            "domain_scope": "annotated",
            "background": gene_background,
        },
    )

    terms_enrich["intersection_pct"] = terms_enrich.apply(
        lambda x: _fraction_pct(x["intersection_size"], x["term_size"]), axis=1
    )
    terms_enrich["-log10(adj_p_value)"] = -np.log10(terms_enrich["p_value"])

    if sources:
        terms_enrich = terms_enrich[terms_enrich["source"].isin(sources)]
    terms_enrich = terms_enrich[
        (terms_enrich["term_size"] >= min_term_size)
        & (terms_enrich["term_size"] <= max_term_size)
    ]

    if output_file is not None:
        output_path = Path(output_file)
        if not output_path.parent.exists():
            raise ValueError(
                f"The directory '{output_path.parent}' does not exist. "
                "Please provide a correct output path."
            )
        extension = output_path.suffix.lstrip(".")
        # Serialise list-valued 'parents' column to pipe-separated strings
        terms_enrich = terms_enrich.copy()
        terms_enrich["parents"] = terms_enrich["parents"].apply(
            lambda p: "|".join(p)
        )
        if extension in ("csv", "tsv"):
            sep = "," if extension == "csv" else "\t"
            terms_enrich.to_csv(output_file, sep=sep, index=False)
        elif extension == "xlsx":
            terms_enrich.to_excel(output_file, index=False)
        else:
            raise ValueError(
                f"Unsupported output file extension '.{extension}'. "
                "Please use .csv, .tsv or .xlsx."
            )

    return terms_enrich
