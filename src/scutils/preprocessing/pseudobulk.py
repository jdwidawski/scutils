"""Pseudobulk aggregation of single-cell count matrices."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_raw_counts(X: Any, n_sample: int = 1000) -> None:
    """Raise ``ValueError`` if *X* does not look like a raw integer count matrix.

    Samples up to *n_sample* non-zero values and checks that they are
    non-negative integers.  This catches log-transformed, normalised and
    scaled matrices that would invalidate pseudobulk DE workflows.

    Args:
        X: The expression matrix (dense or sparse).
        n_sample: Maximum number of non-zero entries to inspect.

    Raises:
        ValueError: If the matrix contains negative or non-integer values.
    """
    if issparse(X):
        data = X.data
    else:
        data = np.asarray(X).ravel()

    if len(data) == 0:
        return

    # Sub-sample for speed on large matrices
    if len(data) > n_sample:
        rng = np.random.default_rng(0)
        data = rng.choice(data, size=n_sample, replace=False)

    data = np.asarray(data, dtype=float)

    if np.any(data < 0):
        raise ValueError(
            "The expression matrix contains negative values.  "
            "Pseudobulk aggregation requires raw (untransformed) integer "
            "counts.  If you have log-transformed or scaled data, pass the "
            "appropriate raw-count layer via the ``layer`` parameter."
        )

    if not np.allclose(data, np.round(data)):
        raise ValueError(
            "The expression matrix contains non-integer values.  "
            "Pseudobulk aggregation requires raw (untransformed) integer "
            "counts.  If you have normalised or log-transformed data, pass "
            "the appropriate raw-count layer via the ``layer`` parameter."
        )


def _propagate_metadata(
    adata: AnnData,
    group_key: str,
    group_value: str,
) -> Dict[str, Any]:
    """Return obs metadata that is constant within a group.

    For every column in ``adata.obs`` (other than *group_key*), if all
    values within the subset are identical the value is propagated;
    otherwise ``None`` is stored.
    """
    subset_obs = adata.obs.loc[adata.obs[group_key] == group_value]
    row: Dict[str, Any] = {}
    for col in adata.obs.columns:
        if col == group_key:
            continue
        unique = subset_obs[col].unique()
        row[col] = unique[0] if len(unique) == 1 else None
    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pseudobulk(
    adata: AnnData,
    sample_col: str,
    groups_col: Optional[str] = None,
    groups_categories: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
    min_cells: int = 10,
    skip_count_check: bool = False,
    log1p_transform: bool = False,
) -> AnnData:
    """Aggregate a single-cell AnnData into a pseudobulk AnnData.

    Sums the raw count matrix across cells within each sample (and
    optionally within each combination of sample × group), producing one
    observation per aggregate.  The resulting object is suitable as input
    to :func:`scutils.tl.deseq2`.

    Args:
        adata: Annotated data matrix with raw integer counts in ``.X``
            (or in the layer specified by *layer*).
        sample_col: Column in ``adata.obs`` that identifies biological
            replicates (e.g. ``"patient_id"``, ``"sample"``).
        groups_col: Optional column in ``adata.obs`` used to stratify the
            aggregation.  When provided, one pseudobulk observation is
            created for every unique ``(sample_col, groups_col)``
            combination.  Useful for deriving per-cell-type pseudobulk
            (e.g. ``groups_col="cell_type"``).  Defaults to ``None``.
        groups_categories: Subset of categories from ``groups_col`` to
            pseudobulk individually.  Cells belonging to categories
            **not** in this list are pooled together into a single
            ``"Rest"`` group per sample.  Requires *groups_col* to be
            set.  Defaults to ``None`` (use all categories).
        layer: Name of a layer in ``adata.layers`` to use instead of
            ``adata.X``.  Defaults to ``None`` (use ``.X``).
        min_cells: Minimum number of cells required in a group for it to
            be retained in the output.  Groups with fewer cells are
            silently dropped.  Defaults to ``10``.
        skip_count_check: When ``True``, skip the raw-count validation
            (useful when you are certain the matrix is correct and want
            to avoid the overhead).  Defaults to ``False``.
        log1p_transform: When ``True``, apply :func:`scanpy.pp.log1p`
            to the aggregated count matrix before returning.  Defaults
            to ``False``.

    Returns:
        A new :class:`~anndata.AnnData` whose ``.X`` contains the summed
        counts, ``.obs`` carries per-sample (and per-group) metadata
        propagated from *adata*, and ``.var`` is copied from the input.
        An additional column ``"n_cells"`` records how many cells were
        aggregated into each observation.

    Raises:
        ValueError: If *sample_col* or *groups_col* is not found in
            ``adata.obs``.
        ValueError: If the expression matrix does not appear to contain
            raw integer counts (unless *skip_count_check* is ``True``).

    Example:
        >>> # Pseudobulk per patient
        >>> pb = scutils.pp.pseudobulk(adata, sample_col="patient_id")

        >>> # Pseudobulk per patient × cell type
        >>> pb = scutils.pp.pseudobulk(
        ...     adata,
        ...     sample_col="patient_id",
        ...     groups_col="cell_type",
        ... )
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if sample_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{sample_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if groups_col is not None and groups_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{groups_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if groups_categories is not None and groups_col is None:
        raise ValueError(
            "'groups_categories' requires 'groups_col' to be set."
        )

    # If groups_categories is provided, remap non-selected categories
    # to "Rest" so they are pooled together per sample.
    if groups_categories is not None:
        adata = adata.copy()
        mapped = adata.obs[groups_col].astype(str).copy()
        mapped[~mapped.isin(groups_categories)] = "Rest"
        adata.obs[groups_col] = mapped

    # Resolve the expression matrix
    X = adata.layers[layer] if layer is not None else adata.X

    if not skip_count_check:
        _check_raw_counts(X)

    # ------------------------------------------------------------------
    # Determine grouping keys
    # ------------------------------------------------------------------
    if groups_col is not None:
        key_cols = [sample_col, groups_col]
    else:
        key_cols = [sample_col]

    # Build a combined key column for iteration
    obs = adata.obs
    if len(key_cols) == 1:
        group_labels = obs[sample_col].astype(str)
    else:
        group_labels = (
            obs[sample_col].astype(str) + "___" + obs[groups_col].astype(str)
        )

    unique_labels = group_labels.unique()

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    summed_rows: List[np.ndarray] = []
    obs_records: List[Dict[str, Any]] = []
    kept_labels: List[str] = []

    for label in unique_labels:
        mask = group_labels == label
        n_cells = int(mask.sum())

        if n_cells < min_cells:
            continue

        sub_X = X[mask.values]
        row_sum = np.asarray(sub_X.sum(axis=0)).ravel()
        summed_rows.append(row_sum)

        # Parse label back into component values
        if groups_col is not None:
            parts = label.split("___", 1)
            sample_val, group_val = parts[0], parts[1]
        else:
            sample_val = label
            group_val = None

        record: Dict[str, Any] = {sample_col: sample_val}
        if groups_col is not None:
            record[groups_col] = group_val

        # Propagate metadata constant within this subset
        subset_obs = obs.loc[mask]
        for col in obs.columns:
            if col in key_cols:
                continue
            unique_vals = subset_obs[col].unique()
            record[col] = unique_vals[0] if len(unique_vals) == 1 else None

        record["n_cells"] = n_cells
        kept_labels.append(label)
        obs_records.append(record)

    if len(summed_rows) == 0:
        raise ValueError(
            f"No groups passed the min_cells={min_cells} threshold.  "
            "Consider lowering min_cells."
        )

    # ------------------------------------------------------------------
    # Assemble new AnnData
    # ------------------------------------------------------------------
    new_X = np.vstack(summed_rows)
    new_obs = pd.DataFrame(obs_records, index=kept_labels)
    new_var = adata.var.copy()

    pb_adata = AnnData(X=new_X, obs=new_obs, var=new_var)

    if log1p_transform:
        sc.pp.log1p(pb_adata)

    return pb_adata
