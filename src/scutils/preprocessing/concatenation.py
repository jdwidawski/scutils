import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Optional
from scipy.sparse import issparse, csr_matrix, vstack, hstack
from collections.abc import Mapping


def concat_anndata_with_zeros(
    adata_left: AnnData,
    adata_right: AnnData,
    gene_ids_left: Optional[str] = None,
    gene_ids_right: Optional[str] = None,
    left_name: Optional[str] = "left",
    right_name: Optional[str] = "right",
) -> AnnData:
    """Concatenate two AnnData objects, adding zeros for missing genes.

    Genes present in one dataset but absent in the other are filled with zeros
    so that the resulting matrix is dense across all genes.  Zero-filling is
    tracked incrementally in ``adata.uns['zero_filled_genes']``: a
    ``pd.DataFrame`` with genes as index and dataset names as boolean columns
    (``True`` = zero-filled, ``False`` = original data).

    Args:
        adata_left: First AnnData object to concatenate.
        adata_right: Second AnnData object to concatenate.
        gene_ids_left: Column name in ``adata_left.var`` containing gene
            identifiers.  When ``None``, uses the index of ``adata_left.var``.
        gene_ids_right: Column name in ``adata_right.var`` containing gene
            identifiers.  When ``None``, uses the index of ``adata_right.var``.
        left_name: Label for the left dataset in the tracking table.  Pass
            ``None`` to skip adding a column for the left dataset (useful when
            *adata_left* is already a combined object from a prior call).
            Defaults to ``"left"``.
        right_name: Label for the right dataset in the tracking table.  Pass
            ``None`` to skip adding a column for the right dataset.  Defaults
            to ``"right"``.

    Returns:
        Concatenated ``AnnData`` with ``adata.uns['zero_filled_genes']``
        recording which genes were zero-filled per dataset.

    Example:
        >>> combined = concat_anndata_with_zeros(
        ...     adata_A, adata_B, left_name="donor_A", right_name="donor_B"
        ... )
        >>> combined.uns["zero_filled_genes"].sum()
    """
    
    # Get gene identifiers
    if gene_ids_left is None:
        genes_left = adata_left.var.index.to_numpy()
    else:
        genes_left = adata_left.var[gene_ids_left].to_numpy()
    
    if gene_ids_right is None:
        genes_right = adata_right.var.index.to_numpy()
    else:
        genes_right = adata_right.var[gene_ids_right].to_numpy()
    
    # Find unique and common genes
    genes_left_set = set(genes_left)
    genes_right_set = set(genes_right)
    
    genes_only_left = sorted(genes_left_set - genes_right_set)
    genes_only_right = sorted(genes_right_set - genes_left_set)
    genes_common = sorted(genes_left_set & genes_right_set)
    
    # Create ordered list of all genes
    all_genes = genes_only_left + genes_common + genes_only_right
    
    # ========== Step 1: Add missing genes with zeros ==========
    
    # Add genes to left dataset that are missing (genes_only_right)
    if len(genes_only_right) > 0:
        n_cells_left = adata_left.n_obs
        n_missing_left = len(genes_only_right)
        
        if issparse(adata_left.X):
            # Add sparse zero columns
            zeros_left = csr_matrix((n_cells_left, n_missing_left), dtype=adata_left.X.dtype)
            X_left_padded = hstack([adata_left.X, zeros_left], format='csr')
        else:
            # Add dense zero columns
            zeros_left = np.zeros((n_cells_left, n_missing_left), dtype=adata_left.X.dtype)
            X_left_padded = np.hstack([adata_left.X, zeros_left])
        
        # Create extended var for left
        var_left_extended = adata_left.var.copy()
        var_missing_left = pd.DataFrame(index=genes_only_right)
        var_left_extended = pd.concat([var_left_extended, var_missing_left], axis=0)
        genes_left_extended = np.concatenate([genes_left, genes_only_right])
    else:
        X_left_padded = adata_left.X
        var_left_extended = adata_left.var.copy()
        genes_left_extended = genes_left
    
    # Add genes to right dataset that are missing (genes_only_left)
    if len(genes_only_left) > 0:
        n_cells_right = adata_right.n_obs
        n_missing_right = len(genes_only_left)
        
        if issparse(adata_right.X):
            # Add sparse zero columns
            zeros_right = csr_matrix((n_cells_right, n_missing_right), dtype=adata_right.X.dtype)
            X_right_padded = hstack([adata_right.X, zeros_right], format='csr')
        else:
            # Add dense zero columns
            zeros_right = np.zeros((n_cells_right, n_missing_right), dtype=adata_right.X.dtype)
            X_right_padded = np.hstack([adata_right.X, zeros_right])
        
        # Create extended var for right
        var_right_extended = adata_right.var.copy()
        var_missing_right = pd.DataFrame(index=genes_only_left)
        var_right_extended = pd.concat([var_right_extended, var_missing_right], axis=0)
        genes_right_extended = np.concatenate([genes_right, genes_only_left])
    else:
        X_right_padded = adata_right.X
        var_right_extended = adata_right.var.copy()
        genes_right_extended = genes_right
    
    # ========== Step 2: Reorder genes to match ==========
    
    # Create mapping from gene to index in all_genes
    gene_to_target_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    
    # Get reordering indices
    reorder_idx_left = np.array([gene_to_target_idx[g] for g in genes_left_extended])
    reorder_idx_right = np.array([gene_to_target_idx[g] for g in genes_right_extended])
    
    # Create inverse permutation to reorder columns
    # We need to find which column in the padded matrix goes to which position
    left_order = np.argsort(reorder_idx_left)
    right_order = np.argsort(reorder_idx_right)
    
    # Reorder columns
    X_left_reordered = X_left_padded[:, left_order]
    X_right_reordered = X_right_padded[:, right_order]
    
    # Reorder var DataFrames
    var_left_reordered = var_left_extended.iloc[left_order].copy()
    var_right_reordered = var_right_extended.iloc[right_order].copy()
    
    # ========== Step 3: Stack the matrices ==========
    
    if issparse(X_left_reordered) or issparse(X_right_reordered):
        X_combined = vstack([X_left_reordered, X_right_reordered], format='csr')
    else:
        X_combined = np.vstack([X_left_reordered, X_right_reordered])
    
    # ========== Merge .obs ==========
    obs_left = adata_left.obs.copy()
    obs_right = adata_right.obs.copy()
    
    # Add batch/dataset identifier
    obs_left['batch'] = left_name if left_name is not None else 'left'
    obs_right['batch'] = right_name if right_name is not None else 'right'
    
    # Concatenate obs
    obs_combined = pd.concat([obs_left, obs_right], axis=0, ignore_index=False)
    
    # ========== Merge .var ==========
    
    # Find common columns in .var
    var_cols_left = set(var_left_reordered.columns)
    var_cols_right = set(var_right_reordered.columns)
    common_var_cols = var_cols_left & var_cols_right
    
    # Initialize var DataFrame with all genes
    var_combined = pd.DataFrame(index=all_genes)
    
    # For common columns, take from left where available, otherwise from right
    for col in common_var_cols:
        # Start with values from left
        var_combined[col] = var_left_reordered[col].values
        
        # Fill missing values (genes only in right) from right dataset
        mask_missing = var_combined[col].isna()
        if mask_missing.any():
            var_combined[col] = var_combined[col].astype(object)
            var_combined.loc[mask_missing, col] = var_right_reordered.loc[mask_missing, col]
    
    # ========== Track zero-filled genes in table format ==========
    
    # Check if table already exists
    if 'zero_filled_genes' in adata_left.uns:
        # Table exists - we're doing incremental concatenation
        existing_table = adata_left.uns['zero_filled_genes'].copy()
        
        # Reindex to include all genes (add new rows for genes only in right)
        zero_filled_table = existing_table.reindex(all_genes, fill_value=False)
        
        # Update existing columns: genes from right that are new get marked as True
        for col in zero_filled_table.columns:
            zero_filled_table.loc[genes_only_right, col] = True
        
        # Add new columns based on left_name and right_name
        if left_name is not None and isinstance(left_name, str):
            # Add column for left dataset
            zero_filled_table[left_name] = False
            zero_filled_table.loc[genes_only_right, left_name] = True
        
        if right_name is not None and isinstance(right_name, str):
            # Add column for right dataset
            zero_filled_table[right_name] = False
            zero_filled_table.loc[genes_only_left, right_name] = True
    
    else:
        # No existing table - create new one
        zero_filled_table = pd.DataFrame(index=all_genes)
        
        # Add columns based on left_name and right_name
        if left_name is not None and isinstance(left_name, str):
            zero_filled_table[left_name] = False
            zero_filled_table.loc[genes_only_right, left_name] = True
        
        if right_name is not None and isinstance(right_name, str):
            zero_filled_table[right_name] = False
            zero_filled_table.loc[genes_only_left, right_name] = True
    
    # ========== Create combined AnnData object ==========
    adata_combined = AnnData(
        X=X_combined,
        obs=obs_combined,
        var=var_combined
    )
    
    # ========== Merge .uns ==========
    def merge_uns(uns_left, uns_right):
        """Merge .uns dictionaries following scanpy's approach."""
        uns_merged = {}
        
        all_keys = set(uns_left.keys()) | set(uns_right.keys())
        
        # Skip zero_filled_genes as we handle it separately
        all_keys.discard('zero_filled_genes')
        
        for key in all_keys:
            if key not in uns_left:
                uns_merged[key] = uns_right[key]
            elif key not in uns_right:
                uns_merged[key] = uns_left[key]
            else:
                val_left = uns_left[key]
                val_right = uns_right[key]
                
                if isinstance(val_left, Mapping) and isinstance(val_right, Mapping):
                    uns_merged[key] = merge_uns(val_left, val_right)
                elif type(val_left) == type(val_right):
                    uns_merged[key] = val_left
                else:
                    uns_merged[f"{key}_{left_name if left_name else 'left'}"] = val_left
                    uns_merged[f"{key}_{right_name if right_name else 'right'}"] = val_right
        
        return uns_merged
    
    adata_combined.uns = merge_uns(adata_left.uns, adata_right.uns)
    
    # Add zero-filling tracking table
    adata_combined.uns['zero_filled_genes'] = zero_filled_table
    
    return adata_combined


# ========== Helper Functions ==========

def get_zero_filled_genes_for_dataset(adata: AnnData, dataset_name: str) -> list:
    """Return genes that were zero-filled for a specific dataset.

    Args:
        adata: Combined ``AnnData`` object produced by
            :func:`concat_anndata_with_zeros`.
        dataset_name: Dataset label (column in ``adata.uns['zero_filled_genes']``).

    Returns:
        List of gene names that were zero-filled for *dataset_name*.  Returns
        an empty list if no tracking information is present or the dataset is
        not found.
    """
    if 'zero_filled_genes' not in adata.uns:
        return []
    
    table = adata.uns['zero_filled_genes']
    
    if dataset_name not in table.columns:
        return []
    
    return table[table[dataset_name]].index.tolist()


def get_datasets_missing_gene(adata: AnnData, gene: str) -> list:
    """Return datasets that are missing a specific gene (i.e. it was zero-filled).

    Args:
        adata: Combined ``AnnData`` object produced by
            :func:`concat_anndata_with_zeros`.
        gene: Gene name to query.

    Returns:
        List of dataset names for which *gene* was zero-filled.  Returns an
        empty list if no tracking information is present or the gene is not
        in the index.
    """
    if 'zero_filled_genes' not in adata.uns:
        return []
    
    table = adata.uns['zero_filled_genes']
    
    if gene not in table.index:
        return []
    
    # Get columns where this gene is True (zero-filled)
    return table.columns[table.loc[gene]].tolist()


def get_zero_filling_stats(adata: AnnData) -> pd.DataFrame:
    """Return summary statistics of zero-filling for each dataset.

    Args:
        adata: Combined ``AnnData`` object produced by
            :func:`concat_anndata_with_zeros`.

    Returns:
        ``pd.DataFrame`` indexed by dataset name with columns:
        ``n_zero_filled``, ``n_original``, ``pct_zero_filled``, and
        (when available) ``n_cells``.  Returns an empty ``DataFrame`` if
        no tracking information is present.
    """
    if 'zero_filled_genes' not in adata.uns:
        return pd.DataFrame()
    
    table = adata.uns['zero_filled_genes']
    
    stats = pd.DataFrame(index=table.columns)
    stats['n_zero_filled'] = table.sum(axis=0)
    stats['n_original'] = len(table) - stats['n_zero_filled']
    stats['pct_zero_filled'] = (stats['n_zero_filled'] / len(table) * 100).round(2)
    
    # Add cell counts if batch info available
    if 'batch' in adata.obs.columns:
        stats['n_cells'] = adata.obs['batch'].value_counts()
    
    return stats


def print_zero_filling_summary(adata: AnnData) -> None:
    """Print a comprehensive summary of zero-filling across all datasets.

    Args:
        adata: Combined ``AnnData`` object produced by
            :func:`concat_anndata_with_zeros`.
    """
    if 'zero_filled_genes' not in adata.uns:
        print("No zero filling information found.")
        return
    
    table = adata.uns['zero_filled_genes']
    
    print("=" * 70)
    print("ZERO-FILLING SUMMARY")
    print("=" * 70)
    print(f"\nTotal datasets: {len(table.columns)}")
    print(f"Total genes in atlas: {len(table)}")
    print(f"Total cells in atlas: {adata.n_obs}")
    
    # Get statistics
    stats = get_zero_filling_stats(adata)
    
    print("\n" + "-" * 70)
    print("Per-dataset statistics:")
    print("-" * 70)
    print(stats.to_string())
    
    # Find genes present in all datasets (no zeros anywhere)
    genes_in_all = table.index[~table.any(axis=1)].tolist()
    print(f"\n\nGenes present in ALL datasets (no zero-filling): {len(genes_in_all)}")
    
    # Find genes missing in all datasets (zeros everywhere)
    if len(table.columns) > 1:
        genes_in_none = table.index[table.all(axis=1)].tolist()
        print(f"Genes missing in ALL datasets: {len(genes_in_none)}")
        if len(genes_in_none) > 0 and len(genes_in_none) <= 5:
            print(f"  {genes_in_none}")
    
    print("\n" + "=" * 70)
    print("\nDetailed table available in .uns['zero_filled_genes']")
    print(f"Shape: {table.shape} (genes × datasets)")


def filter_genes_by_presence(adata: AnnData, min_datasets: int = 1) -> AnnData:
    """Filter genes based on how many datasets they appear in originally.

    Genes that were zero-filled in too many datasets are dropped.  The
    ``adata.uns['zero_filled_genes']`` tracking table is updated to reflect
    only the retained genes.

    Args:
        adata: Combined ``AnnData`` object produced by
            :func:`concat_anndata_with_zeros`.
        min_datasets: Minimum number of datasets in which a gene must be
            *originally present* (not zero-filled) to be retained.  Defaults
            to ``1``.

    Returns:
        Filtered ``AnnData`` object with ``adata.uns['zero_filled_genes']``
        restricted to the surviving genes.
    """
    if 'zero_filled_genes' not in adata.uns:
        print("No zero filling information found. Returning original object.")
        return adata
    
    table = adata.uns['zero_filled_genes']
    
    # Count how many datasets each gene is present in (False = present)
    n_datasets_present = (~table).sum(axis=1)
    
    # Filter genes
    genes_to_keep = table.index[n_datasets_present >= min_datasets].tolist()
    
    print(f"Filtering: keeping {len(genes_to_keep)}/{len(table)} genes")
    print(f"  (present in at least {min_datasets} dataset(s))")
    
    # Filter the AnnData object
    adata_filtered = adata[:, genes_to_keep].copy()
    
    # Filter the zero_filled_genes table to only include retained genes
    adata_filtered.uns['zero_filled_genes'] = table.loc[genes_to_keep].copy()
    
    return adata_filtered