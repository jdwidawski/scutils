# Quick start

```python
import scanpy as sc
import scutils.plotting as pl
import scutils.preprocessing as pp
import scutils.tools as tl

adata = sc.read_h5ad("my_dataset.h5ad")
```

## Embedding plots

```python
# One panel per Leiden cluster
fig = pl.embedding_category_multiplot(adata, column="leiden")

# Panels split by cell type, coloured by a gene
fig = pl.embedding_gene_expression_multiplot(adata, column="cell_type", feature="CD3D")
```

## Boxplots and dotplots

```python
fig = pl.plot_feature_boxplot(
    adata, feature="CD3D", hue_col="condition", split_col="cell_type"
)

fig = pl.dotplot_expression_two_categories(
    adata, feature="CD3D", category_x="cell_type", category_y="condition"
)
```

## Volcano plot

```python
de_df = sc.get.rank_genes_groups_df(adata, group="T cell")
fig = pl.volcano_plot(de_df, pval_cutoff=0.01, lfc_cutoff=1.0)
fig.savefig("volcano.png", dpi=150)
```

## Subclustering

```python
tl.iterative_subcluster(
    adata, cluster_col="leiden", subcluster_resolutions={"3": 0.5, "7": 0.3}
)
tl.rename_subcluster_labels(
    adata,
    col="leiden_subclustered",
    label_map={"CD4 T": ["3,0", "3,1"], "CD8 T": ["7,0"]},
)
```

## Dataset merging

```python
combined = pp.concat_anndata_with_zeros(
    adata_A, adata_B, left_name="donor_A", right_name="donor_B"
)
pp.print_zero_filling_summary(combined)
combined = pp.filter_genes_by_presence(combined, min_datasets=2)
```
