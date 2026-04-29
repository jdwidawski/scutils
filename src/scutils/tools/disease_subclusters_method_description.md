# Disease-Enriched Subcluster Detection

## Goals

1. **Identify disease-enriched subclusters** — spatially coherent regions on a UMAP embedding where cells from one or more disease conditions are statistically over-represented compared to a reference (control) or the full dataset.
2. **Ensure spatial contiguity** — every reported subcluster must occupy a single, visually distinct hub on the UMAP plot. Subclusters should not span disconnected regions.
3. **Minimize user parameter tuning** — the method automatically selects the Leiden resolution, estimates DBSCAN thresholds per-piece, and provides sensible defaults so users can run it out-of-the-box on datasets ranging from 10k to 500k+ cells.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: AnnData with kNN graph (PCA space) + UMAP coordinates   │
│         + disease labels in adata.obs                           │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Adaptive Over-Clustering (Leiden)                      │
│  ─────────────────────────────────────────                      │
│  Resolution = max(2.0, √(n_cells / 1000))                      │
│  Produces fine-grained micro-clusters that tile the UMAP        │
���────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Enrichment Screening (Fisher Exact Test)               │
│  ────────────────────────────────────────────────               │
│  For each micro-cluster: test if disease cells are              │
│  over-represented (fold ≥ min_enrichment_fold).                 │
│  Optionally tests against a reference group instead of          │
│  the full dataset.                                              │
│                                                                 │
│  Modes:                                                         │
│    • None (default): screen per-disease independently           │
│    • "pool": merge all diseases → single binary label           │
│    • "union": screen per-disease, take union of enriched        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: kNN-Graph Connected Components                         │
│  ──────────────────────────────────────                         │
│  Extract the sub-graph of enriched cells from the               │
│  pre-computed kNN graph (PCA space).  Connected                 │
│  components define candidate subclusters.                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: GMM Splitting (BIC Model Selection)                    │
│  ───────────────────────────────────────────                    │
│  For each connected component, fit a Gaussian Mixture           │
│  Model on 2-D UMAP coordinates with k = 1..max_k.              │
│  Select k by BIC (lowest = best).  If k > 1, split             │
│  the component into GMM-assigned pieces.                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: kNN Graph Re-Validation                                │
│  ───────────────────────────────                                │
│  Each GMM piece is checked for connectivity in the              │
│  kNN graph.  Disconnected sub-pieces are split into             │
│  separate subclusters.                                          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: DBSCAN Spatial Enforcement                             │
│  ──────────────────────────────────                             │
│  Final guarantee of UMAP contiguity.  For each piece:           │
│    1. Estimate eps = mean(k-th NN distance) × multiplier        │
│    2. Run DBSCAN (min_samples=15)                               │
│    3. If multiple clusters found → split                        │
│    4. Noise points → assign to nearest cluster                  │
│    5. Small fragments → absorb into nearest valid cluster       │
│                                                                 │
│  The multiplier is controlled by spatial_sensitivity:           │
│    "low" = 2.0 │ "medium" = 1.5 │ "high" = 1.0 │              │
│    "very_high" = 0.7                                            │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7 (Optional): Cell-Type Splitting                         │
│  ──────────────────────────────────────                         │
│  Intersect each subcluster with cell types.  Each               │
│  cell-type × subcluster piece must independently pass:          │
│    • min_subcluster_size                                        │
│    • min_enrichment_fold                                        │
│    • enrichment_fdr (after BH correction)                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: Final Statistical Test + FDR Correction                │
│  ──────────────────────────────────────────────                 │
│  Fisher exact test (one-sided) on each final subcluster.        │
│  Benjamini-Hochberg FDR correction across all candidates.       │
│  Subclusters failing FDR threshold are set to "background".     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                         │
│  ──────                                                         │
│  • adata.obs[result_key]: categorical subcluster labels         │
│  • adata.uns[f'{result_key}_info']: DataFrame with statistics   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Method Description

### Step 1: Adaptive Over-Clustering

The dataset is over-clustered using the Leiden algorithm on the pre-computed kNN graph. The resolution is chosen automatically based on dataset size:

```
resolution = max(2.0, √(n_cells / 1000))
```

| Dataset Size | Resolution | Approx. Micro-Clusters |
|---|---|---|
| 10,000 | 3.2 | ~50–100 |
| 30,000 | 5.5 | ~150–300 |
| 100,000 | 10.0 | ~500–1000 |
| 150,000 | 12.2 | ~750–1500 |
| 500,000 | 22.4 | ~2000–4000 |

**Rationale:** The square-root scaling grows faster for small datasets (ensuring sufficient granularity) and flattens for large datasets (preventing excessively tiny micro-clusters). The floor of 2.0 guarantees reasonable over-clustering even on very small datasets. This was empirically validated on datasets of 100–150k cells where resolutions of 8–10 produced good results.

**Why over-cluster?** The micro-clusters serve as spatial "tiles" on the UMAP. Disease enrichment is first assessed at the micro-cluster level, which provides a natural spatial granularity without requiring a density grid or kernel bandwidth selection.

---

### Step 2: Enrichment Screening

Each micro-cluster is tested for disease enrichment using the **Fisher exact test** (one-sided, testing for over-representation). A micro-cluster is considered enriched if:

```
fold_enrichment ≥ min_enrichment_fold (default: 2.0)
```

where:

```
fold = (n_disease_in_cluster / n_cluster) / (n_disease_total / n_total)
```

**Reference group option:** When `reference_group` is specified (e.g., `"Control"`), the 2×2 contingency table is restricted to disease cells + reference cells only — other diseases are excluded. This tests the biologically relevant question: "Is this disease over-represented here compared to control?" If fewer than `min_reference_cells` (default: 50) reference cells are available, the test falls back to the full dataset. The reference group labels are automatically excluded from the set of diseases being tested.

**Combine diseases modes:**

| Mode | Behavior | Use Case |
|---|---|---|
| `None` | Screen independently per disease | Standard: find disease-specific subclusters |
| `"pool"` | Merge all diseases into one binary label before screening | Find subclusters enriched for any disease vs control |
| `"union"` | Screen per-disease, take union of enriched micro-clusters, then merge/split with combined mask | Sensitive to per-disease signals but produces unified subclusters |

---

### Step 3: kNN-Graph Connected Components

All cells belonging to enriched micro-clusters are extracted. The sub-graph of the pre-computed kNN graph (built in PCA space) is computed for these cells, and its connected components are identified.

**Purpose:** Merge enriched micro-clusters that are topologically connected into candidate subclusters. Non-enriched micro-clusters act as natural barriers preventing unrelated regions from merging.

**Limitation:** The kNN graph is built in PCA space, not UMAP space. Two cells can be kNN neighbors in high-dimensional space but appear far apart on UMAP. This is why subsequent steps (GMM and DBSCAN) are needed to enforce visual/spatial contiguity.

---

### Step 4: GMM Splitting

For each connected component, a Gaussian Mixture Model is fitted on the 2-D UMAP coordinates. The number of components *k* is selected by **Bayesian Information Criterion (BIC)**:

- Fit GMMs with *k* = 1, 2, ..., `gmm_max_components` (default: 5)
- Select *k* with the lowest BIC
- If *k* > 1, split the component into *k* pieces

**Default values:**

| Parameter | Default | Rationale |
|---|---|---|
| `gmm_max_components` | 5 | Most subclusters have ≤ 5 distinct hubs; higher values increase computation without benefit |
| `gmm_covariance_type` | `"full"` | Allows elliptical clusters of any orientation; more flexible than `"spherical"` or `"diag"` |

**Limitations:** GMM assumes Gaussian (elliptical) component shapes. UMAP clusters can be irregular, elongated, or crescent-shaped. GMM may under-split non-Gaussian shapes or occasionally mis-split elongated clusters. The subsequent DBSCAN step compensates for these cases.

---

### Step 5: kNN Graph Re-Validation

Each GMM piece is checked for connectivity in the kNN graph. If a piece is graph-disconnected (i.e., the GMM assigned cells from disconnected graph regions to the same component), it is split into its connected components.

**Purpose:** Catch cases where GMM groups cells that are close in UMAP space but not biologically connected in the transcriptomic graph.

---

### Step 6: DBSCAN Spatial Enforcement

The final guarantee of UMAP spatial contiguity. For each piece after GMM + kNN re-validation:

1. **Estimate eps automatically:**
   ```
   eps = mean(k-th nearest-neighbour distance) × multiplier
   ```
   where *k* = min(15, n_cells_in_piece − 1). The mean k-th NN distance captures the local density of each piece, and the multiplier scales it.

2. **Run DBSCAN** with `min_samples=15` (matching typical kNN *k*).

3. **If multiple clusters are found** → split the piece.

4. **Noise points** (DBSCAN label = −1) → assigned to their nearest non-noise cluster.

5. **Small fragments** (< `min_subcluster_size`) → absorbed into the nearest valid cluster by centroid distance.

**`spatial_sensitivity` parameter:**

| Value | Multiplier | Behavior |
|---|---|---|
| `"low"` | 2.0 | Only splits clearly separated hubs |
| `"medium"` (default) | 1.5 | Balanced — splits most multi-hub pieces |
| `"high"` | 1.0 | Splits smaller protrusions from main clusters |
| `"very_high"` | 0.7 | Very aggressive — splits anything that sticks out |

**Why adaptive eps?** A fixed eps would fail across datasets with different cell densities or across cell types within the same dataset. By computing the eps from each piece's own distance distribution, the threshold automatically adapts: dense pieces get tight eps, sparse pieces get looser eps.

---

### Step 7: Cell-Type Splitting (Optional)

When `celltype_key` is provided, each subcluster is intersected with each cell type. Every cell-type × subcluster piece must independently satisfy:

- `n_cells ≥ min_subcluster_size` (default: 100)
- `fold_enrichment ≥ min_enrichment_fold` (default: 2.0)
- `p_adjusted < enrichment_fdr` (after BH correction in Step 8)

**Design choice:** The pipeline clusters globally (not per cell type) to preserve the spatial structure discovered in UMAP space. Cell-type splitting is purely a post-hoc intersection — it does not influence how subclusters are shaped.

---

### Step 8: Final Statistical Test + FDR Correction

Each final subcluster (after all splitting and filtering) is tested with a one-sided **Fisher exact test** for enrichment. The test uses the same reference configuration as the screening step.

**Benjamini-Hochberg FDR correction** is applied across all candidate subclusters simultaneously. Subclusters that fail the FDR threshold (`enrichment_fdr`, default: 0.05) are removed and their cells labelled as `"background"`.

---

## Multi-Layer Splitting Strategy

The pipeline uses four complementary layers to ensure subclusters are both biologically connected and spatially contiguous:

| Layer | Space | What It Catches |
|---|---|---|
| kNN Connected Components | PCA (high-dim) | Cells with no transcriptomic path between them |
| GMM + BIC | UMAP (2-D) | Multimodal distributions (two blobs with a bridge) |
| kNN Re-Validation | PCA (high-dim) | GMM pieces that are graph-disconnected |
| DBSCAN Enforcement | UMAP (2-D) | Pieces that are graph-connected but spatially distant on UMAP |

Each layer catches failure modes that the others miss. Together they provide a robust guarantee that the final subclusters are both transcriptomically coherent (connected in the kNN graph) and visually contiguous (occupy one hub on the UMAP).

---

## Parameters Summary

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_enrichment_fold` | 2.0 | Minimum fold-enrichment at both screening and final validation |
| `min_subcluster_size` | 100 | Minimum cells per reported subcluster |
| `enrichment_fdr` | 0.05 | Benjamini-Hochberg FDR threshold |
| `spatial_sensitivity` | `"medium"` | DBSCAN strictness for spatial splitting |

### Optional Feature Parameters

| Parameter | Default | Description |
|---|---|---|
| `combine_diseases` | `None` | `"pool"` or `"union"` to combine multiple diseases |
| `reference_group` | `None` | Test enrichment vs specific control group(s) |
| `min_reference_cells` | 50 | Fallback threshold for reference group |

### Advanced Parameters

| Parameter | Default | Description |
|---|---|---|
| `gmm_max_components` | 5 | Maximum Gaussian components for splitting |
| `gmm_covariance_type` | `"full"` | GMM covariance structure |

---

## Output

### `adata.obs[result_key]`

Categorical column with labels:
- `"background"` — cells not in any enriched subcluster
- `"{celltype}|{disease}|sub{id}"` — cells in a detected subcluster (e.g., `"T cells|AD|sub1"`)

### `adata.uns[f'{result_key}_info']`

DataFrame with one row per subcluster containing:

| Column | Description |
|---|---|
| `subset` | Cell type (or `"all_cells"`) |
| `subcluster` | Subcluster ID (e.g., `"sub1"`) |
| `disease` | Disease label (or `"combined"`) |
| `n_cells` | Total cells in subcluster |
| `n_disease_cells` | Disease-positive cells in subcluster |
| `fold_enrichment` | Observed / expected disease proportion |
| `pvalue` | Fisher exact test p-value |
| `pvalue_adj` | BH-adjusted p-value |
| `reference_used` | `"all"` when full dataset was used, or comma-separated reference group labels (e.g., `"Control"` or `"Control, Healthy"`), or `"all (fallback)"` when reference was too small |
| `diseases_contributing` | Original diseases present (only when `combine_diseases` is used) |

---

## Visualization

The plotting function `plot_disease_enriched_subclusters` generates a grid of UMAP panels:

```python
plot_disease_enriched_subclusters(
    adata,
    split_by='celltype',   # or 'disease'
    groups=['Microglia', 'Astrocytes'],
    ncols=2,               # panels per row
    figsize_panel=(5, 5),  # size of each panel
)
```

Each panel shows one group (cell type or disease) with:
- **Light grey:** cells not in the group
- **Darker grey:** cells in the group but not in any enriched subcluster
- **Coloured:** disease-enriched subcluster cells, coloured by subcluster label

Enrichment statistics are available programmatically via `adata.uns[f'{result_key}_info']`.

---

## Pre-requisites

```python
import scanpy as sc

# 1. Standard preprocessing
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)

# 2. Required: kNN graph
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')

# 3. Required: UMAP
sc.tl.umap(adata)
```

## Basic Usage

```python
from disease_subclusters_density import (
    detect_disease_enriched_subclusters,
    plot_disease_enriched_subclusters,
)

# Detect subclusters
detect_disease_enriched_subclusters(
    adata,
    disease_key='disease',
    celltype_key='cell_type',
    reference_group='Control',
)

# View statistics
adata.uns['disease_subcluster_info']

# Plot results — 2-column grid, split by cell type
plot_disease_enriched_subclusters(
    adata,
    celltype_key='cell_type',
    split_by='celltype',
    groups=['Microglia', 'Astrocytes'],
    ncols=2,
)

# Plot results — 3-column grid, split by disease
plot_disease_enriched_subclusters(
    adata,
    celltype_key='cell_type',
    split_by='disease',
    groups=['AD', 'PD'],
    ncols=3,
)
```