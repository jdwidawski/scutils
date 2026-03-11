# scutils

**scutils** is a Python package of utility functions for single-cell RNA-sequencing (scRNA-seq) data analysis. It provides reusable, well-tested helpers for plotting, pre-processing, and analytical tools built on top of [Scanpy](https://scanpy.readthedocs.io/) and [AnnData](https://anndata.readthedocs.io/).

## Installation

### From source (recommended during development)

Requires [uv](https://github.com/astral-sh/uv) and Python ≥ 3.11.

```bash
git clone https://github.com/<your-org>/single_cell_utilities.git
cd single_cell_utilities
uv sync          # creates .venv and installs all runtime dependencies
```

### With optional dependency groups

```bash
uv sync --extra dev   # adds pytest, pytest-cov, ruff
uv sync --extra docs  # adds sphinx, furo, myst-parser, sphinx-autodoc-typehints
```

### Editable install via pip (without uv)

```bash
pip install -e ".[dev]"
```

---

## Quick start

```python
import scanpy as sc
import scutils.plotting as pl
import scutils.preprocessing as pp
import scutils.tools as tl

adata = sc.read_h5ad("my_dataset.h5ad")

# --- Plotting ---

# One subplot per Leiden cluster, coloured by cluster identity
fig = pl.embedding_category_multiplot(adata, column="leiden")

# One subplot per cell type, coloured by a gene
fig = pl.embedding_gene_expression_multiplot(adata, column="cell_type", feature="CD3D")

# Dotplot of mean expression × fraction expressing, split by two obs columns
fig = pl.dotplot_expression_two_categories(
    adata, feature="CD3D", category_x="cell_type", category_y="condition"
)

# Volcano plot from a Scanpy DE result table
de_df = sc.get.rank_genes_groups_df(adata, group="T cell")
fig = pl.volcano_plot(de_df, pval_cutoff=0.01, lfc_cutoff=1.0)

# --- Pre-processing ---

# Merge two datasets, filling absent genes with zeros
combined = pp.concat_anndata_with_zeros(adata_A, adata_B, left_name="A", right_name="B")
pp.print_zero_filling_summary(combined)

# Keep only genes present in at least 2 datasets
combined = pp.filter_genes_by_presence(combined, min_datasets=2)

# --- Tools ---

# Subcluster specific Leiden clusters at a finer resolution
tl.iterative_subcluster(
    adata,
    cluster_col="leiden",
    subcluster_resolutions={"3": 0.5, "7": 0.3},
)

# Rename subclusters using a biologically-meaningful mapping
tl.rename_subcluster_labels(
    adata,
    col="leiden_subclustered",
    label_map={"CD4 T": ["3,0", "3,1"], "CD8 T": ["7,0"]},
)
```

---

## Module reference

| Module | Key functions |
| --- | --- |
| `scutils.plotting` | `embedding_category_multiplot`, `embedding_gene_expression_multiplot`, `plot_feature_boxplot`, `plot_feature_boxplot_aggregated`, `dotplot_expression_two_categories`, `heatmap_expression_two_categories`, `volcano_plot`, `plot_embedding_categories`, `plot_density_embedding`, `plot_density_embedding_multiplot`, `plot_density_embedding_comparison` |
| `scutils.preprocessing` | `concat_anndata_with_zeros`, `get_zero_filled_genes_for_dataset`, `get_datasets_missing_gene`, `get_zero_filling_stats`, `print_zero_filling_summary`, `filter_genes_by_presence` |
| `scutils.tools` | `iterative_subcluster`, `rename_subcluster_labels`, `spatial_split_clusters`, `plot_spatial_split_diagnostics` |

Full API documentation is available at **[GitHub Pages](https://YOUR_ORG.github.io/single_cell_utilities/)** (replace with your URL once deployed — see [docs/deployment.md](docs/deployment.md)).

---

## Development

### Running tests

```bash
uv run pytest                              # all tests
uv run pytest tests/plotting/ -q          # plotting only
uv run pytest --cov=scutils --cov-report=html   # with coverage
```

### Linting and formatting

```bash
uv run ruff check .          # lint
uv run ruff format --check . # check formatting
uv run ruff format .         # apply formatting
```

### Building documentation locally

```bash
uv run sphinx-build docs/ docs/_build/html
# then open docs/_build/html/index.html
```

---

## Project structure

```text
├── src/
│   └── scutils/
│       ├── __init__.py
│       ├── plotting/
│       │   ├── __init__.py
│       │   ├── _utils.py          # shared helpers (_resolve_palette)
│       │   ├── boxplots.py
│       │   ├── density_plotting.py
│       │   ├── dotplots.py
│       │   ├── embeddings.py
│       │   ├── heatmaps.py
│       │   └── volcano_plot.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── concatenation.py
│       └── tools/
│           ├── __init__.py
│           └── clustering.py
├── tests/
│   ├── conftest.py
│   ├── plotting/
│   ├── preprocessing/
│   └── tools/
├── docs/
├── pyproject.toml
└── README.md
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install the dev dependencies: `uv sync --extra dev`.
3. Write tests for your changes in the appropriate `tests/` subdirectory.
4. Ensure `uv run pytest` and `uv run ruff check .` pass before opening a PR.
5. Follow the [Conventional Commits](https://www.conventionalcommits.org/) spec for all commit messages.

---

## License

MIT
