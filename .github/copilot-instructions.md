# GitHub Copilot Instructions

## Project Overview

This repository contains custom Python utility functions for single-cell RNA-sequencing (scRNA-seq) data analysis. It is currently a collection of standalone scripts and is intended to be refactored into a proper Python package in the future.

## Project Structure

```
src/
    plotting/       # Visualisation utilities (embeddings, volcano plots, density, AOI outlines)
    preprocessing/  # Data loading, concatenation, and pre-processing utilities
    tools/          # Analytical tools (clustering, subclustering, etc.)
```

When adding new functions, place them in the most appropriate subdirectory. If no subdirectory fits, propose a new one that follows the same naming convention (lowercase, descriptive noun describing the type of operation).

## Core Libraries and Conventions

### Primary dependencies

- **AnnData / Scanpy**: All data operations work with `anndata.AnnData` objects. Import as:
  ```python
  import scanpy as sc
  import anndata as ad
  from anndata import AnnData  # preferred for type hints
  ```
- **NumPy / pandas**: Use `import numpy as np` and `import pandas as pd`.
- **Matplotlib**: Use `import matplotlib.pyplot as plt` and `import matplotlib`. Return `matplotlib.figure.Figure` objects from plotting functions rather than calling `plt.show()`.
- **Seaborn**: Use `import seaborn as sns` for statistical plots.
- **Plotly**: May be used for interactive visualisations. Prefer returning `plotly.graph_objects.Figure` objects.
- **SciPy**: Use for sparse matrix operations (`scipy.sparse`) and numerical utilities.

### AnnData conventions

Use the standard AnnData slot semantics:

| Slot | Contents |
|---|---|
| `adata.X` | Count / expression matrix (often sparse `csr_matrix`) |
| `adata.obs` | Per-cell metadata (`pd.DataFrame`) |
| `adata.var` | Per-gene metadata (`pd.DataFrame`) |
| `adata.obsm` | Embeddings, e.g. `adata.obsm["X_umap"]` |
| `adata.uns` | Unstructured metadata, e.g. colours `adata.uns["{col}_colors"]` |
| `adata.layers` | Alternative expression matrices |

Always check for sparsity before operating on `adata.X`:
```python
from scipy.sparse import issparse
X = adata.X.toarray() if issparse(adata.X) else adata.X
```

Functions that mutate `adata` should do so **in place** and return `None`. Functions that produce a new object should return it explicitly.

## Code Style

### Type hints

Always annotate function signatures using `typing` or built-in generic types (Python ≥ 3.10 union syntax `X | Y` is acceptable). Prefer `Optional[X]` for nullable parameters with a default of `None`.

```python
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
```

### Docstrings

Write **Google-style docstrings** for every public function and class.

```python
def my_function(adata: AnnData, key: str, n: int = 10) -> pd.DataFrame:
    """Short one-line summary.

    Longer description of what the function does, when to use it, and any
    important implementation notes. Wrap at 88 characters.

    Args:
        adata: Annotated data matrix.
        key: Column name in ``adata.obs`` to operate on.
        n: Maximum number of results to return. Defaults to 10.

    Returns:
        A DataFrame with columns ... and rows indexed by cell barcode.

    Raises:
        ValueError: If ``key`` is not found in ``adata.obs``.
        TypeError: If ``n`` is not a positive integer.

    Example:
        >>> result = my_function(adata, key="leiden", n=5)
        >>> result.head()
    """
```

Key rules:
- Use double backticks for inline code references in docstrings (`` ``adata.obs`` ``).
- Always document `**kwargs` and what they are forwarded to.
- Include an `Example:` block for non-trivial public functions.

### Error handling

Raise informative exceptions with context. Prefer `ValueError` for bad argument values and `KeyError` for missing data slots.

```python
if key not in adata.obs.columns:
    raise ValueError(
        f"Key '{key}' not found in adata.obs. "
        f"Available columns: {list(adata.obs.columns)}"
    )
```

### Plotting function conventions

- Always accept a `figsize: Tuple[float, float]` parameter.
- Accept a `return_fig: bool = False` **or** simply return the figure directly — be consistent within a module.
- Never call `plt.show()` inside a utility function; leave display control to the caller.
- When accepting a `palette` argument, handle: a single colour string, a list of colours, a matplotlib colormap name, and `None` (fall back to `adata.uns` colours if available).
- Restore any `adata.uns` colour state that was temporarily modified during plotting.
- Accept `**kwargs` and forward them to the underlying Scanpy / Matplotlib call where appropriate.

### General style

- Follow **PEP 8**; target a line length of **88 characters** (Black-compatible).
- Use `f-strings` for string formatting.
- Prefer `pathlib.Path` over `os.path` for file-system operations.
- Do not mutate function arguments unless the function is explicitly documented as in-place.
- Sort imports: standard library → third-party → local, with a blank line between groups.

## Testing

Tests live in `tests/` (to be created at the package root) and are written with **pytest**.

### Conventions

- Mirror the `src/` structure: `tests/plotting/`, `tests/preprocessing/`, `tests/tools/`.
- Name test files `test_<module_name>.py` and test functions `test_<function_name>_<scenario>`.
- Use `pytest` fixtures for shared AnnData objects. Prefer small synthetic datasets (e.g. `sc.datasets.pbmc68k_reduced()` or manually constructed `AnnData` objects) over real data files.
- Use `pytest.raises` to test exception paths.
- Use `matplotlib.testing` or simply assert on returned figure types for plotting tests — avoid visual regression tests unless specifically requested.

```python
import pytest
import scanpy as sc
from src.tools.clustering import iterative_subcluster

@pytest.fixture
def small_adata():
    adata = sc.datasets.pbmc68k_reduced()
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, key_added="leiden")
    return adata

def test_iterative_subcluster_basic(small_adata):
    iterative_subcluster(
        small_adata,
        cluster_col="leiden",
        subcluster_resolutions={"0": 0.3},
    )
    assert "leiden_subclustered" in small_adata.obs.columns

def test_iterative_subcluster_invalid_col(small_adata):
    with pytest.raises(ValueError, match="not found in adata.obs"):
        iterative_subcluster(small_adata, cluster_col="nonexistent", subcluster_resolutions={})
```

## Package Layout

The repository is structured as an installable package named `scutils`:

```
single_cell_utilities/
    src/
        scutils/
            __init__.py          # exposes pl, pp, tl shortcuts
            plotting/
                __init__.py
                _utils.py        # shared helpers (e.g. _resolve_palette)
                boxplots.py
                density_plotting.py
                dotplots.py
                embeddings.py
                heatmaps.py
                volcano_plot.py
            preprocessing/
                __init__.py
                concatenation.py
            tools/
                __init__.py
                clustering.py
    tests/
        plotting/
        preprocessing/
        tools/
        conftest.py
    docs/
        conf.py
        index.rst
        Makefile
        _static/
            custom.css
        api/
            plotting.rst
            preprocessing.rst
            tools.rst
        user_guide/
            getting_started.rst
            plotting.rst
            preprocessing.rst
            tools.rst
    pyproject.toml
    README.md
```

Always use absolute imports from the package root (e.g. `from scutils.plotting._utils import _resolve_palette`).

### Subpackage shortcuts

`scutils.__init__` exposes three module aliases that mirror the Scanpy
convention (`sc.pl`, `sc.pp`, `sc.tl`):

```python
import scutils

scutils.pl   # → scutils.plotting
scutils.pp   # → scutils.preprocessing
scutils.tl   # → scutils.tools
```

Usage examples:

```python
import scutils

# Plotting
fig = scutils.pl.volcano_plot(adata, ...)

# Pre-processing
scutils.pp.concat_anndata_with_zeros(adatas, ...)

# Tools / clustering
scutils.tl.iterative_subcluster(adata, ...)
```

When adding a new public function, export it from the relevant subpackage
`__init__.py` so it is accessible via these shortcuts.

---

## Version Control — Conventional Commits

All commits **must** follow the [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit types

| Type | When to use |
|---|---|
| `feat` | A new function or user-visible capability |
| `fix` | A bug fix |
| `refactor` | Code restructuring without behaviour change |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `chore` | Build system, CI, dependencies, tooling |
| `style` | Formatting-only changes (whitespace, imports) |
| `perf` | Performance improvement |

### Examples

```
feat(plotting): add plot_feature_boxplot_aggregated_multiplot function

refactor(plotting): extract _resolve_palette into shared _utils.py

fix(clustering): convert NumPy docstrings to Google style

test(preprocessing): add test suite for concat_anndata_with_zeros

chore: add pyproject.toml, __init__.py files and .gitignore

docs: add Sphinx configuration and autodoc pages
```

Use the **scope** (the parenthesised part) to name the submodule being changed.  Breaking changes are marked with `!` after the type/scope (e.g. `refactor!: rename argument x → column`).

---

## Environment Setup — UV

This project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

### Installing the package in editable mode

```bash
# Create a virtual environment and install all dependencies
uv sync

# Install with optional dependency groups
uv sync --extra dev   # testing + linting
uv sync --extra docs  # sphinx + theme
```

### Running tests

```bash
uv run pytest
uv run pytest tests/plotting/ -q          # subset
uv run pytest --cov=scutils --cov-report=html
```

### Adding / removing dependencies

```bash
uv add scanpy anndata                     # runtime dependency
uv add --dev pytest pytest-cov ruff       # dev-only
uv add --optional docs sphinx furo        # optional group
uv remove some-package
```

### The lockfile (`uv.lock`)

Always commit `uv.lock`.  On CI use `uv sync --frozen` to install exactly what the lockfile specifies.

---

## Documentation — Sphinx

Documentation is generated with [Sphinx](https://www.sphinx-doc.org/) using the
[Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) extension to parse Google-style docstrings.

### Building locally

```bash
uv run sphinx-build docs/ docs/_build/html
# or
cd docs && make html
```

Open `docs/_build/html/index.html` in a browser to preview.

### Adding new pages

1. Create or edit `docs/<module>.rst` with `.. automodule:: scutils.<module>` directives.
2. Add the new page to the `toctree` in `docs/index.rst`.
3. Rebuild.

### Deployment to GitHub Pages

See `docs/deployment.md` for step-by-step instructions on deploying the built HTML to GitHub Pages via the `gh-pages` branch using the `docs.yml` GitHub Actions workflow.
