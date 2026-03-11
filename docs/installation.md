# Installation

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip ≥ 23

## From source (development)

```bash
git clone https://github.com/<your-org>/single_cell_utilities.git
cd single_cell_utilities
uv sync
```

This creates a `.venv/` in the project root and installs all runtime
dependencies into it.

## Optional dependency groups

```bash
uv sync --extra dev   # pytest, pytest-cov, ruff
uv sync --extra docs  # sphinx, furo, myst-parser, sphinx-autodoc-typehints
```

## Via pip (without uv)

```bash
pip install -e ".[dev]"
```
