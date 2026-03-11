# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] – unreleased

### Added

- Initial package release as `scutils` (Python ≥ 3.11, hatchling build system).
- `scutils.plotting`: embedding multiplots, boxplots, dotplots, heatmaps,
  volcano plot, density embedding plots.
- `scutils.preprocessing`: zero-filling concatenation utilities.
- `scutils.tools`: iterative subclustering, label renaming, spatial cluster
  splitting.
- Shared `_resolve_palette` helper extracted to `scutils.plotting._utils`.
- Full test suite with pytest.
- Sphinx documentation with furo theme.
- GitHub Actions CI/CD for tests, linting, and docs deployment.
