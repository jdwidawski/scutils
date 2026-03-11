# Deploying documentation to GitHub Pages

This guide explains how to publish the Sphinx-generated HTML to the
`gh-pages` branch of your repository so it is served at
`https://<your-org>.github.io/single_cell_utilities/`.

---

## Automated deployment (recommended)

The repository includes a GitHub Actions workflow at
`.github/workflows/docs.yml` that automatically builds and deploys the
documentation on every push to `main`.

### One-time setup

1. **Enable GitHub Pages** in your repository settings:
   - Go to **Settings → Pages**.
   - Under *Source*, select **Deploy from a branch**.
   - Choose branch `gh-pages`, folder `/ (root)`.
   - Click **Save**.

2. **Grant Actions write permissions**:
   - Go to **Settings → Actions → General**.
   - Under *Workflow permissions*, select **Read and write permissions**.
   - Click **Save**.

After these two steps, every push to `main` will trigger the workflow and
update your documentation automatically within a few minutes.

---

## Manual deployment

If you want to deploy manually (e.g. for a one-off preview):

```bash
# 1. Install docs dependencies
uv sync --extra docs

# 2. Build the HTML
uv run sphinx-build docs/ docs/_build/html

# 3. Push to gh-pages branch (using ghp-import)
uv run pip install ghp-import
uv run ghp-import -n -p -f docs/_build/html
```

The `-n` flag adds a `.nojekyll` file (required so GitHub Pages serves the
Sphinx `_static/` assets correctly).  The `-p` flag pushes immediately.

---

## Checking the result

After deployment completes (usually 1–2 minutes), visit:

```text
https://<your-org>.github.io/single_cell_utilities/
```

Replace `<your-org>` with your GitHub username or organisation name.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| 404 on the Pages URL | `gh-pages` branch not created | Run a manual deployment or wait for the CI workflow to complete |
| CSS / JS assets not loading | Missing `.nojekyll` file | Ensure `ghp-import -n` was used, or add an empty `.nojekyll` to `gh-pages` root |
| Workflow fails on `permissions` | Insufficient Actions permissions | Enable *Read and write permissions* under Settings → Actions |
| `autodoc` import errors | Package not installed in the build env | The `docs.yml` workflow runs `uv sync --extra docs` which installs the package; check that `pyproject.toml` lists all runtime deps |
