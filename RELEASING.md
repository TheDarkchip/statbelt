# Releasing `statbelt`

This project publishes with GitHub Actions Trusted Publishing:

- Trigger: push a tag matching `v*` (for example, `v0.1.1`)
- Gate: publish to TestPyPI first, verify install, then publish to PyPI
- Auth: PyPI OIDC trusted publisher (no long-lived API token)

## One-time Setup

1. Create the `statbelt` project on:
   - TestPyPI: <https://test.pypi.org/>
   - PyPI: <https://pypi.org/>
2. Add a trusted publisher on both indexes:
   - Owner: `TheDarkchip`
   - Repository: `statbelt`
   - Workflow path: `.github/workflows/release.yml`

## Preflight

Run before tagging:

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest
uv build
uvx twine check dist/*
```

## Release Steps

1. Ensure `pyproject.toml` has the intended version (for example, `0.1.1`).
2. Commit all changes.
3. Create and push a tag:

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

4. Watch the `release` workflow in GitHub Actions:
   - `checks` and `build` must pass
   - package must publish to TestPyPI
   - TestPyPI install verification must pass
   - package then publishes to PyPI

## Post-release Verification

```bash
python -m venv /tmp/statbelt-release-check
source /tmp/statbelt-release-check/bin/activate
python -m pip install --upgrade pip
python -m pip install statbelt==0.1.1
python -c "import statbelt; print(statbelt.__all__)"
deactivate
```

## If Something Fails

- If failure happens before PyPI publish, fix and push a new tag with a bumped version.
- If PyPI publish already happened, do not overwrite files; cut a new release (for example, `0.1.2`).
