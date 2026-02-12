# Repository Guidelines

## Project Structure & Module Organization

- `src/statbelt/` contains the Python package. The current CLI entrypoint is `src/statbelt/__init__.py` via `main()`.
- `pyproject.toml` defines project metadata, dependencies, and the `statbelt` console script.
- `README.md` describes project goals and current status.
- `uv.lock` pins resolved dependency versions for reproducible environments.
- Add tests under `tests/`, mirroring package areas (for example, `tests/test_cli.py`).

## Build, Test, and Development Commands

- `uv sync` installs runtime dependencies.
- `uv sync --all-groups` installs runtime + dev dependencies (`pytest`, `ruff`).
- `uv run statbelt` runs the CLI smoke test (currently prints `Hello from statbelt!`).
- `uv run ruff check .` runs linting.
- `uv run pytest` runs the test suite.
- `uv build` (optional) builds source/wheel distributions.

## Coding Style & Naming Conventions

- Follow PEP 8 with 4-space indentation and explicit type hints for new code.
- Use `snake_case` for modules, files, functions, and variables; use `PascalCase` for classes.
- Keep functions small and behavior-focused; prefer clear names over comments.
- Run `ruff` before opening a PR; fix warnings instead of suppressing them when possible.

## Testing Guidelines

- Use `pytest` for all automated tests.
- Name test files `test_<feature>.py` and test cases `test_<behavior>`.
- Add at least one happy-path and one edge/failure-path test per new feature.
- For CLI changes, assert both exit behavior and stdout/stderr content.
- No coverage gate is configured yet, but new behavior should ship with tests.

## Commit & Pull Request Guidelines

- This repository currently has no commit history; adopt Conventional Commits going forward.
- Example commit messages: `feat: add harness skeleton`, `fix: validate empty input`, `docs: clarify quick start`.
- Keep commits atomic and logically scoped.
- PRs should include: purpose, concise change summary, test evidence (commands run), and linked issues when applicable.

## Security & Configuration Tips

- Target Python `>=3.13` as defined in `pyproject.toml`.
- Do not commit secrets, API keys, or local environment files.
- Keep dependencies minimal and update `uv.lock` when dependency definitions change.
