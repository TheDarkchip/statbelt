# statbelt

[![PyPI version](https://img.shields.io/pypi/v/statbelt.svg)](https://pypi.org/project/statbelt/)
[![Python versions](https://img.shields.io/pypi/pyversions/statbelt.svg)](https://pypi.org/project/statbelt/)
[![License](https://img.shields.io/pypi/l/statbelt.svg)](LICENSE)
[![Release workflow](https://github.com/TheDarkchip/statbelt/actions/workflows/release.yml/badge.svg)](https://github.com/TheDarkchip/statbelt/actions/workflows/release.yml)

`statbelt` is a strict experimental harness for reproducible, statistically aware
model evaluation in Python.

Status: **Alpha** (APIs may evolve).  
Supported Python: **3.11+**.

## Installation

Install from PyPI:

```bash
pip install statbelt
```

For local development:

```bash
uv sync --all-groups
```

## Quick Start

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from statbelt import ExperimentalHarness

X, y = make_classification(n_samples=120, random_state=21)

report = (
    ExperimentalHarness()
    .data(X, y)
    .task("binary_classification")
    .compare(
        ("logreg", LogisticRegression(max_iter=500)),
        ("rf", RandomForestClassifier(n_estimators=25, random_state=21)),
    )
    .metrics("accuracy", "roc_auc", "log_loss")
    .design(cv=5, random_state=42)
    .inference(alpha=0.05, bootstrap_resamples=2000)
    .fasten("statbelt.lock.json")
    .evaluate()
)

print(report.summary())
```

## Core Features

- `ExperimentalHarness` builder-style API for binary classification comparisons.
- Deterministic stratified k-fold evaluation with shared folds across models.
- Bootstrap confidence intervals over fold-level metrics.
- Lock artifact output (`statbelt.lock.json`) with config and split indices.
- Strict staged workflow: configure -> `fasten()` -> `evaluate()`.

## Supported Task and Metrics

Supported task:

- `binary_classification`

Supported metrics:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `log_loss`

Validation is fail-fast. For example:

- `log_loss` requires `predict_proba`.
- `roc_auc` requires `predict_proba` or `decision_function`.

## CLI Smoke Check

```bash
statbelt
```

Expected output:

```text
Hello from statbelt!
```

## Development

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest
```

For release operations (tagging, TestPyPI gate, PyPI publish), see `RELEASING.md`.

## Current Limits

- Binary classification only.
- Confidence intervals only (no pairwise hypothesis tests yet).
- Python API is the primary surface in v0; CLI remains intentionally minimal.

## License

This project is licensed under the GNU Affero General Public License, version 3
or later (`AGPL-3.0-or-later`). See `LICENSE`.
