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
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statbelt import ExperimentalHarness

dataset = load_breast_cancer()
X, y = dataset.data, dataset.target

report = (
    ExperimentalHarness()
    .data(X, y)
    .task("binary_classification")
    .compare(
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=21)),
    )
    .metrics("accuracy", "roc_auc", "log_loss")
    .design(cv=5, random_state=42)
    .inference(alpha=0.05, bootstrap_resamples=2000)
    .fasten("statbelt.lock.json")
    .evaluate()
)

print(report.summary())
```

Sample output:

```text
Task: binary_classification
CV folds: 5
Bootstrap resamples: 2000
Confidence interval: 95%

Model: logreg
  accuracy: 0.9737 (CI 0.9596, 0.9877)
  roc_auc: 0.9953 (CI 0.9902, 0.9990)
  log_loss: 0.0764 (CI 0.0515, 0.1061)

Model: rf
  accuracy: 0.9561 (CI 0.9509, 0.9613)
  roc_auc: 0.9896 (CI 0.9832, 0.9951)
  log_loss: 0.1769 (CI 0.1061, 0.3037)
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

## License

This project is licensed under the GNU Affero General Public License, version 3
or later (`AGPL-3.0-or-later`). See `LICENSE`.
