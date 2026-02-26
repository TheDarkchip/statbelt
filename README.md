# statbelt

[![PyPI version](https://img.shields.io/pypi/v/statbelt.svg)](https://pypi.org/project/statbelt/)
[![Python versions](https://img.shields.io/pypi/pyversions/statbelt.svg)](https://pypi.org/project/statbelt/)
[![License](https://img.shields.io/pypi/l/statbelt.svg)](LICENSE)
[![CI workflow](https://github.com/TheDarkchip/statbelt/actions/workflows/ci.yml/badge.svg)](https://github.com/TheDarkchip/statbelt/actions/workflows/ci.yml)
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
    .design(cv=5, cv_repeats=2, random_state=42)
    .inference(alpha=0.05, bootstrap_resamples=2000)
    .compare_inference(method="paired_bootstrap", alternative="two-sided")
    .multiplicity(method="holm", family="global")
    .practical_significance(accuracy=0.005, roc_auc=0.002, log_loss=0.01)
    .baseline("logreg")
    .guardrails(min_improvement={"accuracy": 0.002}, confidence=0.95)
    .fasten("statbelt.lock.json")
    .evaluate()
)

print(report.summary())
```

Sample output:

```text
Task: binary_classification
CV folds: 5
CV repeats: 2
Bootstrap resamples: 2000
Confidence interval: 95%

Model: logreg
  accuracy: 0.9754 (CI 0.9666, 0.9833)
  roc_auc: 0.9948 (CI 0.9913, 0.9976)
  log_loss: 0.0775 (CI 0.0603, 0.0975)

Model: rf
  accuracy: 0.9596 (CI 0.9526, 0.9666)
  roc_auc: 0.9903 (CI 0.9860, 0.9941)
  log_loss: 0.1478 (CI 0.1066, 0.2145)

Pairwise comparisons:
  logreg - rf [accuracy]: 0.0158 (CI 0.0070, 0.0255), p_adj=0.002999, reject, practical
  logreg - rf [roc_auc]: 0.0045 (CI 0.0002, 0.0098), p_adj=0.03898, reject, practical
  logreg - rf [log_loss]: -0.0703 (CI -0.1396, -0.0245), p_adj=0.002999, reject, practical

Guardrails: FAIL
  rf vs logreg [accuracy]: FAIL (min 0.0020, CI -0.0263, -0.0070)
```

When pairwise inference and guardrails are configured, `summary()` also includes
pairwise comparison lines and an overall guardrail pass/fail section.

## Multiclass Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from statbelt import ExperimentalHarness

dataset = load_iris()
X, y = dataset.data, dataset.target

report = (
    ExperimentalHarness()
    .data(X, y)
    .task("multiclass_classification")
    .compare(
        ("logreg", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
    )
    .metrics(
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
        "roc_auc_ovo_weighted",
    )
    .design(cv=5, random_state=42)
    .inference(alpha=0.05, bootstrap_resamples=1000)
    .fasten("statbelt.lock.json")
    .evaluate()
)
```

## Core Features

- `ExperimentalHarness` builder-style API for binary and multiclass classification comparisons.
- Deterministic repeated stratified k-fold evaluation with shared folds across models.
- Bootstrap confidence intervals over fold-level metrics.
- Pairwise model inference with paired bootstrap/permutation p-values.
- Multiple-comparison correction (`holm`, `bonferroni`, `fdr_bh`).
- Practical-significance thresholds and baseline guardrail checks.
- Machine-readable exports via `EvaluationReport.to_json()` and `.to_dataframe()`.
- Lock artifact output (`statbelt.lock.json`) with config and split indices.
- Strict staged workflow: configure -> `fasten()` -> `evaluate()`.

## Inference Configuration

Use pairwise inference to compare models directly:

```python
.compare_inference(method="paired_bootstrap", alternative="two-sided")
```

Supported values:

- `method`: `paired_bootstrap`, `permutation`
- `alternative`: `two-sided`, `greater`, `less`

How to choose `alternative`:

- `two-sided`: use when you only care whether A and B differ.
- `greater`: use when your question is “is A better than B?”
- `less`: use when your question is “is A worse than B?”

How to choose `method`:

- `paired_bootstrap`: default practical choice for CI + p-value style comparison.
- `permutation`: exact paired-randomization style null test over fold deltas.

Interpretation details:

- Pairwise rows are always `model_a` vs `model_b`; A/B come from `compare(...)` order.
- `delta` in the report is raw metric-space `model_a - model_b`.
- One-sided p-values are metric-direction normalized, so `greater`/`less` keep the same
  meaning across mixed metrics (for example, both `accuracy` and `log_loss`).

Quick example:

```python
report = (
    ExperimentalHarness()
    .data(X, y)
    .task("binary_classification")
    .compare(("candidate", candidate_model), ("baseline", baseline_model))
    .metrics("accuracy", "log_loss")
    .compare_inference(method="paired_bootstrap", alternative="greater")
    .fasten()
    .evaluate()
)
# Here, p-values answer: \"is candidate better than baseline?\"
```

Control multiple testing with:

```python
.multiplicity(method="holm", family="global")
```

Supported values:

- `method`: `holm`, `bonferroni`, `fdr_bh`
- `family`: `global`, `per_metric`

## Practical Significance and Guardrails

Practical thresholds:

```python
.practical_significance(accuracy=0.005, log_loss=0.01)
```

Guardrails against a baseline:

```python
.baseline("logreg")
.guardrails(min_improvement={"accuracy": 0.002}, confidence=0.95)
```

Rules:

- Threshold values must be finite and non-negative.
- Guardrails require `baseline(...)`.
- Guardrail metrics must also be included in `.metrics(...)`.

## Report and Export API

`evaluate()` returns an `EvaluationReport` with:

- `models`: per-model metric intervals
- `pairwise`: pairwise deltas, CIs, raw/adjusted p-values, practical-significance flags
- `guardrails`: per-check pass/fail and aggregate `overall_pass`
- `splits` and `split_metadata`: deterministic split definitions

Export helpers:

```python
report.to_json("report.json")
report.to_dataframe(kind="models")
report.to_dataframe(kind="pairwise")
```

## Lockfile Schema

`fasten()` writes schema version `3` lockfiles, including:

- design: `cv`, `cv_repeats`, `random_state`
- inference config: `alpha`, `bootstrap_resamples`, `pairwise_inference`
- multiplicity config
- practical-significance and guardrail config
- split indices with repeat/fold metadata

## Supported Task and Metrics

Supported tasks:

- `binary_classification`
- `multiclass_classification`

Binary metrics:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `log_loss`

Multiclass metrics:

- `accuracy`
- `precision_macro`, `precision_weighted`, `precision_micro`
- `recall_macro`, `recall_weighted`, `recall_micro`
- `f1_macro`, `f1_weighted`, `f1_micro`
- `roc_auc_ovr_macro`, `roc_auc_ovr_weighted`
- `roc_auc_ovo_macro`, `roc_auc_ovo_weighted`
- `log_loss`

Task-aware metric aliases:

| Metric name | `binary_classification` | `multiclass_classification` |
| --- | --- | --- |
| `precision` | binary precision | `precision_macro` |
| `recall` | binary recall | `recall_macro` |
| `f1` | binary F1 | `f1_macro` |
| `roc_auc` | binary ROC AUC | `roc_auc_ovr_macro` |

Validation is fail-fast. For example:

- `log_loss` requires `predict_proba`.
- binary `roc_auc` accepts `predict_proba` or `decision_function`.
- multiclass ROC AUC metrics require `predict_proba`.

## Development

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest
```

For release operations (tagging, TestPyPI gate, PyPI publish), see `RELEASING.md`.

## Current Limits

- Classification tasks only (regression is not supported yet).

## License

This project is licensed under the GNU Affero General Public License, version 3
or later (`AGPL-3.0-or-later`). See `LICENSE`.
