import math

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from statbelt import ExperimentalHarness
from statbelt.exceptions import ValidationError


def test_data_rejects_infinite_feature_values() -> None:
    X = np.array(
        [
            [0.1, 1.0],
            [0.2, np.inf],
            [0.3, 0.5],
            [0.4, 0.7],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 1], dtype=int)

    with pytest.raises(ValidationError, match="infinity|too large"):
        ExperimentalHarness().data(X, y)


def test_constant_columns_still_produce_finite_metrics(tmp_path) -> None:
    X = np.column_stack(
        [
            np.ones(80, dtype=float),
            np.linspace(-0.5, 0.5, 80, dtype=float),
            np.full(80, 3.14, dtype=float),
        ]
    )
    y = np.array([0, 1] * 40, dtype=int)

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("accuracy", "log_loss")
        .design(cv=4, random_state=12)
        .inference(alpha=0.1, bootstrap_resamples=120)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    intervals = report.models[0].metrics
    for interval in intervals.values():
        assert math.isfinite(interval.point_estimate)
        assert math.isfinite(interval.ci_low)
        assert math.isfinite(interval.ci_high)


def test_extreme_imbalance_requires_minority_support_per_fold(tmp_path) -> None:
    X = np.arange(120, dtype=float).reshape(60, 2)
    y = np.array([0] * 58 + [1, 1], dtype=int)
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("dummy", DummyClassifier(strategy="most_frequent")))
        .metrics("accuracy")
        .design(cv=3, random_state=7)
    )

    with pytest.raises(ValidationError, match="Each class must have at least cv samples"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_tiny_valid_dataset_runs_end_to_end(tmp_path) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("dummy", DummyClassifier(strategy="most_frequent")))
        .metrics("accuracy")
        .design(cv=2, random_state=5)
        .inference(alpha=0.1, bootstrap_resamples=40)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    interval = report.models[0].metrics["accuracy"]
    assert 0.0 <= interval.point_estimate <= 1.0
    assert 0.0 <= interval.ci_low <= 1.0
    assert 0.0 <= interval.ci_high <= 1.0
