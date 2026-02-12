import math

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from statbelt import ExperimentalHarness


def test_binary_evaluation_returns_metric_intervals(tmp_path) -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=10,
        n_informative=6,
        n_redundant=0,
        random_state=21,
    )
    requested_metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")
    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=500)),
            ("rf", RandomForestClassifier(n_estimators=25, random_state=21)),
        )
        .metrics(*requested_metrics)
        .design(cv=5, random_state=21)
        .inference(alpha=0.05, bootstrap_resamples=250)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.task == "binary_classification"
    assert report.cv == 5
    assert len(report.models) == 2
    assert len(report.splits) == 5

    for model_report in report.models:
        assert tuple(model_report.metrics.keys()) == requested_metrics
        for interval in model_report.metrics.values():
            assert math.isfinite(interval.point_estimate)
            assert math.isfinite(interval.ci_low)
            assert math.isfinite(interval.ci_high)
            assert interval.ci_low <= interval.point_estimate <= interval.ci_high

    summary = report.summary()
    assert "Model: logreg" in summary
    assert "Model: rf" in summary


def test_binary_metrics_support_non_default_positive_label(tmp_path) -> None:
    X, y = make_classification(
        n_samples=110,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=99,
    )
    # Use labels {0, 2}; sklearn defaults pos_label=1 would fail for precision/recall/f1.
    y_non_default = np.where(y == 1, 2, 0)

    report = (
        ExperimentalHarness()
        .data(X, y_non_default)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("precision", "recall", "f1", "roc_auc")
        .design(cv=5, random_state=99)
        .inference(alpha=0.05, bootstrap_resamples=100)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    metric_values = report.models[0].metrics
    assert math.isfinite(metric_values["precision"].point_estimate)
    assert math.isfinite(metric_values["recall"].point_estimate)
    assert math.isfinite(metric_values["f1"].point_estimate)
    assert math.isfinite(metric_values["roc_auc"].point_estimate)
