import json
from copy import deepcopy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from statbelt import ExperimentalHarness


def _dataset() -> tuple[object, object]:
    return make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=13,
    )


def test_fasten_writes_lockfile_with_expected_shape(tmp_path) -> None:
    X, y = _dataset()
    lock_path = tmp_path / "statbelt.lock.json"

    (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("accuracy", "roc_auc")
        .fasten(lock_path=str(lock_path))
    )

    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["schema_version"] == 3
    assert lock_payload["task"] == "binary_classification"
    assert lock_payload["cv"] == 5
    assert lock_payload["random_state"] == 42
    assert lock_payload["models"] == ["logreg"]
    assert lock_payload["metrics"] == ["accuracy", "roc_auc"]
    assert len(lock_payload["splits"]) == 5
    assert all("train" in split and "test" in split for split in lock_payload["splits"])


def test_multiclass_lockfile_serializes_resolved_metrics(tmp_path) -> None:
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=55,
    )
    lock_path = tmp_path / "multiclass.lock.json"

    (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(("logreg", LogisticRegression(max_iter=500)))
        .metrics("accuracy", "precision", "roc_auc")
        .fasten(lock_path=str(lock_path))
    )

    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["schema_version"] == 3
    assert lock_payload["task"] == "multiclass_classification"
    assert lock_payload["metrics"] == [
        "accuracy",
        "precision_macro",
        "roc_auc_ovr_macro",
    ]


def test_same_seed_produces_identical_splits(tmp_path) -> None:
    X, y = _dataset()
    harness_a = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("accuracy")
        .design(cv=5, random_state=99)
        .fasten(lock_path=str(tmp_path / "first.lock.json"))
    )
    harness_b = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("accuracy")
        .design(cv=5, random_state=99)
        .fasten(lock_path=str(tmp_path / "second.lock.json"))
    )

    report_a = harness_a.evaluate()
    report_b = harness_b.evaluate()
    assert report_a.splits == report_b.splits


def test_mutating_report_splits_does_not_affect_harness_state(tmp_path) -> None:
    X, y = _dataset()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("accuracy")
        .design(cv=5, random_state=99)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
    )

    report_a = harness.evaluate()
    original_splits = deepcopy(report_a.splits)
    report_a.splits[0][0][0] = 999_999

    report_b = harness.evaluate()
    assert report_b.splits == original_splits


def test_to_dict_splits_are_copied(tmp_path) -> None:
    X, y = _dataset()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("accuracy")
        .design(cv=5, random_state=99)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
    )

    report = harness.evaluate()
    original_splits = deepcopy(report.splits)
    payload = report.to_dict()
    payload["splits"][0]["train"][0] = 999_999

    assert report.splits == original_splits


class _ConstantEstimator:
    def __init__(self, constant_label: int = 0) -> None:
        self.constant_label = constant_label

    def fit(self, X: object, y: object) -> "_ConstantEstimator":
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.full(X.shape[0], self.constant_label, dtype=int)


def test_fasten_snapshots_estimators_from_external_mutation(tmp_path) -> None:
    X = np.arange(100, dtype=float).reshape(100, 1)
    y = np.array([0] * 70 + [1] * 30, dtype=int)
    estimator = _ConstantEstimator(constant_label=0)

    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("constant", estimator))
        .metrics("accuracy")
        .design(cv=5, random_state=42)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
    )

    estimator.constant_label = 1
    report = harness.evaluate()

    assert report.models[0].metrics["accuracy"].point_estimate == 0.7


def test_fasten_snapshots_data_from_external_mutation(tmp_path) -> None:
    X = np.arange(100, dtype=float).reshape(100, 1)
    y = np.array([0] * 70 + [1] * 30, dtype=int)

    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("constant", _ConstantEstimator(constant_label=0)))
        .metrics("accuracy")
        .design(cv=5, random_state=42)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
    )

    X[:] = -9999.0
    y[:] = 1
    report = harness.evaluate()

    assert report.models[0].metrics["accuracy"].point_estimate == 0.7
