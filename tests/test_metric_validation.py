import pytest
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from statbelt import ExperimentalHarness
from statbelt.exceptions import ValidationError


def _data() -> tuple[object, object]:
    return make_classification(
        n_samples=70,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )


def test_unsupported_metric_fails_validation() -> None:
    with pytest.raises(ValidationError, match="Unsupported metric"):
        ExperimentalHarness().metrics("not_a_metric")


def test_log_loss_requires_predict_proba(tmp_path) -> None:
    X, y = _data()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("linear_svc", LinearSVC()))
        .metrics("accuracy", "log_loss")
    )

    with pytest.raises(ValidationError, match="predict_proba"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_roc_auc_accepts_decision_function_without_predict_proba(tmp_path) -> None:
    X, y = _data()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("linear_svc", LinearSVC()))
        .metrics("roc_auc")
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
    )

    report = harness.evaluate()
    assert report.models[0].model_name == "linear_svc"


def test_duplicate_metrics_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate metric"):
        ExperimentalHarness().metrics("accuracy", "accuracy")


def test_metrics_rejects_non_string_entries() -> None:
    metric_names = ["accuracy", "f1"]
    with pytest.raises(ValidationError, match="Metric names must be strings"):
        ExperimentalHarness().metrics(metric_names)  # type: ignore[arg-type]


class _ReversedDecisionEstimator:
    def fit(self, X: object, y: object) -> "_ReversedDecisionEstimator":
        self.classes_ = np.array([1, 0], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return (X[:, 0] > 0).astype(int)

    def decision_function(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        signal = X[:, 0]
        # Column order follows classes_ = [1, 0], so positive-class scores are in column 0.
        return np.column_stack([signal, -signal])


def test_roc_auc_maps_decision_function_column_to_positive_class(tmp_path) -> None:
    X = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("reversed_decision", _ReversedDecisionEstimator()))
        .metrics("roc_auc")
        .design(cv=5, random_state=42)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.models[0].metrics["roc_auc"].point_estimate > 0.95


class _ReversedDecisionVectorEstimator:
    def fit(self, X: object, y: object) -> "_ReversedDecisionVectorEstimator":
        self.classes_ = np.array([1, 0], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return (X[:, 0] > 0).astype(int)

    def decision_function(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        # For binary estimators, 1D decision scores correspond to classes_[1].
        # Here classes_[1] is 0, so positive-class (1) scores require inversion.
        return np.where(X[:, 0] > 0, -1.0, 1.0)


def test_roc_auc_maps_1d_decision_scores_to_positive_class(tmp_path) -> None:
    X = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("reversed_decision_vector", _ReversedDecisionVectorEstimator()))
        .metrics("roc_auc")
        .design(cv=5, random_state=42)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.models[0].metrics["roc_auc"].point_estimate > 0.95


class _ReversedProbaEstimator:
    def fit(self, X: object, y: object) -> "_ReversedProbaEstimator":
        self.classes_ = np.array([1, 0], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return (X[:, 0] > 0).astype(int)

    def predict_proba(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        p_positive = np.where(X[:, 0] > 0, 0.99, 0.01)
        # classes_ order is [1, 0], so probability columns are [P(class 1), P(class 0)].
        return np.column_stack([p_positive, 1.0 - p_positive])


def test_log_loss_respects_estimator_class_ordering(tmp_path) -> None:
    X = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("reversed_proba", _ReversedProbaEstimator()))
        .metrics("log_loss")
        .design(cv=5, random_state=42)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.models[0].metrics["log_loss"].point_estimate < 0.05


class _PredictOnlyEstimator:
    def fit(self, X: object, y: object) -> "_PredictOnlyEstimator":
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return (X[:, 0] > 0).astype(int)

    def predict_proba(self, X: object) -> np.ndarray:
        raise RuntimeError("predict_proba should not be called for prediction-only metrics")


def test_prediction_only_metrics_skip_predict_proba(tmp_path) -> None:
    X = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("predict_only", _PredictOnlyEstimator()))
        .metrics("accuracy")
        .design(cv=5, random_state=42)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.models[0].metrics["accuracy"].point_estimate > 0.95


class _ReversedOneDimProbaEstimator:
    def fit(self, X: object, y: object) -> "_ReversedOneDimProbaEstimator":
        self.classes_ = np.array([1, 0], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return (X[:, 0] > 0).astype(int)

    def predict_proba(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        # 1D output represents P(classes_[1]) = P(class 0).
        return np.where(X[:, 0] > 0, 0.01, 0.99)


def test_1d_predict_proba_respects_estimator_class_ordering(tmp_path) -> None:
    X = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("reversed_1d_proba", _ReversedOneDimProbaEstimator()))
        .metrics("roc_auc", "log_loss")
        .design(cv=5, random_state=42)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    metrics = report.models[0].metrics
    assert metrics["roc_auc"].point_estimate > 0.95
    assert metrics["log_loss"].point_estimate < 0.05


class _NoClassesProbaEstimator:
    def fit(self, X: object, y: object) -> "_NoClassesProbaEstimator":
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return (X[:, 0] > 0).astype(int)

    def predict_proba(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        p_positive = np.where(X[:, 0] > 0, 0.99, 0.01)
        return np.column_stack([1.0 - p_positive, p_positive])


def test_class_dependent_metrics_require_estimator_classes_metadata(tmp_path) -> None:
    X = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)

    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("no_classes", _NoClassesProbaEstimator()))
        .metrics("roc_auc", "log_loss")
        .design(cv=5, random_state=42)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
    )

    with pytest.raises(ValidationError, match="must expose classes_"):
        harness.evaluate()


class _BrokenTagsProbaEstimator(BaseEstimator):
    def __sklearn_tags__(self) -> object:
        raise AttributeError("legacy tag implementation issue")

    def fit(self, X: object, y: object) -> "_BrokenTagsProbaEstimator":
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return (X[:, 0] > 0).astype(int)

    def predict_proba(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        p_positive = np.where(X[:, 0] > 0, 0.95, 0.05)
        return np.column_stack([1.0 - p_positive, p_positive])


def test_class_dependent_metrics_ignore_broken_sklearn_tag_introspection(
    tmp_path,
) -> None:
    X = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("broken_tags", _BrokenTagsProbaEstimator()))
        .metrics("roc_auc", "log_loss")
        .design(cv=5, random_state=42)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    metrics = report.models[0].metrics
    assert metrics["roc_auc"].point_estimate > 0.9
    assert metrics["log_loss"].point_estimate < 0.25
