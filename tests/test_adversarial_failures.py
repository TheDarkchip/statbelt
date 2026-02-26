import numpy as np
import pytest
from sklearn.datasets import make_classification

from statbelt import ExperimentalHarness
from statbelt.exceptions import ValidationError


class _NaNProbabilityEstimator:
    def fit(self, X: object, y: object) -> "_NaNProbabilityEstimator":
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.full((X.shape[0], 2), np.nan, dtype=float)


class _MissingClassesProbabilityEstimator:
    def fit(self, X: object, y: object) -> "_MissingClassesProbabilityEstimator":
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.tile(np.array([0.5, 0.5], dtype=float), (X.shape[0], 1))


class _ThreeColumnProbabilityEstimator:
    def fit(self, X: object, y: object) -> "_ThreeColumnProbabilityEstimator":
        self.classes_ = np.array([0, 1, 2], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.tile(np.array([0.2, 0.3, 0.5], dtype=float), (X.shape[0], 1))


def _binary_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.linspace(-1.0, 1.0, 30, dtype=float).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=120,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=31,
    )


class _MulticlassNaNProbabilityEstimator:
    def fit(self, X: object, y: object) -> "_MulticlassNaNProbabilityEstimator":
        self.classes_ = np.array([0, 1, 2], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.full((X.shape[0], 3), np.nan, dtype=float)


class _TwoColumnMulticlassProbabilityEstimator:
    def fit(self, X: object, y: object) -> "_TwoColumnMulticlassProbabilityEstimator":
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.tile(np.array([0.7, 0.3], dtype=float), (X.shape[0], 1))


def test_nan_predict_proba_raises_explicit_error(tmp_path) -> None:
    X, y = _binary_data()

    with pytest.raises(ValueError, match="NaN"):
        (
            ExperimentalHarness()
            .data(X, y)
            .task("binary_classification")
            .compare(("nan_proba", _NaNProbabilityEstimator()))
            .metrics("log_loss")
            .design(cv=3, random_state=19)
            .inference(alpha=0.1, bootstrap_resamples=60)
            .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
            .evaluate()
        )


def test_multiclass_nan_predict_proba_raises_explicit_error(tmp_path) -> None:
    X, y = _multiclass_data()

    with pytest.raises(ValueError, match="NaN"):
        (
            ExperimentalHarness()
            .data(X, y)
            .task("multiclass_classification")
            .compare(("nan_proba_multiclass", _MulticlassNaNProbabilityEstimator()))
            .metrics("log_loss")
            .design(cv=3, random_state=19)
            .inference(alpha=0.1, bootstrap_resamples=60)
            .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
            .evaluate()
        )


def test_missing_classes_attribute_raises_validation_error(tmp_path) -> None:
    X, y = _binary_data()

    with pytest.raises(ValidationError, match="must expose classes_"):
        (
            ExperimentalHarness()
            .data(X, y)
            .task("binary_classification")
            .compare(("missing_classes", _MissingClassesProbabilityEstimator()))
            .metrics("log_loss")
            .design(cv=3, random_state=19)
            .inference(alpha=0.1, bootstrap_resamples=60)
            .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
            .evaluate()
        )


def test_predict_proba_with_extra_columns_raises_validation_error(tmp_path) -> None:
    X, y = _binary_data()

    with pytest.raises(ValidationError, match="one column per dataset class"):
        (
            ExperimentalHarness()
            .data(X, y)
            .task("binary_classification")
            .compare(("extra_columns", _ThreeColumnProbabilityEstimator()))
            .metrics("log_loss")
            .design(cv=3, random_state=19)
            .inference(alpha=0.1, bootstrap_resamples=60)
            .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
            .evaluate()
        )


def test_multiclass_predict_proba_with_missing_columns_raises_validation_error(tmp_path) -> None:
    X, y = _multiclass_data()

    with pytest.raises(ValidationError, match="one column per dataset class"):
        (
            ExperimentalHarness()
            .data(X, y)
            .task("multiclass_classification")
            .compare(("two_columns", _TwoColumnMulticlassProbabilityEstimator()))
            .metrics("log_loss")
            .design(cv=3, random_state=19)
            .inference(alpha=0.1, bootstrap_resamples=60)
            .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
            .evaluate()
        )
