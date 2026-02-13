import pytest
import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from statbelt import ExperimentalHarness
from statbelt.exceptions import ConfigurationError, StateError, ValidationError


def _binary_data() -> tuple[object, object]:
    return make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=123,
    )


def test_evaluate_requires_fasten() -> None:
    harness = ExperimentalHarness()
    with pytest.raises(StateError, match="fasten"):
        harness.evaluate()


def test_fasten_requires_complete_configuration() -> None:
    X, y = _binary_data()
    harness = ExperimentalHarness().data(X, y).task("binary_classification")
    with pytest.raises(ConfigurationError, match="compare"):
        harness.fasten()


def test_fasten_requires_data_configuration() -> None:
    harness = (
        ExperimentalHarness()
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=300)))
        .metrics("accuracy")
    )

    with pytest.raises(ConfigurationError, match="data\\(X, y\\)"):
        harness.fasten()


def test_fasten_requires_task_configuration() -> None:
    X, y = _binary_data()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .compare(("logreg", LogisticRegression(max_iter=300)))
        .metrics("accuracy")
    )

    with pytest.raises(ConfigurationError, match="task"):
        harness.fasten()


def test_fasten_requires_metrics_configuration() -> None:
    X, y = _binary_data()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=300)))
    )

    with pytest.raises(ConfigurationError, match="metrics"):
        harness.fasten()


def test_configuration_is_immutable_after_fasten(tmp_path) -> None:
    X, y = _binary_data()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=300)))
        .metrics("accuracy")
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
    )

    with pytest.raises(StateError, match="immutable"):
        harness.metrics("f1")


def test_invalid_task_name_raises() -> None:
    with pytest.raises(ValidationError, match="binary_classification"):
        ExperimentalHarness().task("regression")


def test_data_rejects_zero_feature_matrix() -> None:
    X = np.empty((10, 0))
    y = np.array([0, 1] * 5)
    with pytest.raises(ValidationError, match="at least one feature column"):
        ExperimentalHarness().data(X, y)


def test_data_rejects_non_1d_targets() -> None:
    X = np.arange(12).reshape(6, 2)
    y = np.array([[0], [1], [0], [1], [0], [1]])
    with pytest.raises(ValidationError, match="one-dimensional array-like"):
        ExperimentalHarness().data(X, y)


def test_data_accepts_nan_values() -> None:
    X = np.array(
        [
            [0.1, np.nan],
            [0.3, 1.2],
            [0.5, np.nan],
            [0.7, 1.0],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 1], dtype=int)

    harness = ExperimentalHarness().data(X, y)
    assert harness is not None


def test_data_accepts_sparse_feature_matrices(tmp_path) -> None:
    X, y = _binary_data()
    # Use COO input to verify .data() normalizes to an indexable sparse format.
    X_sparse = sparse.coo_matrix(X)

    report = (
        ExperimentalHarness()
        .data(X_sparse, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=300)))
        .metrics("accuracy")
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.models[0].metrics["accuracy"].point_estimate >= 0.0


def test_data_wraps_check_xy_typeerror_as_validationerror(monkeypatch) -> None:
    def _raise_typeerror(*args: object, **kwargs: object) -> tuple[object, object]:
        raise TypeError("simulated type problem")

    monkeypatch.setattr("statbelt.harness.check_X_y", _raise_typeerror)
    X = np.array([[0.1], [0.2], [0.3]], dtype=float)
    y = np.array([0, 1, 0], dtype=int)

    with pytest.raises(ValidationError, match="simulated type problem"):
        ExperimentalHarness().data(X, y)


def test_fasten_rejects_non_binary_targets(tmp_path) -> None:
    X = np.arange(30, dtype=float).reshape(15, 2)
    y = np.array([0, 1, 2] * 5)
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=300)))
        .metrics("accuracy")
    )

    with pytest.raises(ValidationError, match="exactly two classes"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_design_rejects_none_random_state() -> None:
    with pytest.raises(ValidationError, match="random_state must be an integer"):
        ExperimentalHarness().design(random_state=None)  # type: ignore[arg-type]


def test_design_rejects_out_of_range_random_state() -> None:
    with pytest.raises(ValidationError, match="random_state must be between"):
        ExperimentalHarness().design(random_state=-1)
    with pytest.raises(ValidationError, match="random_state must be between"):
        ExperimentalHarness().design(random_state=2**32)


def test_design_rejects_non_integer_cv() -> None:
    with pytest.raises(ValidationError, match="cv must be an integer"):
        ExperimentalHarness().design(cv=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="cv must be an integer"):
        ExperimentalHarness().design(cv=True)  # type: ignore[arg-type]


def test_inference_rejects_non_integer_bootstrap_resamples() -> None:
    with pytest.raises(ValidationError, match="bootstrap_resamples must be an integer"):
        ExperimentalHarness().inference(bootstrap_resamples=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="bootstrap_resamples must be an integer"):
        ExperimentalHarness().inference(bootstrap_resamples=True)  # type: ignore[arg-type]


def test_inference_rejects_bootstrap_resamples_below_two() -> None:
    with pytest.raises(ValidationError, match="bootstrap_resamples must be at least 2"):
        ExperimentalHarness().inference(bootstrap_resamples=1)
    with pytest.raises(ValidationError, match="bootstrap_resamples must be at least 2"):
        ExperimentalHarness().inference(bootstrap_resamples=0)


def test_inference_rejects_non_numeric_alpha() -> None:
    with pytest.raises(ValidationError, match="alpha must be a number"):
        ExperimentalHarness().inference(alpha="0.1")  # type: ignore[arg-type]
