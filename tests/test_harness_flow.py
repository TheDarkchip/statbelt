import pytest
import numpy as np
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


def test_inference_rejects_non_numeric_alpha() -> None:
    with pytest.raises(ValidationError, match="alpha must be a number"):
        ExperimentalHarness().inference(alpha="0.1")  # type: ignore[arg-type]
