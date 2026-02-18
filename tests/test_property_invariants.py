import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.dummy import DummyClassifier

from statbelt import ExperimentalHarness


@st.composite
def _binary_dataset(draw) -> tuple[np.ndarray, np.ndarray]:
    n_samples = draw(st.integers(min_value=8, max_value=20))
    n_features = draw(st.integers(min_value=1, max_value=4))
    X = draw(
        arrays(
            np.float64,
            (n_samples, n_features),
            elements=st.floats(
                min_value=-50.0,
                max_value=50.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    y = draw(
        arrays(
            np.int64,
            n_samples,
            elements=st.integers(min_value=0, max_value=1),
        )
    )

    count_zero = int(np.count_nonzero(y == 0))
    count_one = y.size - count_zero
    assume(count_zero >= 2 and count_one >= 2)

    return X, y


def _run_dummy_report(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lock_path: str,
):
    return (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("dummy", DummyClassifier(strategy="most_frequent")))
        .metrics("accuracy")
        .design(cv=2, random_state=23)
        .inference(alpha=0.1, bootstrap_resamples=40)
        .fasten(lock_path=lock_path)
        .evaluate()
    )


@settings(max_examples=12, deadline=None)
@given(dataset=_binary_dataset())
def test_fixed_seed_produces_identical_reports(dataset) -> None:
    X, y = dataset
    with TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        first = _run_dummy_report(X, y, lock_path=str(temp_path / "first.lock.json"))
        second = _run_dummy_report(X, y, lock_path=str(temp_path / "second.lock.json"))

    assert first.to_dict() == second.to_dict()


@settings(max_examples=16, deadline=None)
@given(dataset=_binary_dataset())
def test_generated_inputs_keep_accuracy_interval_finite_and_bounded(
    dataset,
) -> None:
    X, y = dataset
    with TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        report = _run_dummy_report(X, y, lock_path=str(temp_path / "statbelt.lock.json"))
    interval = report.models[0].metrics["accuracy"]

    assert math.isfinite(interval.point_estimate)
    assert math.isfinite(interval.ci_low)
    assert math.isfinite(interval.ci_high)
    assert 0.0 <= interval.point_estimate <= 1.0
    assert 0.0 <= interval.ci_low <= interval.ci_high <= 1.0
