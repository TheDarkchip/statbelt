import math
import numpy as np

from statbelt.harness import _bootstrap_mean_interval


def test_bootstrap_interval_is_deterministic_for_seed() -> None:
    values = np.array([0.75, 0.8, 0.82, 0.9, 0.95], dtype=float)

    first = _bootstrap_mean_interval(
        values,
        alpha=0.05,
        n_resamples=500,
        random_state=42,
    )
    second = _bootstrap_mean_interval(
        values,
        alpha=0.05,
        n_resamples=500,
        random_state=42,
    )

    assert first == second


def test_bootstrap_interval_single_value_collapses_to_point() -> None:
    point = 0.731
    ci_low, ci_high = _bootstrap_mean_interval(
        np.array([point], dtype=float),
        alpha=0.05,
        n_resamples=500,
        random_state=42,
    )

    assert ci_low == point
    assert ci_high == point


def test_bootstrap_interval_is_finite_and_ordered() -> None:
    ci_low, ci_high = _bootstrap_mean_interval(
        np.array([0.4, 0.5, 0.6, 0.7, 0.8], dtype=float),
        alpha=0.1,
        n_resamples=1000,
        random_state=21,
    )

    assert math.isfinite(ci_low)
    assert math.isfinite(ci_high)
    assert ci_low <= ci_high

