from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss as sklearn_log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from statbelt.exceptions import ValidationError

MetricRequirement = Literal["pred", "score", "proba"]


@dataclass(frozen=True)
class MetricSpec:
    requirement: MetricRequirement
    scorer: Callable[[NDArray[np.generic], NDArray[np.generic], object | None], float]


METRIC_REGISTRY: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(
        requirement="pred",
        scorer=lambda y_true, y_pred, _positive_label: float(
            accuracy_score(y_true, y_pred)
        ),
    ),
    "precision": MetricSpec(
        requirement="pred",
        scorer=lambda y_true, y_pred, positive_label: float(
            precision_score(
                y_true, y_pred, pos_label=positive_label, zero_division=0
            )
        ),
    ),
    "recall": MetricSpec(
        requirement="pred",
        scorer=lambda y_true, y_pred, positive_label: float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
    ),
    "f1": MetricSpec(
        requirement="pred",
        scorer=lambda y_true, y_pred, positive_label: float(
            f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
    ),
    "roc_auc": MetricSpec(
        requirement="score",
        scorer=lambda y_true, y_score, _positive_label: float(roc_auc_score(y_true, y_score)),
    ),
    "log_loss": MetricSpec(
        requirement="proba",
        scorer=lambda y_true, y_proba, _positive_label: float(
            sklearn_log_loss(y_true, y_proba)
        ),
    ),
}

SUPPORTED_METRICS: tuple[str, ...] = tuple(METRIC_REGISTRY.keys())


def validate_metric_names(metric_names: Sequence[str]) -> tuple[str, ...]:
    if not metric_names:
        raise ValidationError("At least one metric must be provided.")

    normalized: list[str] = []
    seen: set[str] = set()
    for metric_name in metric_names:
        if not isinstance(metric_name, str):
            raise ValidationError("Metric names must be strings.")
        if metric_name not in METRIC_REGISTRY:
            supported = ", ".join(SUPPORTED_METRICS)
            raise ValidationError(
                f"Unsupported metric '{metric_name}'. Supported metrics: {supported}."
            )
        if metric_name in seen:
            raise ValidationError(f"Duplicate metric '{metric_name}' is not allowed.")
        seen.add(metric_name)
        normalized.append(metric_name)

    return tuple(normalized)


def metric_requires_score(metric_name: str) -> bool:
    return METRIC_REGISTRY[metric_name].requirement == "score"


def metric_requires_probability(metric_name: str) -> bool:
    return METRIC_REGISTRY[metric_name].requirement == "proba"


def validate_estimator_for_metrics(
    estimator: object,
    estimator_name: str,
    metric_names: Sequence[str],
) -> None:
    fit_method = getattr(estimator, "fit", None)
    predict_method = getattr(estimator, "predict", None)
    if not callable(fit_method) or not callable(predict_method):
        raise ValidationError(
            f"Estimator '{estimator_name}' must implement callable fit and predict methods."
        )

    has_predict_proba = callable(getattr(estimator, "predict_proba", None))
    has_decision_function = callable(getattr(estimator, "decision_function", None))
    for metric_name in metric_names:
        if metric_requires_probability(metric_name) and not has_predict_proba:
            raise ValidationError(
                f"Estimator '{estimator_name}' must implement predict_proba for '{metric_name}'."
            )
        if metric_requires_score(metric_name) and not (
            has_predict_proba or has_decision_function
        ):
            raise ValidationError(
                f"Estimator '{estimator_name}' must implement predict_proba or "
                f"decision_function for '{metric_name}'."
            )


def compute_metric(
    metric_name: str,
    y_true: NDArray[np.generic],
    *,
    y_pred: NDArray[np.generic],
    y_score: NDArray[np.float64] | None,
    y_proba: NDArray[np.float64] | None,
    positive_label: object,
    class_labels: NDArray[np.generic],
) -> float:
    spec = METRIC_REGISTRY[metric_name]
    if spec.requirement == "pred":
        return spec.scorer(y_true, y_pred, positive_label)
    if spec.requirement == "score":
        if y_score is None:
            raise ValidationError(f"Metric '{metric_name}' requires score values.")
        return spec.scorer(y_true, y_score, positive_label)
    if y_proba is None:
        raise ValidationError(f"Metric '{metric_name}' requires probability values.")
    if metric_name == "log_loss":
        return float(sklearn_log_loss(y_true, y_proba, labels=class_labels))
    return spec.scorer(y_true, y_proba, positive_label)
