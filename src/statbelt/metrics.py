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

TaskName = Literal["binary_classification", "multiclass_classification"]
MetricRequirement = Literal["pred", "score", "proba"]

_TASK_BINARY_CLASSIFICATION: TaskName = "binary_classification"
_TASK_MULTICLASS_CLASSIFICATION: TaskName = "multiclass_classification"


@dataclass(frozen=True)
class MetricSpec:
    requirement: MetricRequirement
    higher_is_better: bool
    scorer: Callable[[NDArray[np.generic], NDArray[np.generic], object | None], float]


@dataclass(frozen=True)
class MulticlassAucConfig:
    multi_class: Literal["ovr", "ovo"]
    average: Literal["macro", "weighted"]


METRIC_REGISTRY: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(accuracy_score(y_true, y_pred)),
    ),
    "precision": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, positive_label: float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
    ),
    "recall": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, positive_label: float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
    ),
    "f1": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, positive_label: float(
            f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
    ),
    "precision_macro": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    ),
    "precision_weighted": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    ),
    "precision_micro": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
    ),
    "recall_macro": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    ),
    "recall_weighted": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    ),
    "recall_micro": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
    ),
    "f1_macro": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    ),
    "f1_weighted": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    ),
    "f1_micro": MetricSpec(
        requirement="pred",
        higher_is_better=True,
        scorer=lambda y_true, y_pred, _positive_label: float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
    ),
    "roc_auc": MetricSpec(
        requirement="score",
        higher_is_better=True,
        scorer=lambda y_true, y_score, _positive_label: float(roc_auc_score(y_true, y_score)),
    ),
    "roc_auc_ovr_macro": MetricSpec(
        requirement="proba",
        higher_is_better=True,
        scorer=lambda _y_true, _y_proba, _positive_label: float("nan"),
    ),
    "roc_auc_ovr_weighted": MetricSpec(
        requirement="proba",
        higher_is_better=True,
        scorer=lambda _y_true, _y_proba, _positive_label: float("nan"),
    ),
    "roc_auc_ovo_macro": MetricSpec(
        requirement="proba",
        higher_is_better=True,
        scorer=lambda _y_true, _y_proba, _positive_label: float("nan"),
    ),
    "roc_auc_ovo_weighted": MetricSpec(
        requirement="proba",
        higher_is_better=True,
        scorer=lambda _y_true, _y_proba, _positive_label: float("nan"),
    ),
    "log_loss": MetricSpec(
        requirement="proba",
        higher_is_better=False,
        scorer=lambda y_true, y_proba, _positive_label: float(sklearn_log_loss(y_true, y_proba)),
    ),
}

_MULTICLASS_AUC_CONFIG: dict[str, MulticlassAucConfig] = {
    "roc_auc_ovr_macro": MulticlassAucConfig(multi_class="ovr", average="macro"),
    "roc_auc_ovr_weighted": MulticlassAucConfig(multi_class="ovr", average="weighted"),
    "roc_auc_ovo_macro": MulticlassAucConfig(multi_class="ovo", average="macro"),
    "roc_auc_ovo_weighted": MulticlassAucConfig(multi_class="ovo", average="weighted"),
}

_BINARY_TASK_METRICS: set[str] = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
}

_MULTICLASS_TASK_METRICS: set[str] = {
    "accuracy",
    "precision_macro",
    "precision_weighted",
    "precision_micro",
    "recall_macro",
    "recall_weighted",
    "recall_micro",
    "f1_macro",
    "f1_weighted",
    "f1_micro",
    "roc_auc_ovr_macro",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_macro",
    "roc_auc_ovo_weighted",
    "log_loss",
}

_MULTICLASS_ALIASES: dict[str, str] = {
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
    "roc_auc": "roc_auc_ovr_macro",
}

_ALL_TASK_METRICS = _BINARY_TASK_METRICS | _MULTICLASS_TASK_METRICS
_ALL_USER_METRICS = _ALL_TASK_METRICS | set(_MULTICLASS_ALIASES.keys())
SUPPORTED_METRICS: tuple[str, ...] = tuple(sorted(_ALL_USER_METRICS))


def _supported_metrics_for_task(task_name: TaskName) -> tuple[str, ...]:
    if task_name == _TASK_BINARY_CLASSIFICATION:
        return tuple(sorted(_BINARY_TASK_METRICS))
    if task_name == _TASK_MULTICLASS_CLASSIFICATION:
        return tuple(sorted(_MULTICLASS_TASK_METRICS | set(_MULTICLASS_ALIASES.keys())))
    raise ValidationError(f"Unsupported task '{task_name}'.")


def resolve_metric_names(metric_names: Sequence[str], *, task_name: TaskName) -> tuple[str, ...]:
    if task_name == _TASK_BINARY_CLASSIFICATION:
        allowed = _BINARY_TASK_METRICS
    elif task_name == _TASK_MULTICLASS_CLASSIFICATION:
        allowed = _MULTICLASS_TASK_METRICS
    else:
        raise ValidationError(f"Unsupported task '{task_name}'.")

    resolved: list[str] = []
    seen: set[str] = set()
    for metric_name in metric_names:
        candidate = _MULTICLASS_ALIASES.get(metric_name, metric_name)
        if task_name == _TASK_BINARY_CLASSIFICATION and candidate != metric_name:
            candidate = metric_name
        if candidate not in allowed:
            supported = ", ".join(_supported_metrics_for_task(task_name))
            raise ValidationError(
                f"Unsupported metric '{metric_name}' for task '{task_name}'. "
                f"Supported metrics: {supported}."
            )
        if candidate in seen:
            raise ValidationError(
                f"Duplicate metric '{metric_name}' is not allowed for task '{task_name}'."
            )
        seen.add(candidate)
        resolved.append(candidate)
    return tuple(resolved)


def validate_metric_names(
    metric_names: Sequence[str],
    *,
    task_name: TaskName | None = None,
) -> tuple[str, ...]:
    if not metric_names:
        raise ValidationError("At least one metric must be provided.")

    normalized: list[str] = []
    seen: set[str] = set()
    for metric_name in metric_names:
        if not isinstance(metric_name, str):
            raise ValidationError("Metric names must be strings.")
        if metric_name not in _ALL_USER_METRICS:
            supported = ", ".join(SUPPORTED_METRICS)
            raise ValidationError(
                f"Unsupported metric '{metric_name}'. Supported metrics: {supported}."
            )
        if metric_name in seen:
            raise ValidationError(f"Duplicate metric '{metric_name}' is not allowed.")
        seen.add(metric_name)
        normalized.append(metric_name)

    if task_name is None:
        return tuple(normalized)
    return resolve_metric_names(normalized, task_name=task_name)


def metric_requires_score(metric_name: str) -> bool:
    return METRIC_REGISTRY[metric_name].requirement == "score"


def metric_requires_probability(metric_name: str) -> bool:
    return METRIC_REGISTRY[metric_name].requirement == "proba"


def metric_higher_is_better(metric_name: str) -> bool:
    return METRIC_REGISTRY[metric_name].higher_is_better


def validate_estimator_for_metrics(
    estimator: object,
    estimator_name: str,
    metric_names: Sequence[str],
    *,
    task_name: TaskName = _TASK_BINARY_CLASSIFICATION,
) -> None:
    resolved_metric_names = resolve_metric_names(metric_names, task_name=task_name)

    fit_method = getattr(estimator, "fit", None)
    predict_method = getattr(estimator, "predict", None)
    if not callable(fit_method) or not callable(predict_method):
        raise ValidationError(
            f"Estimator '{estimator_name}' must implement callable fit and predict methods."
        )

    has_predict_proba = callable(getattr(estimator, "predict_proba", None))
    has_decision_function = callable(getattr(estimator, "decision_function", None))
    for metric_name in resolved_metric_names:
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
    task_name: TaskName = _TASK_BINARY_CLASSIFICATION,
) -> float:
    if metric_name in METRIC_REGISTRY and metric_name not in _ALL_TASK_METRICS:
        resolved_metric_name = metric_name
    else:
        resolved_metric_name = resolve_metric_names((metric_name,), task_name=task_name)[0]
    spec = METRIC_REGISTRY[resolved_metric_name]

    if spec.requirement == "pred":
        return spec.scorer(y_true, y_pred, positive_label)

    if spec.requirement == "score":
        if y_score is None:
            raise ValidationError(f"Metric '{resolved_metric_name}' requires score values.")
        return spec.scorer(y_true, y_score, positive_label)

    if y_proba is None:
        raise ValidationError(f"Metric '{resolved_metric_name}' requires probability values.")
    if resolved_metric_name == "log_loss":
        return float(sklearn_log_loss(y_true, y_proba, labels=class_labels))
    if resolved_metric_name in _MULTICLASS_AUC_CONFIG:
        config = _MULTICLASS_AUC_CONFIG[resolved_metric_name]
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                labels=class_labels,
                multi_class=config.multi_class,
                average=config.average,
            )
        )
    return spec.scorer(y_true, y_proba, positive_label)
