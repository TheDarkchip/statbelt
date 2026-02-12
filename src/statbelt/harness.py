from __future__ import annotations

import copy
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from statbelt.exceptions import ConfigurationError, StateError, ValidationError
from statbelt.metrics import (
    compute_metric,
    metric_requires_probability,
    metric_requires_score,
    validate_estimator_for_metrics,
    validate_metric_names,
)
from statbelt.report import EvaluationReport, MetricInterval, ModelReport


class ExperimentalHarness:
    """Builder-style API for reproducible binary classification evaluation."""

    _TASK_BINARY_CLASSIFICATION = "binary_classification"
    _STATE_CONFIGURING = "configuring"
    _STATE_FASTENED = "fastened"
    _STATE_EVALUATED = "evaluated"
    _MAX_RANDOM_STATE = int(np.iinfo(np.uint32).max)

    def __init__(self) -> None:
        self._X: NDArray[np.generic] | None = None
        self._y: NDArray[np.generic] | None = None
        self._task: str | None = None
        self._models: list[tuple[str, object]] = []
        self._metric_names: tuple[str, ...] = ()
        self._cv: int = 5
        self._random_state: int = 42
        self._alpha: float = 0.05
        self._bootstrap_resamples: int = 2000
        self._splits: list[tuple[list[int], list[int]]] | None = None
        self._lock_path: str | None = None
        self._state: str = self._STATE_CONFIGURING

    def data(self, X: object, y: object) -> Self:
        self._ensure_configuring()
        X_array = np.asarray(X)
        y_array = np.asarray(y)

        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        if X_array.ndim < 2:
            raise ValidationError("X must be array-like with at least one feature column.")
        if X_array.shape[1] < 1:
            raise ValidationError("X must be array-like with at least one feature column.")
        if y_array.ndim != 1:
            raise ValidationError("y must be a one-dimensional array-like.")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValidationError("X and y must have matching sample counts.")
        if X_array.shape[0] < 2:
            raise ValidationError("At least two samples are required.")

        self._X = X_array
        self._y = y_array
        return self

    def task(self, task_name: str) -> Self:
        self._ensure_configuring()
        if task_name != self._TASK_BINARY_CLASSIFICATION:
            raise ValidationError(
                f"Only '{self._TASK_BINARY_CLASSIFICATION}' is supported in v0."
            )
        self._task = task_name
        return self

    def compare(self, *models: tuple[str, object]) -> Self:
        self._ensure_configuring()
        if not models:
            raise ValidationError("At least one model must be provided to compare().")

        seen_model_names: set[str] = set()
        normalized_models: list[tuple[str, object]] = []
        for model in models:
            if not isinstance(model, tuple) or len(model) != 2:
                raise ValidationError(
                    "Each model entry must be a tuple of (model_name, estimator)."
                )
            model_name, estimator = model
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValidationError("Model names must be non-empty strings.")
            if model_name in seen_model_names:
                raise ValidationError(f"Duplicate model name '{model_name}' is not allowed.")
            seen_model_names.add(model_name)
            if not callable(getattr(estimator, "fit", None)) or not callable(
                getattr(estimator, "predict", None)
            ):
                raise ValidationError(
                    f"Estimator '{model_name}' must implement callable fit and predict methods."
                )
            normalized_models.append((model_name, estimator))

        self._models = normalized_models
        return self

    def metrics(self, *metric_names: str) -> Self:
        self._ensure_configuring()
        self._metric_names = validate_metric_names(metric_names)
        return self

    def design(self, cv: int = 5, random_state: int = 42) -> Self:
        self._ensure_configuring()
        if isinstance(cv, bool) or not isinstance(cv, Integral):
            raise ValidationError("cv must be an integer.")
        cv_int = int(cv)
        if cv_int < 2:
            raise ValidationError("cv must be at least 2.")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise ValidationError("random_state must be an integer.")
        random_state_int = int(random_state)
        if not 0 <= random_state_int <= self._MAX_RANDOM_STATE:
            raise ValidationError(
                f"random_state must be between 0 and {self._MAX_RANDOM_STATE}."
            )
        self._cv = cv_int
        self._random_state = random_state_int
        return self

    def inference(self, alpha: float = 0.05, bootstrap_resamples: int = 2000) -> Self:
        self._ensure_configuring()
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise ValidationError("alpha must be a number.")
        alpha_float = float(alpha)
        if not 0 < alpha_float < 1:
            raise ValidationError("alpha must be between 0 and 1.")
        if isinstance(bootstrap_resamples, bool) or not isinstance(
            bootstrap_resamples, Integral
        ):
            raise ValidationError("bootstrap_resamples must be an integer.")
        if bootstrap_resamples < 1:
            raise ValidationError("bootstrap_resamples must be at least 1.")
        self._alpha = alpha_float
        self._bootstrap_resamples = int(bootstrap_resamples)
        return self

    def fasten(self, lock_path: str = "statbelt.lock.json") -> Self:
        self._ensure_configuring()
        self._validate_configuration()
        self._X, self._y = self._snapshot_data()
        self._models = self._snapshot_models()
        self._splits = self._build_splits()
        self._write_lockfile(lock_path=lock_path, splits=self._splits)
        self._lock_path = lock_path
        self._state = self._STATE_FASTENED
        return self

    def evaluate(self) -> EvaluationReport:
        self._ensure_ready_for_evaluation()
        assert self._X is not None
        assert self._y is not None
        assert self._task is not None
        assert self._splits is not None

        class_labels = np.unique(self._y)
        positive_label = class_labels[-1]
        needs_probability = any(
            metric_requires_probability(metric_name) for metric_name in self._metric_names
        )
        needs_score = any(metric_requires_score(metric_name) for metric_name in self._metric_names)
        model_reports: list[ModelReport] = []

        for model_index, (model_name, estimator) in enumerate(self._models):
            fold_metrics: dict[str, list[float]] = {
                metric_name: [] for metric_name in self._metric_names
            }
            for train_indices, test_indices in self._splits:
                train_idx = np.asarray(train_indices, dtype=int)
                test_idx = np.asarray(test_indices, dtype=int)
                X_train = self._X[train_idx]
                y_train = self._y[train_idx]
                X_test = self._X[test_idx]
                y_test = self._y[test_idx]

                fitted_estimator = self._clone_estimator(estimator)
                fitted_estimator.fit(X_train, y_train)

                y_pred = np.asarray(fitted_estimator.predict(X_test))
                y_proba = None
                y_score = None
                if needs_probability or needs_score:
                    y_proba = self._predict_proba_or_none(
                        fitted_estimator,
                        X_test,
                        class_labels=class_labels,
                    )
                if needs_score:
                    y_score = self._score_for_binary_auc(
                        estimator=fitted_estimator,
                        X_test=X_test,
                        y_proba=y_proba,
                        class_labels=class_labels,
                    )

                for metric_name in self._metric_names:
                    metric_value = compute_metric(
                        metric_name,
                        y_test,
                        y_pred=y_pred,
                        y_score=y_score,
                        y_proba=y_proba,
                        positive_label=positive_label,
                        class_labels=class_labels,
                    )
                    fold_metrics[metric_name].append(metric_value)

            metric_intervals: dict[str, MetricInterval] = {}
            for metric_index, metric_name in enumerate(self._metric_names):
                metric_values = np.asarray(fold_metrics[metric_name], dtype=float)
                ci_low, ci_high = _bootstrap_mean_interval(
                    metric_values,
                    alpha=self._alpha,
                    n_resamples=self._bootstrap_resamples,
                    random_state=_derive_seed(
                        self._random_state, model_index=model_index, metric_index=metric_index
                    ),
                )
                metric_intervals[metric_name] = MetricInterval(
                    point_estimate=float(metric_values.mean()),
                    ci_low=ci_low,
                    ci_high=ci_high,
                    alpha=self._alpha,
                )
            model_reports.append(ModelReport(model_name=model_name, metrics=metric_intervals))

        self._state = self._STATE_EVALUATED
        return EvaluationReport(
            task=self._task,
            cv=self._cv,
            random_state=self._random_state,
            bootstrap_resamples=self._bootstrap_resamples,
            models=model_reports,
            splits=_copy_splits(self._splits),
        )

    def _ensure_configuring(self) -> None:
        if self._state != self._STATE_CONFIGURING:
            raise StateError(
                "Configuration is immutable after fasten(). Create a new harness to reconfigure."
            )

    def _ensure_ready_for_evaluation(self) -> None:
        if self._state not in {self._STATE_FASTENED, self._STATE_EVALUATED}:
            raise StateError("Call fasten() before evaluate().")

    def _validate_configuration(self) -> None:
        if self._X is None or self._y is None:
            raise ConfigurationError("data(X, y) must be configured before fasten().")
        if self._task is None:
            raise ConfigurationError("task(...) must be configured before fasten().")
        if not self._models:
            raise ConfigurationError("compare(...) must be configured before fasten().")
        if not self._metric_names:
            raise ConfigurationError("metrics(...) must be configured before fasten().")
        if self._task != self._TASK_BINARY_CLASSIFICATION:
            raise ConfigurationError(
                f"Only '{self._TASK_BINARY_CLASSIFICATION}' is supported in v0."
            )

        class_labels, counts = np.unique(self._y, return_counts=True)
        if class_labels.size != 2:
            raise ValidationError("binary_classification task requires exactly two classes.")
        if counts.min() < self._cv:
            raise ValidationError(
                "Each class must have at least cv samples for stratified k-fold."
            )
        if self._X.shape[0] < self._cv:
            raise ValidationError("Number of samples must be at least cv.")

        for model_name, estimator in self._models:
            validate_estimator_for_metrics(estimator, model_name, self._metric_names)

    def _build_splits(self) -> list[tuple[list[int], list[int]]]:
        assert self._X is not None
        assert self._y is not None
        splitter = StratifiedKFold(
            n_splits=self._cv,
            shuffle=True,
            random_state=self._random_state,
        )
        return [
            (train_idx.tolist(), test_idx.tolist())
            for train_idx, test_idx in splitter.split(self._X, self._y)
        ]

    def _write_lockfile(
        self,
        *,
        lock_path: str,
        splits: list[tuple[list[int], list[int]]],
    ) -> None:
        payload = {
            "task": self._task,
            "metrics": list(self._metric_names),
            "cv": self._cv,
            "random_state": self._random_state,
            "alpha": self._alpha,
            "bootstrap_resamples": self._bootstrap_resamples,
            "models": [model_name for model_name, _ in self._models],
            "splits": [
                {"train": train_indices, "test": test_indices}
                for train_indices, test_indices in splits
            ],
        }
        lockfile_path = Path(lock_path)
        lockfile_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _predict_proba_or_none(
        self,
        estimator: object,
        X_test: NDArray[np.generic],
        *,
        class_labels: NDArray[np.generic],
    ) -> NDArray[np.float64] | None:
        predict_proba = getattr(estimator, "predict_proba", None)
        if callable(predict_proba):
            probabilities = np.asarray(predict_proba(X_test), dtype=float)
            return _align_probability_columns(
                estimator=estimator,
                y_proba=probabilities,
                class_labels=class_labels,
            )
        return None

    def _score_for_binary_auc(
        self,
        *,
        estimator: object,
        X_test: NDArray[np.generic],
        y_proba: NDArray[np.float64] | None,
        class_labels: NDArray[np.generic],
    ) -> NDArray[np.float64] | None:
        if not any(metric_requires_score(metric_name) for metric_name in self._metric_names):
            return None

        if y_proba is not None:
            return y_proba[:, -1]

        decision_function = getattr(estimator, "decision_function", None)
        if not callable(decision_function):
            return None
        decision_values = np.asarray(decision_function(X_test), dtype=float)
        if decision_values.ndim == 1:
            return _align_binary_decision_vector(
                estimator=estimator,
                decision_values=decision_values,
                class_labels=class_labels,
            )
        if decision_values.ndim == 2 and decision_values.shape[1] == 1:
            return _align_binary_decision_vector(
                estimator=estimator,
                decision_values=decision_values[:, 0],
                class_labels=class_labels,
            )
        if decision_values.ndim == 2 and decision_values.shape[1] == 2:
            return _positive_class_decision_score(
                estimator=estimator,
                decision_values=decision_values,
                class_labels=class_labels,
            )
        raise ValidationError(
            "decision_function must return a 1D vector for binary classification."
        )

    @staticmethod
    def _clone_estimator(estimator: object) -> object:
        try:
            return clone(estimator)
        except Exception:
            return copy.deepcopy(estimator)

    def _snapshot_models(self) -> list[tuple[str, object]]:
        return [
            (model_name, self._clone_estimator(estimator))
            for model_name, estimator in self._models
        ]

    def _snapshot_data(self) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
        assert self._X is not None
        assert self._y is not None
        return np.array(self._X, copy=True), np.array(self._y, copy=True)


def _align_probability_columns(
    *,
    estimator: object,
    y_proba: NDArray[np.float64],
    class_labels: NDArray[np.generic],
) -> NDArray[np.float64]:
    if y_proba.ndim == 1:
        if class_labels.size != 2:
            raise ValidationError(
                "1D predict_proba output is only supported for binary classification."
            )
        estimator_classes = _require_estimator_classes(estimator, expected_size=2)

        probability_class = estimator_classes[1]
        if probability_class == class_labels[1]:
            return np.column_stack([1 - y_proba, y_proba])
        if probability_class == class_labels[0]:
            return np.column_stack([y_proba, 1 - y_proba])
        raise ValidationError(
            "Could not map 1D predict_proba output to dataset class labels."
        )
    if y_proba.ndim != 2:
        raise ValidationError("predict_proba output must be a 1D or 2D array.")

    estimator_classes = _require_estimator_classes(estimator, expected_size=y_proba.shape[1])
    if y_proba.shape[1] != class_labels.size:
        raise ValidationError(
            "predict_proba output must include one column per dataset class."
        )

    column_indices: list[int] = []
    for class_label in class_labels:
        class_positions = np.where(estimator_classes == class_label)[0]
        if class_positions.size == 0:
            raise ValidationError(
                "Could not map estimator probability columns to dataset class labels."
            )
        column_indices.append(int(class_positions[0]))
    if len(column_indices) != len(set(column_indices)):
        raise ValidationError(
            "Estimator probability columns map ambiguously to dataset class labels."
        )
    return y_proba[:, column_indices]


def _bootstrap_mean_interval(
    values: NDArray[np.float64],
    *,
    alpha: float,
    n_resamples: int,
    random_state: int,
) -> tuple[float, float]:
    if values.ndim != 1:
        raise ValidationError("Bootstrap values must be one-dimensional.")
    if values.size == 0:
        raise ValidationError("Bootstrap values cannot be empty.")
    if values.size == 1:
        point = float(values[0])
        return point, point

    rng = np.random.default_rng(random_state)
    samples = rng.choice(values, size=(n_resamples, values.size), replace=True)
    sample_means = samples.mean(axis=1)
    ci_low = float(np.quantile(sample_means, alpha / 2))
    ci_high = float(np.quantile(sample_means, 1 - (alpha / 2)))
    return ci_low, ci_high


def _derive_seed(base_seed: int, *, model_index: int, metric_index: int) -> int:
    # Keep bootstrap deterministic without relying on Python hash randomization.
    return int(base_seed + model_index * 1000 + metric_index * 17)


def _positive_class_decision_score(
    *,
    estimator: object,
    decision_values: NDArray[np.float64],
    class_labels: NDArray[np.generic],
) -> NDArray[np.float64]:
    positive_class = class_labels[-1]
    estimator_classes = _require_estimator_classes(estimator, expected_size=2)
    positive_class_positions = np.where(estimator_classes == positive_class)[0]
    if positive_class_positions.size == 0:
        raise ValidationError(
            "Could not map estimator decision-function columns to dataset class labels."
        )
    positive_class_idx = int(positive_class_positions[0])
    return decision_values[:, positive_class_idx]


def _align_binary_decision_vector(
    *,
    estimator: object,
    decision_values: NDArray[np.float64],
    class_labels: NDArray[np.generic],
) -> NDArray[np.float64]:
    if decision_values.ndim != 1:
        raise ValidationError("decision_function output must be one-dimensional.")
    if class_labels.size != 2:
        raise ValidationError("decision_function alignment requires exactly two classes.")

    positive_class = class_labels[-1]
    estimator_classes = _require_estimator_classes(estimator, expected_size=2)

    if estimator_classes[1] == positive_class:
        return decision_values
    if estimator_classes[0] == positive_class:
        return -decision_values
    raise ValidationError(
        "Could not map estimator decision-function scores to dataset class labels."
    )


def _require_estimator_classes(
    estimator: object,
    *,
    expected_size: int,
) -> NDArray[np.generic]:
    estimator_classes_raw = getattr(estimator, "classes_", None)
    if estimator_classes_raw is None:
        raise ValidationError(
            "Estimator must expose classes_ after fit for class-dependent metrics."
        )
    estimator_classes = np.asarray(estimator_classes_raw)
    if estimator_classes.ndim != 1:
        raise ValidationError("Estimator classes_ must be one-dimensional.")
    if estimator_classes.size != expected_size:
        raise ValidationError(
            "Estimator classes_ must match the class-dependent output dimension."
        )
    return estimator_classes


def _copy_splits(
    splits: list[tuple[list[int], list[int]]],
) -> list[tuple[list[int], list[int]]]:
    return [(train.copy(), test.copy()) for train, test in splits]
