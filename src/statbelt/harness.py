from __future__ import annotations

import copy
import json
from dataclasses import replace
from itertools import combinations
from numbers import Integral, Real
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse, spmatrix
from scipy.stats import (
    bootstrap as scipy_bootstrap,
    false_discovery_control as scipy_false_discovery_control,
    permutation_test as scipy_permutation_test,
)
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted

from statbelt.exceptions import ConfigurationError, StateError, ValidationError
from statbelt.metrics import (
    compute_metric,
    metric_higher_is_better,
    metric_requires_probability,
    metric_requires_score,
    resolve_metric_names,
    validate_estimator_for_metrics,
    validate_metric_names,
)
from statbelt.report import (
    EvaluationReport,
    GuardrailCheck,
    GuardrailReport,
    MetricInterval,
    ModelReport,
    PairwiseComparison,
)


class ExperimentalHarness:
    """Builder-style API for reproducible classification evaluation."""

    _TASK_BINARY_CLASSIFICATION = "binary_classification"
    _TASK_MULTICLASS_CLASSIFICATION = "multiclass_classification"
    _SUPPORTED_TASKS = {
        _TASK_BINARY_CLASSIFICATION,
        _TASK_MULTICLASS_CLASSIFICATION,
    }
    _STATE_CONFIGURING = "configuring"
    _STATE_FASTENED = "fastened"
    _STATE_EVALUATED = "evaluated"
    _MAX_RANDOM_STATE = int(np.iinfo(np.uint32).max)
    _PAIRWISE_METHODS = {"paired_bootstrap", "permutation"}
    _PAIRWISE_ALTERNATIVES = {"two-sided", "greater", "less"}
    _MULTIPLICITY_METHODS = {"holm", "bonferroni", "fdr_bh"}
    _MULTIPLICITY_FAMILIES = {"global", "per_metric"}

    def __init__(self) -> None:
        self._X: NDArray[np.generic] | spmatrix | None = None
        self._y: NDArray[np.generic] | None = None
        self._task: str | None = None
        self._models: list[tuple[str, object]] = []
        self._metric_names: tuple[str, ...] = ()
        self._cv: int = 5
        self._cv_repeats: int = 1
        self._random_state: int = 42
        self._alpha: float = 0.05
        self._bootstrap_resamples: int = 2000
        self._splits: list[tuple[list[int], list[int]]] | None = None
        self._split_metadata: list[dict[str, int]] | None = None
        self._pairwise_method: str = "paired_bootstrap"
        self._pairwise_alternative: str = "two-sided"
        self._multiplicity_method: str = "holm"
        self._multiplicity_family: str = "global"
        self._practical_significance: dict[str, float] = {}
        self._baseline_model_name: str | None = None
        self._guardrail_min_improvement: dict[str, float] = {}
        self._guardrail_confidence: float = 0.95
        self._lock_path: str | None = None
        self._state: str = self._STATE_CONFIGURING

    def data(self, X: object, y: object) -> Self:
        self._ensure_configuring()
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValidationError("y must be a one-dimensional array-like.")

        if issparse(X):
            X_features: object = X
        else:
            X_array = np.asarray(X)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            X_features = X_array

        try:
            validated_X, validated_y = check_X_y(
                X_features,
                y_array,
                dtype=None,
                ensure_all_finite="allow-nan",
                ensure_min_samples=2,
                ensure_min_features=1,
                multi_output=False,
                accept_sparse="csr",
            )
        except (TypeError, ValueError) as exc:
            message = str(exc)
            if "0 feature(s)" in message or "Expected 2D array" in message:
                raise ValidationError("X must be array-like with at least one feature column.") from exc
            if "y should be a 1d array" in message:
                raise ValidationError("y must be a one-dimensional array-like.") from exc
            if "inconsistent numbers of samples" in message:
                raise ValidationError("X and y must have matching sample counts.") from exc
            if "minimum of 2 is required" in message:
                raise ValidationError("At least two samples are required.") from exc
            raise ValidationError(message) from exc

        self._X = validated_X
        self._y = np.asarray(validated_y)
        return self

    def task(self, task_name: str) -> Self:
        self._ensure_configuring()
        if task_name not in self._SUPPORTED_TASKS:
            supported = ", ".join(sorted(self._SUPPORTED_TASKS))
            raise ValidationError(
                f"Unsupported task '{task_name}'. Supported tasks: {supported}."
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

    def design(self, cv: int = 5, random_state: int = 42, cv_repeats: int = 1) -> Self:
        self._ensure_configuring()
        if isinstance(cv, bool) or not isinstance(cv, Integral):
            raise ValidationError("cv must be an integer.")
        cv_int = int(cv)
        if cv_int < 2:
            raise ValidationError("cv must be at least 2.")
        if isinstance(cv_repeats, bool) or not isinstance(cv_repeats, Integral):
            raise ValidationError("cv_repeats must be an integer.")
        cv_repeats_int = int(cv_repeats)
        if cv_repeats_int < 1:
            raise ValidationError("cv_repeats must be at least 1.")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise ValidationError("random_state must be an integer.")
        random_state_int = int(random_state)
        if not 0 <= random_state_int <= self._MAX_RANDOM_STATE:
            raise ValidationError(
                f"random_state must be between 0 and {self._MAX_RANDOM_STATE}."
            )
        self._cv = cv_int
        self._cv_repeats = cv_repeats_int
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
        if bootstrap_resamples < 2:
            raise ValidationError("bootstrap_resamples must be at least 2.")
        self._alpha = alpha_float
        self._bootstrap_resamples = int(bootstrap_resamples)
        return self

    def compare_inference(
        self,
        method: str = "paired_bootstrap",
        alternative: str = "two-sided",
    ) -> Self:
        """Configure pairwise model-comparison inference.

        Parameters:
            method:
                - ``paired_bootstrap``: bootstrap over fold-level paired deltas.
                - ``permutation``: paired permutation test over fold-level deltas.
            alternative:
                - ``two-sided``: model A differs from model B.
                - ``greater``: model A is better than model B.
                - ``less``: model A is worse than model B.

        Notes:
            - Pairwise records are emitted as model A vs model B where A/B come from
              the order passed to ``compare(...)``.
            - The reported ``delta`` remains raw metric-space ``model_a - model_b``.
            - One-sided p-values are direction-normalized by metric orientation, so
              ``greater``/``less`` keep the same meaning across mixed metrics
              (for example, ``accuracy`` and ``log_loss``).
        """
        self._ensure_configuring()
        if not isinstance(method, str):
            raise ValidationError("compare_inference method must be a string.")
        if not isinstance(alternative, str):
            raise ValidationError("compare_inference alternative must be a string.")
        if method not in self._PAIRWISE_METHODS:
            supported = ", ".join(sorted(self._PAIRWISE_METHODS))
            raise ValidationError(
                f"compare_inference method '{method}' is not supported. "
                f"Supported methods: {supported}."
            )
        if alternative not in self._PAIRWISE_ALTERNATIVES:
            supported = ", ".join(sorted(self._PAIRWISE_ALTERNATIVES))
            raise ValidationError(
                f"compare_inference alternative '{alternative}' is not supported. "
                f"Supported alternatives: {supported}."
            )
        self._pairwise_method = method
        self._pairwise_alternative = alternative
        return self

    def multiplicity(self, method: str = "holm", family: str = "global") -> Self:
        self._ensure_configuring()
        if not isinstance(method, str):
            raise ValidationError("multiplicity method must be a string.")
        if not isinstance(family, str):
            raise ValidationError("multiplicity family must be a string.")
        if method not in self._MULTIPLICITY_METHODS:
            supported = ", ".join(sorted(self._MULTIPLICITY_METHODS))
            raise ValidationError(
                f"multiplicity method '{method}' is not supported. "
                f"Supported methods: {supported}."
            )
        if family not in self._MULTIPLICITY_FAMILIES:
            supported = ", ".join(sorted(self._MULTIPLICITY_FAMILIES))
            raise ValidationError(
                f"multiplicity family '{family}' is not supported. "
                f"Supported families: {supported}."
            )
        self._multiplicity_method = method
        self._multiplicity_family = family
        return self

    def practical_significance(self, **metric_thresholds: float) -> Self:
        self._ensure_configuring()
        if not metric_thresholds:
            raise ValidationError("practical_significance requires at least one metric threshold.")

        validated_thresholds: dict[str, float] = {}
        for metric_name, threshold in metric_thresholds.items():
            validate_metric_names((metric_name,))
            if isinstance(threshold, bool) or not isinstance(threshold, Real):
                raise ValidationError(
                    f"Practical significance threshold for '{metric_name}' must be a number."
                )
            threshold_float = float(threshold)
            if not np.isfinite(threshold_float):
                raise ValidationError(
                    f"Practical significance threshold for '{metric_name}' must be finite."
                )
            if threshold_float < 0:
                raise ValidationError(
                    f"Practical significance threshold for '{metric_name}' must be non-negative."
                )
            validated_thresholds[metric_name] = threshold_float

        self._practical_significance = validated_thresholds
        return self

    def baseline(self, model_name: str) -> Self:
        self._ensure_configuring()
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("baseline model_name must be a non-empty string.")
        self._baseline_model_name = model_name
        return self

    def guardrails(
        self,
        *,
        min_improvement: dict[str, float],
        confidence: float = 0.95,
    ) -> Self:
        self._ensure_configuring()
        if not isinstance(min_improvement, dict) or not min_improvement:
            raise ValidationError("guardrails min_improvement must be a non-empty dictionary.")
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise ValidationError("guardrails confidence must be a number.")
        confidence_float = float(confidence)
        if not 0 < confidence_float < 1:
            raise ValidationError("guardrails confidence must be between 0 and 1.")

        validated_thresholds: dict[str, float] = {}
        for metric_name, threshold in min_improvement.items():
            validate_metric_names((metric_name,))
            if isinstance(threshold, bool) or not isinstance(threshold, Real):
                raise ValidationError(
                    f"Guardrail threshold for '{metric_name}' must be a number."
                )
            threshold_float = float(threshold)
            if not np.isfinite(threshold_float):
                raise ValidationError(
                    f"Guardrail threshold for '{metric_name}' must be finite."
                )
            if threshold_float < 0:
                raise ValidationError(
                    f"Guardrail threshold for '{metric_name}' must be non-negative."
                )
            validated_thresholds[metric_name] = threshold_float

        self._guardrail_min_improvement = validated_thresholds
        self._guardrail_confidence = confidence_float
        return self

    def fasten(self, lock_path: str = "statbelt.lock.json") -> Self:
        self._ensure_configuring()
        self._validate_configuration()
        self._X, self._y = self._snapshot_data()
        self._models = self._snapshot_models()
        self._splits, self._split_metadata = self._build_splits()
        self._write_lockfile(
            lock_path=lock_path,
            splits=self._splits,
            split_metadata=self._split_metadata,
        )
        self._lock_path = lock_path
        self._state = self._STATE_FASTENED
        return self

    def evaluate(self) -> EvaluationReport:
        self._ensure_ready_for_evaluation()
        assert self._X is not None
        assert self._y is not None
        assert self._task is not None
        assert self._splits is not None
        assert self._split_metadata is not None

        class_labels = np.unique(self._y)
        positive_label = class_labels[-1]
        task_name = self._task
        needs_probability = any(
            metric_requires_probability(metric_name) for metric_name in self._metric_names
        )
        needs_score = any(metric_requires_score(metric_name) for metric_name in self._metric_names)
        model_reports: list[ModelReport] = []
        fold_values_by_model: dict[str, dict[str, NDArray[np.float64]]] = {}

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
                        task_name=task_name,
                    )
                    fold_metrics[metric_name].append(metric_value)

            metric_intervals: dict[str, MetricInterval] = {}
            model_metric_values: dict[str, NDArray[np.float64]] = {}
            for metric_index, metric_name in enumerate(self._metric_names):
                metric_values = np.asarray(fold_metrics[metric_name], dtype=float)
                model_metric_values[metric_name] = metric_values
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
            fold_values_by_model[model_name] = model_metric_values

        pairwise = self._build_pairwise_comparisons(fold_values_by_model)
        pairwise = self._apply_multiplicity_correction(pairwise)
        guardrails = self._build_guardrail_report(fold_values_by_model)

        self._state = self._STATE_EVALUATED
        return EvaluationReport(
            task=self._task,
            cv=self._cv,
            cv_repeats=self._cv_repeats,
            random_state=self._random_state,
            bootstrap_resamples=self._bootstrap_resamples,
            models=model_reports,
            splits=_copy_splits(self._splits),
            split_metadata=[meta.copy() for meta in self._split_metadata],
            pairwise=pairwise,
            guardrails=guardrails,
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
        if self._task not in self._SUPPORTED_TASKS:
            supported = ", ".join(sorted(self._SUPPORTED_TASKS))
            raise ConfigurationError(
                f"Unsupported task '{self._task}'. Supported tasks: {supported}."
            )

        try:
            target_type = type_of_target(self._y, raise_unknown=True)
            class_labels = unique_labels(self._y)
        except ValueError as exc:
            if self._task == self._TASK_BINARY_CLASSIFICATION:
                raise ValidationError(
                    "binary_classification task requires exactly two classes."
                ) from exc
            raise ValidationError(
                "multiclass_classification task requires at least three classes."
            ) from exc

        if self._task == self._TASK_BINARY_CLASSIFICATION:
            if target_type != "binary" or class_labels.size != 2:
                raise ValidationError(
                    "binary_classification task requires exactly two classes."
                )
        else:
            if target_type != "multiclass" or class_labels.size < 3:
                raise ValidationError(
                    "multiclass_classification task requires at least three classes."
                )
        _, counts = np.unique(self._y, return_counts=True)
        if counts.min() < self._cv:
            raise ValidationError(
                "Each class must have at least cv samples for stratified k-fold."
            )
        if self._X.shape[0] < self._cv:
            raise ValidationError("Number of samples must be at least cv.")

        resolved_metric_names = resolve_metric_names(self._metric_names, task_name=self._task)
        resolved_practical_significance = self._resolve_threshold_metrics(
            self._practical_significance,
            setting_name="practical_significance",
        )
        resolved_guardrail_thresholds = self._resolve_threshold_metrics(
            self._guardrail_min_improvement,
            setting_name="guardrails",
        )

        model_names = {model_name for model_name, _ in self._models}
        if self._baseline_model_name is not None and self._baseline_model_name not in model_names:
            raise ValidationError(
                f"Baseline model '{self._baseline_model_name}' was not found in compare(...)."
            )
        for metric_name in resolved_practical_significance:
            if metric_name not in resolved_metric_names:
                raise ConfigurationError(
                    f"practical_significance metric '{metric_name}' must also be included in metrics(...)."
                )
        if resolved_guardrail_thresholds and self._baseline_model_name is None:
            raise ConfigurationError("guardrails(...) requires baseline(...).")
        if resolved_guardrail_thresholds and len(self._models) < 2:
            raise ConfigurationError("guardrails(...) requires at least two models.")
        for metric_name in resolved_guardrail_thresholds:
            if metric_name not in resolved_metric_names:
                raise ConfigurationError(
                    f"guardrails metric '{metric_name}' must also be included in metrics(...)."
                )

        for model_name, estimator in self._models:
            validate_estimator_for_metrics(
                estimator,
                model_name,
                resolved_metric_names,
                task_name=self._task,
            )

        self._metric_names = resolved_metric_names
        self._practical_significance = resolved_practical_significance
        self._guardrail_min_improvement = resolved_guardrail_thresholds

    def _resolve_threshold_metrics(
        self,
        thresholds: dict[str, float],
        *,
        setting_name: str,
    ) -> dict[str, float]:
        assert self._task is not None
        resolved: dict[str, float] = {}
        for metric_name, threshold in thresholds.items():
            resolved_metric_name = resolve_metric_names(
                (metric_name,),
                task_name=self._task,
            )[0]
            if resolved_metric_name in resolved:
                raise ConfigurationError(
                    f"{setting_name} contains duplicate thresholds after task-based metric "
                    f"resolution ('{metric_name}' maps to '{resolved_metric_name}')."
                )
            resolved[resolved_metric_name] = threshold
        return resolved

    def _build_splits(
        self,
    ) -> tuple[list[tuple[list[int], list[int]]], list[dict[str, int]]]:
        assert self._X is not None
        assert self._y is not None
        splitter = RepeatedStratifiedKFold(
            n_splits=self._cv,
            n_repeats=self._cv_repeats,
            random_state=self._random_state,
        )
        splits: list[tuple[list[int], list[int]]] = []
        split_metadata: list[dict[str, int]] = []
        for split_index, (train_idx, test_idx) in enumerate(splitter.split(self._X, self._y)):
            repeat_index, fold_index = divmod(split_index, self._cv)
            splits.append((train_idx.tolist(), test_idx.tolist()))
            split_metadata.append({"repeat": repeat_index, "fold": fold_index})

        return splits, split_metadata

    def _write_lockfile(
        self,
        *,
        lock_path: str,
        splits: list[tuple[list[int], list[int]]],
        split_metadata: list[dict[str, int]],
    ) -> None:
        payload = {
            "schema_version": 3,
            "task": self._task,
            "metrics": list(self._metric_names),
            "cv": self._cv,
            "cv_repeats": self._cv_repeats,
            "random_state": self._random_state,
            "alpha": self._alpha,
            "bootstrap_resamples": self._bootstrap_resamples,
            "pairwise_inference": {
                "method": self._pairwise_method,
                "alternative": self._pairwise_alternative,
            },
            "multiplicity": {
                "method": self._multiplicity_method,
                "family": self._multiplicity_family,
            },
            "practical_significance": self._practical_significance,
            "baseline_model": self._baseline_model_name,
            "guardrails": (
                {
                    "min_improvement": self._guardrail_min_improvement,
                    "confidence": self._guardrail_confidence,
                }
                if self._guardrail_min_improvement
                else None
            ),
            "models": [model_name for model_name, _ in self._models],
            "splits": [
                {
                    "train": train_indices,
                    "test": test_indices,
                    "repeat": split_meta["repeat"],
                    "fold": split_meta["fold"],
                }
                for (train_indices, test_indices), split_meta in zip(
                    splits, split_metadata, strict=True
                )
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

    def _build_pairwise_comparisons(
        self,
        fold_values_by_model: dict[str, dict[str, NDArray[np.float64]]],
    ) -> list[PairwiseComparison]:
        if len(self._models) < 2:
            return []

        pairwise: list[PairwiseComparison] = []
        model_names = [model_name for model_name, _ in self._models]
        for metric_index, metric_name in enumerate(self._metric_names):
            practical_threshold = self._practical_significance.get(metric_name, 0.0)
            for first_index, second_index in combinations(range(len(model_names)), 2):
                model_a = model_names[first_index]
                model_b = model_names[second_index]
                deltas = (
                    fold_values_by_model[model_a][metric_name]
                    - fold_values_by_model[model_b][metric_name]
                )
                directional_deltas = (
                    deltas if metric_higher_is_better(metric_name) else -deltas
                )
                pair_seed = _derive_pairwise_seed(
                    self._random_state,
                    first_index=first_index,
                    second_index=second_index,
                    metric_index=metric_index,
                )
                ci_low, ci_high = _bootstrap_mean_interval(
                    deltas,
                    alpha=self._alpha,
                    n_resamples=self._bootstrap_resamples,
                    random_state=pair_seed,
                )
                if self._pairwise_method == "paired_bootstrap":
                    p_value = _paired_bootstrap_p_value(
                        directional_deltas,
                        n_resamples=self._bootstrap_resamples,
                        random_state=pair_seed,
                        alternative=self._pairwise_alternative,
                    )
                else:
                    p_value = _paired_permutation_p_value(
                        directional_deltas,
                        n_resamples=self._bootstrap_resamples,
                        random_state=pair_seed,
                        alternative=self._pairwise_alternative,
                    )

                delta = float(deltas.mean())
                pairwise.append(
                    PairwiseComparison(
                        model_a=model_a,
                        model_b=model_b,
                        metric=metric_name,
                        delta=delta,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        alpha=self._alpha,
                        p_value=p_value,
                        p_adjusted=p_value,
                        reject_null=p_value <= self._alpha,
                        is_practically_significant=abs(delta) > practical_threshold,
                    )
                )

        return pairwise

    def _apply_multiplicity_correction(
        self, pairwise: list[PairwiseComparison]
    ) -> list[PairwiseComparison]:
        if not pairwise:
            return []

        grouped_indices: dict[str, list[int]] = {}
        if self._multiplicity_family == "global":
            grouped_indices["global"] = list(range(len(pairwise)))
        else:
            for index, comparison in enumerate(pairwise):
                grouped_indices.setdefault(comparison.metric, []).append(index)

        adjusted: list[float] = [0.0] * len(pairwise)
        for index_group in grouped_indices.values():
            p_values = [pairwise[index].p_value for index in index_group]
            corrected = _apply_pvalue_adjustment(p_values, method=self._multiplicity_method)
            for index, corrected_value in zip(index_group, corrected, strict=True):
                adjusted[index] = corrected_value

        return [
            replace(
                comparison,
                p_adjusted=adjusted_value,
                reject_null=adjusted_value <= self._alpha,
            )
            for comparison, adjusted_value in zip(pairwise, adjusted, strict=True)
        ]

    def _build_guardrail_report(
        self,
        fold_values_by_model: dict[str, dict[str, NDArray[np.float64]]],
    ) -> GuardrailReport | None:
        if not self._guardrail_min_improvement:
            return None
        assert self._baseline_model_name is not None

        checks: list[GuardrailCheck] = []
        model_names = [model_name for model_name, _ in self._models]
        baseline_values = fold_values_by_model[self._baseline_model_name]
        guardrail_alpha = 1 - self._guardrail_confidence
        guardrail_metric_names = [
            metric_name
            for metric_name in self._metric_names
            if metric_name in self._guardrail_min_improvement
        ]

        for model_index, model_name in enumerate(model_names):
            if model_name == self._baseline_model_name:
                continue
            for metric_index, metric_name in enumerate(guardrail_metric_names):
                min_improvement = self._guardrail_min_improvement[metric_name]
                raw_delta = (
                    fold_values_by_model[model_name][metric_name] - baseline_values[metric_name]
                )
                if metric_higher_is_better(metric_name):
                    improvement_values = raw_delta
                else:
                    improvement_values = -raw_delta

                ci_low, ci_high = _bootstrap_mean_interval(
                    improvement_values,
                    alpha=guardrail_alpha,
                    n_resamples=self._bootstrap_resamples,
                    random_state=_derive_seed(
                        self._random_state,
                        model_index=9_000 + model_index,
                        metric_index=4_000 + metric_index,
                    ),
                )
                improvement_point_estimate = float(improvement_values.mean())
                checks.append(
                    GuardrailCheck(
                        challenger_model=model_name,
                        baseline_model=self._baseline_model_name,
                        metric=metric_name,
                        min_improvement=min_improvement,
                        confidence=self._guardrail_confidence,
                        improvement_point_estimate=improvement_point_estimate,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        passed=ci_low >= min_improvement,
                    )
                )

        return GuardrailReport(
            overall_pass=all(check.passed for check in checks),
            checks=checks,
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

    def _snapshot_data(
        self,
    ) -> tuple[NDArray[np.generic] | spmatrix, NDArray[np.generic]]:
        assert self._X is not None
        assert self._y is not None
        return _copy_feature_matrix(self._X), np.array(self._y, copy=True)


def _align_probability_columns(
    *,
    estimator: object,
    y_proba: NDArray[np.float64],
    class_labels: NDArray[np.generic],
) -> NDArray[np.float64]:
    if y_proba.ndim == 1:
        return _align_binary_probability_vector(
            estimator=estimator,
            probability_values=y_proba,
            class_labels=class_labels,
            source="1D",
        )
    if y_proba.ndim == 2 and y_proba.shape[1] == 1:
        return _align_binary_probability_vector(
            estimator=estimator,
            probability_values=y_proba[:, 0],
            class_labels=class_labels,
            source="single-column",
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

    result = scipy_bootstrap(
        (values,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=1 - alpha,
        method="percentile",
        rng=np.random.default_rng(random_state),
    )
    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)
    return ci_low, ci_high


def _align_binary_probability_vector(
    *,
    estimator: object,
    probability_values: NDArray[np.float64],
    class_labels: NDArray[np.generic],
    source: str,
) -> NDArray[np.float64]:
    if class_labels.size != 2:
        raise ValidationError(
            f"{source} predict_proba output is only supported for binary classification."
        )
    estimator_classes = _require_estimator_classes(estimator, expected_size=2)
    probability_class = estimator_classes[1]
    if probability_class == class_labels[1]:
        return np.column_stack([1 - probability_values, probability_values])
    if probability_class == class_labels[0]:
        return np.column_stack([probability_values, 1 - probability_values])
    raise ValidationError(
        f"Could not map {source} predict_proba output to dataset class labels."
    )


def _derive_seed(base_seed: int, *, model_index: int, metric_index: int) -> int:
    # Keep bootstrap deterministic without relying on Python hash randomization.
    return int(base_seed + model_index * 1000 + metric_index * 17)


def _derive_pairwise_seed(
    base_seed: int,
    *,
    first_index: int,
    second_index: int,
    metric_index: int,
) -> int:
    pair_index = _cantor_pair(first_index, second_index)
    metric_pair_index = _cantor_pair(pair_index, metric_index + 3_000)
    return _cantor_pair(base_seed, metric_pair_index)


def _cantor_pair(left: int, right: int) -> int:
    # Injective pairing function over non-negative integers.
    total = left + right
    return (total * (total + 1)) // 2 + right


def _paired_bootstrap_p_value(
    values: NDArray[np.float64],
    *,
    n_resamples: int,
    random_state: int,
    alternative: str,
) -> float:
    bootstrap_result = scipy_bootstrap(
        (values,),
        np.mean,
        n_resamples=n_resamples,
        method="percentile",
        rng=np.random.default_rng(random_state),
    )
    bootstrap_distribution = np.asarray(
        bootstrap_result.bootstrap_distribution,
        dtype=float,
    )
    n = bootstrap_distribution.size
    p_less_or_equal_zero = (int(np.count_nonzero(bootstrap_distribution <= 0)) + 1) / (n + 1)
    p_greater_or_equal_zero = (int(np.count_nonzero(bootstrap_distribution >= 0)) + 1) / (n + 1)

    if alternative == "greater":
        return p_less_or_equal_zero
    if alternative == "less":
        return p_greater_or_equal_zero
    if alternative == "two-sided":
        return min(1.0, 2 * min(p_less_or_equal_zero, p_greater_or_equal_zero))
    raise ValidationError(f"Unsupported alternative '{alternative}'.")


def _paired_permutation_p_value(
    values: NDArray[np.float64],
    *,
    n_resamples: int,
    random_state: int,
    alternative: str,
) -> float:
    if alternative not in ExperimentalHarness._PAIRWISE_ALTERNATIVES:
        raise ValidationError(f"Unsupported alternative '{alternative}'.")
    result = scipy_permutation_test(
        data=(values, np.zeros_like(values)),
        statistic=lambda left, right: float(np.mean(left - right)),
        permutation_type="samples",
        n_resamples=n_resamples,
        alternative=alternative,
        rng=np.random.default_rng(random_state),
    )
    return float(result.pvalue)


def _apply_pvalue_adjustment(p_values: list[float], *, method: str) -> list[float]:
    if not p_values:
        return []

    p_values_array = np.asarray(p_values, dtype=float)
    m = p_values_array.size
    if method == "bonferroni":
        return np.clip(p_values_array * m, 0.0, 1.0).tolist()

    if method == "holm":
        order = np.argsort(p_values_array)
        adjusted = np.empty(m, dtype=float)
        running_max = 0.0
        for rank, index in enumerate(order):
            value = float((m - rank) * p_values_array[index])
            running_max = max(running_max, value)
            adjusted[index] = min(1.0, running_max)
        return adjusted.tolist()

    if method == "fdr_bh":
        return scipy_false_discovery_control(p_values_array, method="bh").tolist()

    raise ValidationError(f"Unsupported multiplicity method '{method}'.")


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
    if isinstance(estimator, BaseEstimator):
        try:
            check_is_fitted(estimator, attributes=["classes_"])
        except Exception:
            # Best-effort only: fall back to explicit classes_ checks below.
            pass

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


def _copy_feature_matrix(
    X: NDArray[np.generic] | spmatrix,
) -> NDArray[np.generic] | spmatrix:
    return X.copy()
