import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from statbelt import ExperimentalHarness
from statbelt.exceptions import ConfigurationError, ValidationError
from statbelt.harness import (
    _align_binary_decision_vector,
    _align_binary_probability_vector,
    _align_probability_columns,
    _apply_pvalue_adjustment,
    _bootstrap_mean_interval,
    _paired_bootstrap_p_value,
    _paired_permutation_p_value,
    _positive_class_decision_score,
    _require_estimator_classes,
)
from statbelt.metrics import (
    METRIC_REGISTRY,
    MetricSpec,
    _supported_metrics_for_task,
    compute_metric,
    resolve_metric_names,
    validate_estimator_for_metrics,
    validate_metric_names,
)
from statbelt.report import EvaluationReport, GuardrailCheck, GuardrailReport, MetricInterval, ModelReport


class _NoPredictEstimator:
    def fit(self, X: object, y: object) -> "_NoPredictEstimator":
        return self


class _PredictOnlyEstimator:
    def fit(self, X: object, y: object) -> "_PredictOnlyEstimator":
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.zeros(X.shape[0], dtype=int)


class _DecisionOneColumnEstimator(_PredictOnlyEstimator):
    def decision_function(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.ones((X.shape[0], 1), dtype=float)


class _DecisionThreeColumnEstimator(_PredictOnlyEstimator):
    def decision_function(self, X: object) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        return np.ones((X.shape[0], 3), dtype=float)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("y should be a 1d array", "one-dimensional array-like"),
        ("inconsistent numbers of samples", "matching sample counts"),
        ("minimum of 2 is required", "At least two samples are required"),
    ],
)
def test_data_maps_specific_check_xy_messages(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    expected: str,
) -> None:
    def _raise(*args: object, **kwargs: object) -> tuple[object, object]:
        raise ValueError(message)

    monkeypatch.setattr("statbelt.harness.check_X_y", _raise)
    with pytest.raises(ValidationError, match=expected):
        ExperimentalHarness().data(np.array([[1.0], [2.0]]), np.array([0, 1]))


def test_data_accepts_1d_feature_input_and_reshapes() -> None:
    harness = ExperimentalHarness().data(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 0, 1]))
    assert harness._X is not None
    assert harness._X.shape == (4, 1)


def test_compare_validation_edges() -> None:
    with pytest.raises(ValidationError, match="At least one model"):
        ExperimentalHarness().compare()

    with pytest.raises(ValidationError, match="tuple of \\(model_name, estimator\\)"):
        ExperimentalHarness().compare(("only_name",))  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="non-empty strings"):
        ExperimentalHarness().compare((" ", LogisticRegression(max_iter=100)))

    with pytest.raises(ValidationError, match="Duplicate model name"):
        ExperimentalHarness().compare(
            ("dup", LogisticRegression(max_iter=100)),
            ("dup", LogisticRegression(max_iter=100)),
        )

    with pytest.raises(ValidationError, match="fit and predict methods"):
        ExperimentalHarness().compare(("bad", _NoPredictEstimator()))


def test_design_and_threshold_validation_edges() -> None:
    with pytest.raises(ValidationError, match="cv must be at least 2"):
        ExperimentalHarness().design(cv=1)

    with pytest.raises(ValidationError, match="cv_repeats must be an integer"):
        ExperimentalHarness().design(cv_repeats=1.5)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="requires at least one metric threshold"):
        ExperimentalHarness().practical_significance()

    with pytest.raises(ValidationError, match="must be a number"):
        ExperimentalHarness().practical_significance(accuracy=True)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="must be a non-empty string"):
        ExperimentalHarness().baseline(" ")

    with pytest.raises(ValidationError, match="non-empty dictionary"):
        ExperimentalHarness().guardrails(min_improvement={})

    with pytest.raises(ValidationError, match="confidence must be a number"):
        ExperimentalHarness().guardrails(
            min_improvement={"accuracy": 0.1},
            confidence="0.9",  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError, match="must be a number"):
        ExperimentalHarness().guardrails(
            min_improvement={"accuracy": "0.01"}  # type: ignore[arg-type]
        )


def test_fasten_rejects_when_task_internal_state_is_invalid(tmp_path) -> None:
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=11,
    )
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=200)))
        .metrics("accuracy")
    )
    harness._task = "regression"

    with pytest.raises(ConfigurationError, match="Unsupported task"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_fasten_maps_type_of_target_errors_to_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=17,
    )
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=200)))
        .metrics("accuracy")
    )

    def _raise(*args: object, **kwargs: object) -> str:
        raise ValueError("target type error")

    monkeypatch.setattr("statbelt.harness.type_of_target", _raise)
    with pytest.raises(ValidationError, match="requires exactly two classes"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_fasten_maps_type_of_target_errors_for_multiclass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=17,
    )
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(("logreg", LogisticRegression(max_iter=200)))
        .metrics("accuracy")
    )

    def _raise(*args: object, **kwargs: object) -> str:
        raise ValueError("target type error")

    monkeypatch.setattr("statbelt.harness.type_of_target", _raise)
    with pytest.raises(ValidationError, match="at least three classes"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_fasten_requires_each_class_to_have_at_least_cv_samples(tmp_path) -> None:
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 19 + [1], dtype=int)
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=200)))
        .metrics("accuracy")
        .design(cv=3, random_state=7)
    )

    with pytest.raises(ValidationError, match="Each class must have at least cv samples"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_fasten_rejects_when_sample_count_is_below_cv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=200)))
        .metrics("accuracy")
        .design(cv=5, random_state=7)
    )

    real_unique = np.unique

    def _fake_unique(values: object, *args: object, **kwargs: object):
        return_counts = bool(kwargs.get("return_counts", False))
        if return_counts:
            return np.array([0, 1], dtype=int), np.array([5, 5], dtype=int)
        return real_unique(values, *args, **kwargs)

    monkeypatch.setattr("statbelt.harness.np.unique", _fake_unique)
    with pytest.raises(ValidationError, match="Number of samples must be at least cv"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_score_for_binary_auc_helper_branches() -> None:
    harness = ExperimentalHarness().metrics("accuracy")
    X_test = np.array([[0.0], [1.0]], dtype=float)
    class_labels = np.array([0, 1], dtype=int)

    assert (
        harness._score_for_binary_auc(
            estimator=_PredictOnlyEstimator(),
            X_test=X_test,
            y_proba=None,
            class_labels=class_labels,
        )
        is None
    )

    harness = ExperimentalHarness().metrics("roc_auc")
    assert (
        harness._score_for_binary_auc(
            estimator=_PredictOnlyEstimator(),
            X_test=X_test,
            y_proba=None,
            class_labels=class_labels,
        )
        is None
    )

    aligned_from_single_col = harness._score_for_binary_auc(
        estimator=_DecisionOneColumnEstimator().fit(X_test, np.array([0, 1], dtype=int)),
        X_test=X_test,
        y_proba=None,
        class_labels=class_labels,
    )
    assert aligned_from_single_col is not None
    assert aligned_from_single_col.shape == (2,)

    with pytest.raises(ValidationError, match="decision_function must return a 1D vector"):
        harness._score_for_binary_auc(
            estimator=_DecisionThreeColumnEstimator().fit(X_test, np.array([0, 1], dtype=int)),
            X_test=X_test,
            y_proba=None,
            class_labels=class_labels,
        )


def test_probability_and_decision_alignment_helper_edges() -> None:
    class _Estimator:
        classes_ = np.array([0, 1], dtype=int)

    class _MismatchedEstimator:
        classes_ = np.array([2, 3], dtype=int)

    with pytest.raises(ValidationError, match="1D or 2D array"):
        _align_probability_columns(
            estimator=_Estimator(),
            y_proba=np.zeros((2, 2, 1), dtype=float),
            class_labels=np.array([0, 1], dtype=int),
        )

    with pytest.raises(ValidationError, match="Could not map estimator probability columns"):
        _align_probability_columns(
            estimator=_MismatchedEstimator(),
            y_proba=np.array([[0.2, 0.8], [0.7, 0.3]], dtype=float),
            class_labels=np.array([0, 1], dtype=int),
        )

    with pytest.raises(ValidationError, match="map ambiguously"):
        _align_probability_columns(
            estimator=_Estimator(),
            y_proba=np.array([[0.2, 0.8], [0.7, 0.3]], dtype=float),
            class_labels=np.array([0, 0], dtype=int),
        )

    with pytest.raises(ValidationError, match="only supported for binary classification"):
        _align_binary_probability_vector(
            estimator=_Estimator(),
            probability_values=np.array([0.1, 0.9], dtype=float),
            class_labels=np.array([0, 1, 2], dtype=int),
            source="1D",
        )

    ordered = _align_binary_probability_vector(
        estimator=_Estimator(),
        probability_values=np.array([0.2, 0.8], dtype=float),
        class_labels=np.array([0, 1], dtype=int),
        source="1D",
    )
    assert np.allclose(ordered[:, 1], np.array([0.2, 0.8], dtype=float))

    with pytest.raises(ValidationError, match="Could not map 1D predict_proba output"):
        _align_binary_probability_vector(
            estimator=_MismatchedEstimator(),
            probability_values=np.array([0.2, 0.8], dtype=float),
            class_labels=np.array([0, 1], dtype=int),
            source="1D",
        )

    with pytest.raises(ValidationError, match="one-dimensional"):
        _align_binary_decision_vector(
            estimator=_Estimator(),
            decision_values=np.ones((2, 2), dtype=float),
            class_labels=np.array([0, 1], dtype=int),
        )

    with pytest.raises(ValidationError, match="requires exactly two classes"):
        _align_binary_decision_vector(
            estimator=_Estimator(),
            decision_values=np.ones(2, dtype=float),
            class_labels=np.array([0, 1, 2], dtype=int),
        )

    with pytest.raises(ValidationError, match="Could not map estimator decision-function scores"):
        _align_binary_decision_vector(
            estimator=_MismatchedEstimator(),
            decision_values=np.ones(2, dtype=float),
            class_labels=np.array([0, 1], dtype=int),
        )

    with pytest.raises(ValidationError, match="Could not map estimator decision-function columns"):
        _positive_class_decision_score(
            estimator=_MismatchedEstimator(),
            decision_values=np.ones((2, 2), dtype=float),
            class_labels=np.array([0, 1], dtype=int),
        )

    class _BadDimEstimator:
        classes_ = np.array([[0, 1]])

    class _BadSizeEstimator:
        classes_ = np.array([0, 1, 2], dtype=int)

    with pytest.raises(ValidationError, match="must be one-dimensional"):
        _require_estimator_classes(_BadDimEstimator(), expected_size=2)

    with pytest.raises(ValidationError, match="must match the class-dependent output dimension"):
        _require_estimator_classes(_BadSizeEstimator(), expected_size=2)


def test_statistical_helper_error_paths() -> None:
    with pytest.raises(ValidationError, match="one-dimensional"):
        _bootstrap_mean_interval(
            np.ones((2, 1), dtype=float),
            alpha=0.05,
            n_resamples=50,
            random_state=7,
        )

    with pytest.raises(ValidationError, match="cannot be empty"):
        _bootstrap_mean_interval(
            np.array([], dtype=float),
            alpha=0.05,
            n_resamples=50,
            random_state=7,
        )

    values = np.array([0.1, 0.2, 0.3], dtype=float)
    with pytest.raises(ValidationError, match="Unsupported alternative"):
        _paired_bootstrap_p_value(values, n_resamples=100, random_state=7, alternative="bad")

    with pytest.raises(ValidationError, match="Unsupported alternative"):
        _paired_permutation_p_value(values, n_resamples=100, random_state=7, alternative="bad")

    assert _apply_pvalue_adjustment([], method="bonferroni") == []
    with pytest.raises(ValidationError, match="Unsupported multiplicity method"):
        _apply_pvalue_adjustment([0.1, 0.2], method="bad")


def test_metrics_helper_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="At least one metric must be provided"):
        validate_metric_names(())

    with pytest.raises(ValidationError, match="fit and predict methods"):
        validate_estimator_for_metrics(object(), "obj", ("accuracy",))

    with pytest.raises(ValidationError, match="predict_proba or decision_function"):
        validate_estimator_for_metrics(_PredictOnlyEstimator(), "predict_only", ("roc_auc",))

    y_true = np.array([0, 1], dtype=int)
    y_pred = np.array([0, 1], dtype=int)

    with pytest.raises(ValidationError, match="requires score values"):
        compute_metric(
            "roc_auc",
            y_true,
            y_pred=y_pred,
            y_score=None,
            y_proba=None,
            positive_label=1,
            class_labels=np.array([0, 1], dtype=int),
        )

    with pytest.raises(ValidationError, match="requires probability values"):
        compute_metric(
            "log_loss",
            y_true,
            y_pred=y_pred,
            y_score=None,
            y_proba=None,
            positive_label=1,
            class_labels=np.array([0, 1], dtype=int),
        )

    monkeypatch.setitem(
        METRIC_REGISTRY,
        "dummy_proba_metric",
        MetricSpec(
            requirement="proba",
            higher_is_better=True,
            scorer=lambda _y_true, _y_proba, _positive: 0.123,
        ),
    )
    try:
        assert (
            compute_metric(
                "dummy_proba_metric",
                y_true,
                y_pred=y_pred,
                y_score=None,
                y_proba=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float),
                positive_label=1,
                class_labels=np.array([0, 1], dtype=int),
            )
            == 0.123
        )
    finally:
        METRIC_REGISTRY.pop("dummy_proba_metric", None)


def test_metric_resolution_helper_paths() -> None:
    assert validate_metric_names(
        ("precision", "roc_auc"),
        task_name="multiclass_classification",
    ) == ("precision_macro", "roc_auc_ovr_macro")

    with pytest.raises(ValidationError, match="Unsupported metric"):
        resolve_metric_names(("precision_macro",), task_name="binary_classification")

    with pytest.raises(ValidationError, match="Unsupported metric"):
        resolve_metric_names(("precision_binary",), task_name="multiclass_classification")

    with pytest.raises(ValidationError, match="Duplicate metric"):
        resolve_metric_names(("precision", "precision_macro"), task_name="multiclass_classification")

    with pytest.raises(ValidationError, match="Unsupported task"):
        resolve_metric_names(("accuracy",), task_name="regression")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="Unsupported task"):
        _supported_metrics_for_task("regression")  # type: ignore[arg-type]


def test_fasten_rejects_duplicate_thresholds_after_alias_resolution(tmp_path) -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=22,
    )
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(("logreg", LogisticRegression(max_iter=200)))
        .metrics("precision_macro")
        .practical_significance(precision=0.01, precision_macro=0.02)
    )

    with pytest.raises(ConfigurationError, match="duplicate thresholds after task-based metric"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_report_guardrail_serialization_and_summary_block() -> None:
    report = EvaluationReport(
        task="binary_classification",
        cv=5,
        cv_repeats=1,
        random_state=42,
        bootstrap_resamples=200,
        models=[
            ModelReport(
                model_name="model_a",
                metrics={
                    "accuracy": MetricInterval(
                        point_estimate=0.8,
                        ci_low=0.7,
                        ci_high=0.9,
                        alpha=0.05,
                    )
                },
            )
        ],
        splits=[([0, 1], [2, 3])],
        split_metadata=[{"repeat": 0, "fold": 0}],
        guardrails=GuardrailReport(
            overall_pass=True,
            checks=[
                GuardrailCheck(
                    challenger_model="model_a",
                    baseline_model="baseline",
                    metric="accuracy",
                    min_improvement=0.01,
                    confidence=0.95,
                    improvement_point_estimate=0.05,
                    ci_low=0.02,
                    ci_high=0.08,
                    passed=True,
                )
            ],
        ),
    )

    payload = report.to_dict()
    assert payload["guardrails"]["overall_pass"] is True
    assert payload["guardrails"]["checks"][0]["passed"] is True

    summary = report.summary()
    assert "Guardrails: PASS" in summary
    assert "model_a vs baseline [accuracy]: PASS" in summary
