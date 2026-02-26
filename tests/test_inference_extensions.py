import json

import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from statbelt import ExperimentalHarness
from statbelt.exceptions import ConfigurationError, ValidationError


def _strong_binary_data() -> tuple[object, object]:
    return make_classification(
        n_samples=500,
        n_features=10,
        n_informative=7,
        n_redundant=0,
        class_sep=2.2,
        flip_y=0.0,
        random_state=123,
    )


def _strong_multiclass_data() -> tuple[object, object]:
    return make_classification(
        n_samples=600,
        n_features=12,
        n_informative=9,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=2.0,
        flip_y=0.0,
        random_state=222,
    )


def _pairwise_by_metric(report: object) -> dict[str, object]:
    return {comparison.metric: comparison for comparison in report.pairwise}


def test_pairwise_comparisons_are_emitted_and_deterministic(tmp_path) -> None:
    X, y = _strong_binary_data()

    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=3, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=300)
        .compare_inference(method="paired_bootstrap", alternative="two-sided")
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
    )

    first = harness.evaluate()
    second = harness.evaluate()

    assert len(first.pairwise) == 2
    assert first.pairwise == second.pairwise

    pairwise = _pairwise_by_metric(first)
    assert pairwise["accuracy"].delta > 0
    assert pairwise["log_loss"].delta < 0


def test_permutation_pairwise_inference_rejects_two_sided_alias(tmp_path) -> None:
    X, y = _strong_binary_data()

    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy")
        .design(cv=5, cv_repeats=2, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=200)
    )

    with pytest.raises(ValidationError, match="Supported alternatives"):
        harness.compare_inference(method="permutation", alternative="two_sided")


def test_permutation_pairwise_inference_accepts_two_sided_with_hyphen(tmp_path) -> None:
    X, y = _strong_binary_data()

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy")
        .design(cv=5, cv_repeats=2, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=200)
        .compare_inference(method="permutation", alternative="two-sided")
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert len(report.pairwise) == 1
    assert 0 <= report.pairwise[0].p_value <= 1


def test_one_sided_pairwise_respects_metric_direction(tmp_path) -> None:
    X, y = _strong_binary_data()

    report_greater = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=3, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=300)
        .compare_inference(method="paired_bootstrap", alternative="greater")
        .fasten(lock_path=str(tmp_path / "greater.lock.json"))
        .evaluate()
    )
    report_less = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=3, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=300)
        .compare_inference(method="paired_bootstrap", alternative="less")
        .fasten(lock_path=str(tmp_path / "less.lock.json"))
        .evaluate()
    )

    greater = _pairwise_by_metric(report_greater)
    less = _pairwise_by_metric(report_less)

    assert greater["accuracy"].delta > 0
    assert greater["accuracy"].p_value < less["accuracy"].p_value

    assert greater["log_loss"].delta < 0
    assert greater["log_loss"].p_value < less["log_loss"].p_value


def test_multiclass_pairwise_comparisons_are_emitted(tmp_path) -> None:
    X, y = _strong_multiclass_data()

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=600)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "roc_auc", "log_loss")
        .design(cv=5, cv_repeats=2, random_state=19)
        .inference(alpha=0.05, bootstrap_resamples=200)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert {comparison.metric for comparison in report.pairwise} == {
        "accuracy",
        "roc_auc_ovr_macro",
        "log_loss",
    }


def test_multiclass_one_sided_pairwise_respects_metric_direction(tmp_path) -> None:
    X, y = _strong_multiclass_data()

    report_greater = (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=600)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=2, random_state=23)
        .inference(alpha=0.05, bootstrap_resamples=200)
        .compare_inference(method="paired_bootstrap", alternative="greater")
        .fasten(lock_path=str(tmp_path / "greater.lock.json"))
        .evaluate()
    )
    report_less = (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=600)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=2, random_state=23)
        .inference(alpha=0.05, bootstrap_resamples=200)
        .compare_inference(method="paired_bootstrap", alternative="less")
        .fasten(lock_path=str(tmp_path / "less.lock.json"))
        .evaluate()
    )

    greater = _pairwise_by_metric(report_greater)
    less = _pairwise_by_metric(report_less)

    assert greater["accuracy"].delta > 0
    assert greater["accuracy"].p_value < less["accuracy"].p_value
    assert greater["log_loss"].delta < 0
    assert greater["log_loss"].p_value < less["log_loss"].p_value


def test_compare_inference_rejects_non_string_options() -> None:
    harness = ExperimentalHarness()

    with pytest.raises(ValidationError, match="method must be a string"):
        harness.compare_inference(method=[])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="alternative must be a string"):
        harness.compare_inference(alternative={})  # type: ignore[arg-type]


def test_compare_inference_rejects_unsupported_string_options() -> None:
    harness = ExperimentalHarness()

    with pytest.raises(ValidationError, match="Supported methods"):
        harness.compare_inference(method="unsupported")
    with pytest.raises(ValidationError, match="Supported alternatives"):
        harness.compare_inference(alternative="unsupported")


def test_multiplicity_rejects_non_string_options() -> None:
    harness = ExperimentalHarness()

    with pytest.raises(ValidationError, match="method must be a string"):
        harness.multiplicity(method=[])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="family must be a string"):
        harness.multiplicity(family={})  # type: ignore[arg-type]


def test_multiplicity_rejects_unsupported_string_options() -> None:
    harness = ExperimentalHarness()

    with pytest.raises(ValidationError, match="Supported methods"):
        harness.multiplicity(method="unsupported")
    with pytest.raises(ValidationError, match="Supported families"):
        harness.multiplicity(family="unsupported")


def test_multiplicity_per_metric_differs_from_global_bonferroni(tmp_path) -> None:
    X, y = _strong_binary_data()

    report_global = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=2, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=250)
        .multiplicity(method="bonferroni", family="global")
        .fasten(lock_path=str(tmp_path / "global.lock.json"))
        .evaluate()
    )

    report_per_metric = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=2, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=250)
        .multiplicity(method="bonferroni", family="per_metric")
        .fasten(lock_path=str(tmp_path / "per_metric.lock.json"))
        .evaluate()
    )

    by_metric_global = _pairwise_by_metric(report_global)
    by_metric_per_metric = _pairwise_by_metric(report_per_metric)

    for metric_name in ("accuracy", "log_loss"):
        global_comparison = by_metric_global[metric_name]
        per_metric_comparison = by_metric_per_metric[metric_name]
        assert per_metric_comparison.p_adjusted == pytest.approx(per_metric_comparison.p_value)
        assert global_comparison.p_adjusted == pytest.approx(
            min(1.0, global_comparison.p_value * 2)
        )
        assert global_comparison.p_adjusted >= per_metric_comparison.p_adjusted


def test_multiclass_multiplicity_per_metric_differs_from_global_bonferroni(tmp_path) -> None:
    X, y = _strong_multiclass_data()

    report_global = (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=600)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=2, random_state=29)
        .inference(alpha=0.05, bootstrap_resamples=180)
        .multiplicity(method="bonferroni", family="global")
        .fasten(lock_path=str(tmp_path / "global.lock.json"))
        .evaluate()
    )

    report_per_metric = (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=600)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=2, random_state=29)
        .inference(alpha=0.05, bootstrap_resamples=180)
        .multiplicity(method="bonferroni", family="per_metric")
        .fasten(lock_path=str(tmp_path / "per_metric.lock.json"))
        .evaluate()
    )

    by_metric_global = _pairwise_by_metric(report_global)
    by_metric_per_metric = _pairwise_by_metric(report_per_metric)
    for metric_name in ("accuracy", "log_loss"):
        global_comparison = by_metric_global[metric_name]
        per_metric_comparison = by_metric_per_metric[metric_name]
        assert per_metric_comparison.p_adjusted == pytest.approx(per_metric_comparison.p_value)
        assert global_comparison.p_adjusted == pytest.approx(
            min(1.0, global_comparison.p_value * 2)
        )
        assert global_comparison.p_adjusted >= per_metric_comparison.p_adjusted


def test_multiplicity_fdr_bh_path_emits_adjusted_p_values(tmp_path) -> None:
    X, y = _strong_binary_data()

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=2, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=250)
        .multiplicity(method="fdr_bh", family="global")
        .fasten(lock_path=str(tmp_path / "fdr_bh.lock.json"))
        .evaluate()
    )

    for comparison in report.pairwise:
        assert 0.0 <= comparison.p_adjusted <= 1.0
        assert comparison.p_adjusted + 1e-12 >= comparison.p_value


def test_practical_significance_threshold_flags_small_effects(tmp_path) -> None:
    X, y = _strong_binary_data()

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy")
        .design(cv=5, cv_repeats=2, random_state=21)
        .inference(alpha=0.05, bootstrap_resamples=200)
        .practical_significance(accuracy=0.95)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.pairwise[0].is_practically_significant is False


def test_default_practical_significance_requires_nonzero_effect(tmp_path) -> None:
    X, y = _strong_binary_data()

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("dummy_a", DummyClassifier(strategy="most_frequent")),
            ("dummy_b", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy")
        .design(cv=5, cv_repeats=2, random_state=21)
        .inference(alpha=0.05, bootstrap_resamples=200)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.pairwise[0].delta == pytest.approx(0.0)
    assert report.pairwise[0].is_practically_significant is False


def test_design_rejects_cv_repeats_below_one() -> None:
    with pytest.raises(ValidationError, match="cv_repeats must be at least 1"):
        ExperimentalHarness().design(cv_repeats=0)


def test_inference_rejects_alpha_out_of_range() -> None:
    with pytest.raises(ValidationError, match="alpha must be between 0 and 1"):
        ExperimentalHarness().inference(alpha=0.0)
    with pytest.raises(ValidationError, match="alpha must be between 0 and 1"):
        ExperimentalHarness().inference(alpha=1.0)


def test_practical_significance_rejects_negative_threshold() -> None:
    with pytest.raises(ValidationError, match="must be non-negative"):
        ExperimentalHarness().practical_significance(accuracy=-0.001)


def test_practical_significance_rejects_non_finite_thresholds() -> None:
    with pytest.raises(ValidationError, match="must be finite"):
        ExperimentalHarness().practical_significance(accuracy=float("nan"))
    with pytest.raises(ValidationError, match="must be finite"):
        ExperimentalHarness().practical_significance(accuracy=float("inf"))


def test_guardrails_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError, match="confidence must be between 0 and 1"):
        ExperimentalHarness().guardrails(
            min_improvement={"accuracy": 0.01},
            confidence=1.0,
        )


def test_guardrails_rejects_negative_threshold() -> None:
    with pytest.raises(ValidationError, match="must be non-negative"):
        ExperimentalHarness().guardrails(min_improvement={"accuracy": -0.001})


def test_guardrails_reject_non_finite_thresholds() -> None:
    with pytest.raises(ValidationError, match="must be finite"):
        ExperimentalHarness().guardrails(min_improvement={"accuracy": float("nan")})
    with pytest.raises(ValidationError, match="must be finite"):
        ExperimentalHarness().guardrails(min_improvement={"accuracy": float("inf")})


def test_fasten_rejects_practical_significance_metric_not_in_metrics(tmp_path) -> None:
    X, y = _strong_binary_data()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy")
        .practical_significance(log_loss=0.01)
    )

    with pytest.raises(ConfigurationError, match="practical_significance metric"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_fasten_rejects_guardrail_metric_not_in_metrics(tmp_path) -> None:
    X, y = _strong_binary_data()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy")
        .baseline("dummy")
        .guardrails(min_improvement={"log_loss": 0.01})
    )

    with pytest.raises(ConfigurationError, match="guardrails metric"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_fasten_rejects_guardrails_with_single_model(tmp_path) -> None:
    X, y = _strong_binary_data()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("dummy", DummyClassifier(strategy="most_frequent")))
        .metrics("accuracy")
        .baseline("dummy")
        .guardrails(min_improvement={"accuracy": 0.01})
    )

    with pytest.raises(ConfigurationError, match="at least two models"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_cv_repeats_lockfile_contains_repeat_metadata(tmp_path) -> None:
    X, y = _strong_binary_data()
    lock_path = tmp_path / "statbelt.lock.json"

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("accuracy")
        .design(cv=4, cv_repeats=2, random_state=33)
        .fasten(lock_path=str(lock_path))
        .evaluate()
    )

    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_payload["schema_version"] == 3
    assert lock_payload["cv_repeats"] == 2
    assert len(lock_payload["splits"]) == 8
    assert {split["repeat"] for split in lock_payload["splits"]} == {0, 1}
    assert {split["fold"] for split in lock_payload["splits"]} == {0, 1, 2, 3}

    assert len(report.splits) == 8
    assert len(report.split_metadata) == 8
    assert {meta["repeat"] for meta in report.split_metadata} == {0, 1}


def test_guardrails_support_baseline_checks_for_accuracy_and_log_loss(tmp_path) -> None:
    X, y = _strong_binary_data()

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=4, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=300)
        .baseline("dummy")
        .guardrails(min_improvement={"accuracy": 0.2, "log_loss": 0.3}, confidence=0.9)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.guardrails is not None
    checks = {check.metric: check for check in report.guardrails.checks}
    assert checks["accuracy"].passed is True
    assert checks["log_loss"].passed is True
    assert report.guardrails.overall_pass is True


def test_guardrails_fail_when_required_improvement_is_too_large(tmp_path) -> None:
    X, y = _strong_binary_data()

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy")
        .design(cv=5, cv_repeats=3, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=250)
        .baseline("dummy")
        .guardrails(min_improvement={"accuracy": 0.85}, confidence=0.9)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.guardrails is not None
    assert report.guardrails.overall_pass is False
    assert report.guardrails.checks[0].passed is False


def test_report_exports_json_and_dataframes(tmp_path) -> None:
    X, y = _strong_binary_data()

    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=400)),
            ("dummy", DummyClassifier(strategy="most_frequent")),
        )
        .metrics("accuracy", "log_loss")
        .design(cv=5, cv_repeats=2, random_state=17)
        .inference(alpha=0.05, bootstrap_resamples=200)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    models_df = report.to_dataframe(kind="models")
    assert set(models_df.columns) == {
        "model_name",
        "metric",
        "point_estimate",
        "ci_low",
        "ci_high",
        "alpha",
    }
    assert len(models_df) == 4

    pairwise_df = report.to_dataframe(kind="pairwise")
    assert set(pairwise_df.columns) >= {
        "model_a",
        "model_b",
        "metric",
        "delta",
        "p_value",
        "p_adjusted",
    }
    assert len(pairwise_df) == 2

    payload_text = report.to_json()
    payload = json.loads(payload_text)
    assert "pairwise" in payload
    assert payload["cv_repeats"] == 2

    output_path = tmp_path / "report.json"
    returned_payload = report.to_json(path=output_path)
    assert output_path.read_text(encoding="utf-8") == returned_payload
    with pytest.raises(ValueError, match="kind must be either"):
        report.to_dataframe(kind="unsupported")  # type: ignore[arg-type]


def test_guardrails_require_baseline_before_fasten(tmp_path) -> None:
    X, y = _strong_binary_data()

    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("accuracy")
        .guardrails(min_improvement={"accuracy": 0.1})
    )

    with pytest.raises(ConfigurationError, match="requires baseline"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))


def test_baseline_name_must_exist_in_compare_models(tmp_path) -> None:
    X, y = _strong_binary_data()

    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("binary_classification")
        .compare(("logreg", LogisticRegression(max_iter=400)))
        .metrics("accuracy")
        .baseline("missing_model")
    )

    with pytest.raises(ValidationError, match="Baseline model"):
        harness.fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
