import math

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from statbelt import ExperimentalHarness


def _multiclass_data() -> tuple[object, object]:
    return make_classification(
        n_samples=180,
        n_features=10,
        n_informative=7,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=77,
    )


def test_multiclass_evaluation_returns_full_metric_intervals(tmp_path) -> None:
    X, y = _multiclass_data()
    requested_metrics = (
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
    )
    report = (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(
            ("logreg", LogisticRegression(max_iter=600)),
            ("rf", RandomForestClassifier(n_estimators=40, random_state=77)),
        )
        .metrics(*requested_metrics)
        .design(cv=4, random_state=77)
        .inference(alpha=0.05, bootstrap_resamples=120)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
        .evaluate()
    )

    assert report.task == "multiclass_classification"
    assert len(report.models) == 2
    assert len(report.pairwise) == len(requested_metrics)

    for model_report in report.models:
        assert tuple(model_report.metrics.keys()) == requested_metrics
        for interval in model_report.metrics.values():
            assert math.isfinite(interval.point_estimate)
            assert math.isfinite(interval.ci_low)
            assert math.isfinite(interval.ci_high)
            assert interval.ci_low <= interval.point_estimate <= interval.ci_high


def test_multiclass_evaluation_is_deterministic_for_fixed_seed(tmp_path) -> None:
    X, y = _multiclass_data()
    harness = (
        ExperimentalHarness()
        .data(X, y)
        .task("multiclass_classification")
        .compare(("logreg", LogisticRegression(max_iter=600)))
        .metrics("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")
        .design(cv=4, cv_repeats=2, random_state=33)
        .inference(alpha=0.05, bootstrap_resamples=100)
        .fasten(lock_path=str(tmp_path / "statbelt.lock.json"))
    )

    first = harness.evaluate()
    second = harness.evaluate()
    assert first.to_dict() == second.to_dict()
