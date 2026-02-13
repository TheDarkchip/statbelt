from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd


@dataclass(frozen=True)
class MetricInterval:
    point_estimate: float
    ci_low: float
    ci_high: float
    alpha: float

    def to_dict(self) -> dict[str, float]:
        return {
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class ModelReport:
    model_name: str
    metrics: dict[str, MetricInterval]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "metrics": {
                metric_name: interval.to_dict()
                for metric_name, interval in self.metrics.items()
            },
        }


@dataclass(frozen=True)
class PairwiseComparison:
    model_a: str
    model_b: str
    metric: str
    delta: float
    ci_low: float
    ci_high: float
    alpha: float
    p_value: float
    p_adjusted: float
    reject_null: bool
    is_practically_significant: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "alpha": self.alpha,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "reject_null": self.reject_null,
            "is_practically_significant": self.is_practically_significant,
        }


@dataclass(frozen=True)
class GuardrailCheck:
    challenger_model: str
    baseline_model: str
    metric: str
    min_improvement: float
    confidence: float
    improvement_point_estimate: float
    ci_low: float
    ci_high: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenger_model": self.challenger_model,
            "baseline_model": self.baseline_model,
            "metric": self.metric,
            "min_improvement": self.min_improvement,
            "confidence": self.confidence,
            "improvement_point_estimate": self.improvement_point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class GuardrailReport:
    overall_pass: bool
    checks: list[GuardrailCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pass": self.overall_pass,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class EvaluationReport:
    task: str
    cv: int
    cv_repeats: int
    random_state: int
    bootstrap_resamples: int
    models: list[ModelReport]
    splits: list[tuple[list[int], list[int]]]
    split_metadata: list[dict[str, int]]
    pairwise: list[PairwiseComparison] = field(default_factory=list)
    guardrails: GuardrailReport | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "cv": self.cv,
            "cv_repeats": self.cv_repeats,
            "random_state": self.random_state,
            "bootstrap_resamples": self.bootstrap_resamples,
            "models": [model_report.to_dict() for model_report in self.models],
            "splits": [
                {"train": train_indices.copy(), "test": test_indices.copy()}
                for train_indices, test_indices in self.splits
            ],
            "split_metadata": [meta.copy() for meta in self.split_metadata],
            "pairwise": [comparison.to_dict() for comparison in self.pairwise],
        }
        if self.guardrails is not None:
            payload["guardrails"] = self.guardrails.to_dict()
        return payload

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2) + "\n"
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_dataframe(self, kind: Literal["models", "pairwise"] = "models") -> pd.DataFrame:
        if kind == "models":
            rows: list[dict[str, Any]] = []
            for model_report in self.models:
                for metric_name, interval in model_report.metrics.items():
                    rows.append(
                        {
                            "model_name": model_report.model_name,
                            "metric": metric_name,
                            "point_estimate": interval.point_estimate,
                            "ci_low": interval.ci_low,
                            "ci_high": interval.ci_high,
                            "alpha": interval.alpha,
                        }
                    )
            return pd.DataFrame(rows)

        if kind == "pairwise":
            return pd.DataFrame([comparison.to_dict() for comparison in self.pairwise])

        raise ValueError("kind must be either 'models' or 'pairwise'.")

    def summary(self) -> str:
        confidence_pct = 95
        if self.models and self.models[0].metrics:
            alpha = self.models[0].metrics[next(iter(self.models[0].metrics))].alpha
            confidence_pct = int(round((1 - alpha) * 100))

        lines = [
            f"Task: {self.task}",
            f"CV folds: {self.cv}",
            f"CV repeats: {self.cv_repeats}",
            f"Bootstrap resamples: {self.bootstrap_resamples}",
            f"Confidence interval: {confidence_pct}%",
            "",
        ]

        for model_report in self.models:
            lines.append(f"Model: {model_report.model_name}")
            for metric_name, interval in model_report.metrics.items():
                lines.append(
                    f"  {metric_name}: {interval.point_estimate:.4f} "
                    f"(CI {interval.ci_low:.4f}, {interval.ci_high:.4f})"
                )
            lines.append("")

        if self.pairwise:
            lines.append("Pairwise comparisons:")
            for comparison in self.pairwise:
                status = "reject" if comparison.reject_null else "retain"
                practical = "practical" if comparison.is_practically_significant else "not practical"
                lines.append(
                    f"  {comparison.model_a} - {comparison.model_b} [{comparison.metric}]: "
                    f"{comparison.delta:.4f} (CI {comparison.ci_low:.4f}, {comparison.ci_high:.4f}), "
                    f"p_adj={comparison.p_adjusted:.4g}, {status}, {practical}"
                )
            lines.append("")

        if self.guardrails is not None:
            status = "PASS" if self.guardrails.overall_pass else "FAIL"
            lines.append(f"Guardrails: {status}")
            for check in self.guardrails.checks:
                check_status = "PASS" if check.passed else "FAIL"
                lines.append(
                    f"  {check.challenger_model} vs {check.baseline_model} [{check.metric}]: "
                    f"{check_status} (min {check.min_improvement:.4f}, "
                    f"CI {check.ci_low:.4f}, {check.ci_high:.4f})"
                )

        return "\n".join(lines).strip()
