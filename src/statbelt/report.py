from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
class EvaluationReport:
    task: str
    cv: int
    random_state: int
    bootstrap_resamples: int
    models: list[ModelReport]
    splits: list[tuple[list[int], list[int]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "cv": self.cv,
            "random_state": self.random_state,
            "bootstrap_resamples": self.bootstrap_resamples,
            "models": [model_report.to_dict() for model_report in self.models],
            "splits": [
                {"train": train_indices.copy(), "test": test_indices.copy()}
                for train_indices, test_indices in self.splits
            ],
        }

    def summary(self) -> str:
        confidence_pct = int(round((1 - self.models[0].metrics[next(iter(self.models[0].metrics))].alpha) * 100)) if self.models and self.models[0].metrics else 95
        lines = [
            f"Task: {self.task}",
            f"CV folds: {self.cv}",
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
        return "\n".join(lines).strip()
