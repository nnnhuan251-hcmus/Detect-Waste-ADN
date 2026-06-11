from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ErrorAnalysisReport:
    total_samples: int = 0
    total_errors: int = 0
    errors_by_true_class: Dict[str, int] = field(default_factory=dict)
    errors_by_pred_class: Dict[str, int] = field(default_factory=dict)
    confusion_pairs: Dict[str, int] = field(default_factory=dict)
    wrong_predictions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0

        return self.total_errors / self.total_samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "total_errors": self.total_errors,
            "error_rate": self.error_rate,
            "errors_by_true_class": self.errors_by_true_class,
            "errors_by_pred_class": self.errors_by_pred_class,
            "confusion_pairs": self.confusion_pairs,
            "wrong_predictions": self.wrong_predictions,
        }


class ClassificationErrorAnalyzer:
    """
    Phân tích lỗi classifier từ y_true, y_pred và prediction rows.
    """

    @staticmethod
    def analyze(
        prediction_rows: List[Dict[str, Any]],
    ) -> ErrorAnalysisReport:
        report = ErrorAnalysisReport(total_samples=len(prediction_rows))

        for row in prediction_rows:
            true_class = row.get("true_class")
            pred_class = row.get("pred_class")
            is_correct = bool(row.get("is_correct", False))

            if is_correct:
                continue

            report.total_errors += 1

            report.errors_by_true_class[true_class] = (
                report.errors_by_true_class.get(true_class, 0) + 1
            )

            report.errors_by_pred_class[pred_class] = (
                report.errors_by_pred_class.get(pred_class, 0) + 1
            )

            pair_key = f"{true_class} -> {pred_class}"
            report.confusion_pairs[pair_key] = report.confusion_pairs.get(pair_key, 0) + 1

            report.wrong_predictions.append(row)

        report.errors_by_true_class = dict(
            sorted(report.errors_by_true_class.items())
        )

        report.errors_by_pred_class = dict(
            sorted(report.errors_by_pred_class.items())
        )

        report.confusion_pairs = dict(
            sorted(
                report.confusion_pairs.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        return report
