from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ClassificationMetrics:
    """
    Metrics cho crop classifier.
    """

    @staticmethod
    def compute(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
    ) -> Dict:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "macro_precision": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
