from __future__ import annotations

from typing import Any, Dict


class DetectionMetrics:
    """
    Helper trích xuất metrics từ Ultralytics validation result.
    """

    @staticmethod
    def from_ultralytics(metrics: Any) -> Dict[str, Any]:
        box = getattr(metrics, "box", None)

        if box is None:
            return {
                "map50_95": None,
                "map50": None,
                "map75": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "per_class_map": [],
            }

        precision = DetectionMetrics._safe_float(getattr(box, "mp", None))
        recall = DetectionMetrics._safe_float(getattr(box, "mr", None))

        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        maps = getattr(box, "maps", [])

        try:
            per_class_map = [float(value) for value in maps]
        except TypeError:
            per_class_map = []

        return {
            "map50_95": DetectionMetrics._safe_float(getattr(box, "map", None)),
            "map50": DetectionMetrics._safe_float(getattr(box, "map50", None)),
            "map75": DetectionMetrics._safe_float(getattr(box, "map75", None)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_map": per_class_map,
        }

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None
