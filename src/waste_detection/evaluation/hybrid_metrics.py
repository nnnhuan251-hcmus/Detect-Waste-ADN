from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class HybridMetricReport:
    iou_threshold: float
    num_images: int = 0
    num_ground_truths: int = 0
    num_predictions: int = 0

    detection_tp: int = 0
    detection_fp: int = 0
    detection_fn: int = 0

    classification_correct_on_matched: int = 0
    classification_wrong_on_matched: int = 0

    localization_errors: int = 0
    classification_errors: int = 0
    missed_objects: int = 0
    false_positives: int = 0

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    classification_accuracy_on_matched: float = 0.0

    map50: float = 0.0
    map50_95: float = 0.0
    per_class_ap50: Dict[str, float] = field(default_factory=dict)
    per_class_ap50_95: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "num_images": self.num_images,
            "num_ground_truths": self.num_ground_truths,
            "num_predictions": self.num_predictions,
            "detection_tp": self.detection_tp,
            "detection_fp": self.detection_fp,
            "detection_fn": self.detection_fn,
            "classification_correct_on_matched": self.classification_correct_on_matched,
            "classification_wrong_on_matched": self.classification_wrong_on_matched,
            "localization_errors": self.localization_errors,
            "classification_errors": self.classification_errors,
            "missed_objects": self.missed_objects,
            "false_positives": self.false_positives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "classification_accuracy_on_matched": self.classification_accuracy_on_matched,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "per_class_ap50": self.per_class_ap50,
            "per_class_ap50_95": self.per_class_ap50_95,
        }


class HybridMetrics:
    """
    Metrics cho hybrid end-to-end.

    Input prediction format:
    {
        "image_id": int,
        "xyxy": [x1, y1, x2, y2],
        "class_id": int,
        "class_name": str,
        "final_confidence": float,
        "detector_confidence": float,
        "classifier_confidence": float
    }

    Input ground truth format:
    {
        "image_id": int,
        "xyxy": [x1, y1, x2, y2],
        "class_id": int,
        "class_name": str
    }
    """

    @staticmethod
    def compute(
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        class_names: List[str],
        iou_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        report = HybridMetricReport(
            iou_threshold=iou_threshold,
            num_images=len({gt["image_id"] for gt in ground_truths}),
            num_ground_truths=len(ground_truths),
            num_predictions=len(predictions),
        )

        matches, unmatched_predictions, unmatched_ground_truths = (
            HybridMetrics.match_predictions_to_ground_truths(
                predictions=predictions,
                ground_truths=ground_truths,
                iou_threshold=iou_threshold,
                class_aware=False,
            )
        )

        report.detection_tp = len(matches)
        report.detection_fp = len(unmatched_predictions)
        report.detection_fn = len(unmatched_ground_truths)

        report.false_positives = report.detection_fp
        report.missed_objects = report.detection_fn

        for match in matches:
            prediction = match["prediction"]
            ground_truth = match["ground_truth"]

            if int(prediction["class_id"]) == int(ground_truth["class_id"]):
                report.classification_correct_on_matched += 1
            else:
                report.classification_wrong_on_matched += 1

        report.classification_errors = report.classification_wrong_on_matched
        report.localization_errors = report.detection_fn + report.detection_fp

        report.precision = HybridMetrics._safe_divide(
            report.detection_tp,
            report.detection_tp + report.detection_fp,
        )

        report.recall = HybridMetrics._safe_divide(
            report.detection_tp,
            report.detection_tp + report.detection_fn,
        )

        report.f1 = HybridMetrics._f1(report.precision, report.recall)

        matched_count = (
            report.classification_correct_on_matched
            + report.classification_wrong_on_matched
        )

        report.classification_accuracy_on_matched = HybridMetrics._safe_divide(
            report.classification_correct_on_matched,
            matched_count,
        )

        report.per_class_ap50 = HybridMetrics.compute_per_class_ap(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            iou_threshold=0.50,
        )

        report.map50 = float(np.mean(list(report.per_class_ap50.values()))) if report.per_class_ap50 else 0.0

        thresholds = np.arange(0.50, 1.00, 0.05)

        per_class_ap_all_thresholds: Dict[str, List[float]] = {
            class_name: [] for class_name in class_names
        }

        for threshold in thresholds:
            per_class_ap = HybridMetrics.compute_per_class_ap(
                predictions=predictions,
                ground_truths=ground_truths,
                class_names=class_names,
                iou_threshold=float(threshold),
            )

            for class_name in class_names:
                per_class_ap_all_thresholds[class_name].append(
                    per_class_ap.get(class_name, 0.0)
                )

        report.per_class_ap50_95 = {
            class_name: float(np.mean(values)) if values else 0.0
            for class_name, values in per_class_ap_all_thresholds.items()
        }

        report.map50_95 = (
            float(np.mean(list(report.per_class_ap50_95.values())))
            if report.per_class_ap50_95
            else 0.0
        )

        return report.to_dict()

    @staticmethod
    def match_predictions_to_ground_truths(
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        iou_threshold: float,
        class_aware: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        predictions_by_image = defaultdict(list)
        ground_truths_by_image = defaultdict(list)

        for prediction in predictions:
            predictions_by_image[prediction["image_id"]].append(prediction)

        for ground_truth in ground_truths:
            ground_truths_by_image[ground_truth["image_id"]].append(ground_truth)

        matches: List[Dict[str, Any]] = []
        unmatched_predictions: List[Dict[str, Any]] = []
        unmatched_ground_truths: List[Dict[str, Any]] = []

        image_ids = set(predictions_by_image.keys()).union(ground_truths_by_image.keys())

        for image_id in image_ids:
            image_predictions = sorted(
                predictions_by_image.get(image_id, []),
                key=lambda item: float(item.get("final_confidence", 0.0)),
                reverse=True,
            )

            image_ground_truths = ground_truths_by_image.get(image_id, [])
            used_gt_indices = set()

            for prediction in image_predictions:
                best_iou = 0.0
                best_gt_index = None

                for gt_index, ground_truth in enumerate(image_ground_truths):
                    if gt_index in used_gt_indices:
                        continue

                    if class_aware and int(prediction["class_id"]) != int(ground_truth["class_id"]):
                        continue

                    iou = HybridMetrics.compute_iou_xyxy(
                        prediction["xyxy"],
                        ground_truth["xyxy"],
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_index = gt_index

                if best_gt_index is not None and best_iou >= iou_threshold:
                    used_gt_indices.add(best_gt_index)
                    matches.append(
                        {
                            "image_id": image_id,
                            "prediction": prediction,
                            "ground_truth": image_ground_truths[best_gt_index],
                            "iou": best_iou,
                        }
                    )
                else:
                    unmatched_predictions.append(prediction)

            for gt_index, ground_truth in enumerate(image_ground_truths):
                if gt_index not in used_gt_indices:
                    unmatched_ground_truths.append(ground_truth)

        return matches, unmatched_predictions, unmatched_ground_truths

    @staticmethod
    def compute_per_class_ap(
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        class_names: List[str],
        iou_threshold: float,
    ) -> Dict[str, float]:
        per_class_ap: Dict[str, float] = {}

        for class_id, class_name in enumerate(class_names):
            class_predictions = [
                prediction
                for prediction in predictions
                if int(prediction["class_id"]) == class_id
            ]

            class_ground_truths = [
                ground_truth
                for ground_truth in ground_truths
                if int(ground_truth["class_id"]) == class_id
            ]

            ap = HybridMetrics.compute_average_precision_for_class(
                predictions=class_predictions,
                ground_truths=class_ground_truths,
                iou_threshold=iou_threshold,
            )

            per_class_ap[class_name] = ap

        return per_class_ap

    @staticmethod
    def compute_average_precision_for_class(
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        iou_threshold: float,
    ) -> float:
        if not ground_truths:
            return 0.0

        predictions = sorted(
            predictions,
            key=lambda item: float(item.get("final_confidence", 0.0)),
            reverse=True,
        )

        ground_truths_by_image = defaultdict(list)

        for ground_truth in ground_truths:
            ground_truths_by_image[ground_truth["image_id"]].append(ground_truth)

        used_gt = {
            image_id: set()
            for image_id in ground_truths_by_image.keys()
        }

        true_positives = np.zeros(len(predictions))
        false_positives = np.zeros(len(predictions))

        for pred_index, prediction in enumerate(predictions):
            image_id = prediction["image_id"]
            candidate_ground_truths = ground_truths_by_image.get(image_id, [])

            best_iou = 0.0
            best_gt_index = None

            for gt_index, ground_truth in enumerate(candidate_ground_truths):
                if gt_index in used_gt[image_id]:
                    continue

                iou = HybridMetrics.compute_iou_xyxy(
                    prediction["xyxy"],
                    ground_truth["xyxy"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            if best_gt_index is not None and best_iou >= iou_threshold:
                true_positives[pred_index] = 1
                used_gt[image_id].add(best_gt_index)
            else:
                false_positives[pred_index] = 1

        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)

        recalls = cumulative_tp / max(1, len(ground_truths))
        precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)

        return HybridMetrics.compute_ap_101_point(
            recalls=recalls,
            precisions=precisions,
        )

    @staticmethod
    def compute_ap_101_point(
        recalls: np.ndarray,
        precisions: np.ndarray,
    ) -> float:
        recall_levels = np.linspace(0.0, 1.0, 101)
        interpolated_precisions = []

        for recall_level in recall_levels:
            precisions_at_recall = precisions[recalls >= recall_level]

            if precisions_at_recall.size == 0:
                interpolated_precisions.append(0.0)
            else:
                interpolated_precisions.append(float(np.max(precisions_at_recall)))

        return float(np.mean(interpolated_precisions))

    @staticmethod
    def coco_bbox_to_xyxy(bbox: List[float]) -> List[float]:
        x, y, width, height = map(float, bbox)

        return [
            x,
            y,
            x + width,
            y + height,
        ]

    @staticmethod
    def compute_iou_xyxy(
        box_a: List[float],
        box_b: List[float],
    ) -> float:
        ax1, ay1, ax2, ay2 = map(float, box_a)
        bx1, by1, bx2, by2 = map(float, box_b)

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0

        return float(2 * precision * recall / (precision + recall))
