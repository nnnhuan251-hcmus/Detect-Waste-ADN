from __future__ import annotations

from typing import List

from waste_detection.inference.detector_predictor import DetectionPrediction
from waste_detection.postprocessing.bbox_utils import BBoxUtils


class WeightedBoxesFusionPostProcessor:
    """
    Weighted Boxes Fusion cho detection predictions.

    Dùng cho:
    - Gộp prediction từ TTA views.
    - Gộp prediction từ nhiều detector nếu sau này cần ensemble.
    """

    @staticmethod
    def fuse(
        predictions_per_view: List[List[DetectionPrediction]],
        image_width: int,
        image_height: int,
        iou_thr: float = 0.55,
        skip_box_thr: float = 0.001,
    ) -> List[DetectionPrediction]:
        try:
            from ensemble_boxes import weighted_boxes_fusion
        except ImportError as error:
            raise ImportError(
                "Thiếu package ensemble-boxes. "
                "Hãy thêm `ensemble-boxes` vào requirements.txt."
            ) from error

        boxes_list = []
        scores_list = []
        labels_list = []

        for predictions in predictions_per_view:
            boxes = []
            scores = []
            labels = []

            for prediction in predictions:
                boxes.append(
                    BBoxUtils.normalize_xyxy(
                        box=prediction.xyxy,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )
                scores.append(float(prediction.confidence))
                labels.append(int(prediction.class_id))

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        if not any(boxes_list):
            return []

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=None,
            iou_thr=float(iou_thr),
            skip_box_thr=float(skip_box_thr),
        )

        fused_predictions: List[DetectionPrediction] = []

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            denormalized_box = BBoxUtils.denormalize_xyxy(
                box=list(box),
                image_width=image_width,
                image_height=image_height,
            )

            fused_predictions.append(
                DetectionPrediction(
                    xyxy=denormalized_box,
                    confidence=float(score),
                    class_id=int(label),
                )
            )

        fused_predictions.sort(
            key=lambda prediction: prediction.confidence,
            reverse=True,
        )

        return fused_predictions
