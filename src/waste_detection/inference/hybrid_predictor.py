from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from waste_detection.inference.classifier_predictor import (
    ClassificationPrediction,
    ClassifierPredictor,
)
from waste_detection.inference.detector_predictor import (
    DetectionPrediction,
    DetectorPredictor,
)
from waste_detection.inference.predictor_base import PredictorBase

from waste_detection.postprocessing.tta import HorizontalFlipTTA
from waste_detection.postprocessing.wbf import WeightedBoxesFusionPostProcessor

logger = logging.getLogger("HybridPredictor")


@dataclass
class HybridPrediction:
    xyxy: List[float]
    detector_confidence: float
    classifier_confidence: float
    final_confidence: float
    class_id: int
    class_name: str
    detector_class_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xyxy": self.xyxy,
            "detector_confidence": self.detector_confidence,
            "classifier_confidence": self.classifier_confidence,
            "final_confidence": self.final_confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "detector_class_id": self.detector_class_id,
        }


class HybridPredictor(PredictorBase):
    """
    Hybrid inference pipeline:

    image
      -> detector predicts boxes
      -> crop predicted boxes
      -> EfficientNet-B0 classifies crops
      -> final prediction
    """

    def __init__(
        self,
        detector_weights: str | Path,
        classifier_weights: str | Path,
        class_names: List[str],
        detector_name: str = "yolov8n",
        detector_family: str | None = None,
        detector_confidence_threshold: float = 0.25,
        detector_iou_threshold: float = 0.50,
        max_detections: int = 300,
        classifier_image_size: int = 224,
        crop_margin: float = 0.10,
        final_confidence_rule: str = "multiply",
        final_confidence_threshold: float = 0.25,
        device: torch.device | str = "cpu",

        use_tta: bool = False,
        use_wbf: bool = False,
        wbf_iou_threshold: float = 0.55,
        wbf_skip_box_threshold: float = 0.001,
    ) -> None:
        self.class_names = class_names
        self.crop_margin = crop_margin
        self.final_confidence_rule = final_confidence_rule
        self.final_confidence_threshold = final_confidence_threshold
        self.device = torch.device(device)

        self.use_tta = bool(use_tta)
        self.use_wbf = bool(use_wbf)
        self.wbf_iou_threshold = float(wbf_iou_threshold)
        self.wbf_skip_box_threshold = float(wbf_skip_box_threshold)

        self.detector = DetectorPredictor(
            weights_path=detector_weights,
            detector_name=detector_name,
            family=detector_family,
            confidence_threshold=detector_confidence_threshold,
            iou_threshold=detector_iou_threshold,
            max_detections=max_detections,
        )

        self.classifier = ClassifierPredictor(
            weights_path=classifier_weights,
            class_names=class_names,
            image_size=classifier_image_size,
            device=self.device,
            input_is_bgr=True,
        )

    def predict(self, source: str | Path, **kwargs) -> List[HybridPrediction]:
        image_path = Path(source)

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

        image_height, image_width = image_bgr.shape[:2]

        use_tta = bool(kwargs.get("use_tta", self.use_tta))
        use_wbf = bool(kwargs.get("use_wbf", self.use_wbf))

        if use_tta:
            detections = self._predict_detector_with_tta(
                image_bgr=image_bgr,
                image_width=image_width,
                image_height=image_height,
                use_wbf=use_wbf,
            )
        else:
            detections = self.detector.predict(image_path)

        hybrid_predictions: List[HybridPrediction] = []

        for detection in detections:
            crop = self._crop_from_detection(
                image_bgr=image_bgr,
                detection=detection,
                image_width=image_width,
                image_height=image_height,
            )

            if crop is None:
                continue

            classification = self.classifier.predict_one(crop)

            final_confidence = self._compute_final_confidence(
                detector_confidence=detection.confidence,
                classifier_confidence=classification.confidence,
            )

            if final_confidence < self.final_confidence_threshold:
                continue

            hybrid_predictions.append(
                HybridPrediction(
                    xyxy=detection.xyxy,
                    detector_confidence=detection.confidence,
                    classifier_confidence=classification.confidence,
                    final_confidence=final_confidence,
                    class_id=classification.class_id,
                    class_name=classification.class_name,
                    detector_class_id=detection.class_id,
                )
            )

        logger.info(
            "Hybrid inference done: image=%s, predictions=%d",
            image_path,
            len(hybrid_predictions),
        )

        return hybrid_predictions
        
    def _predict_detector_with_tta(
        self,
        image_bgr: np.ndarray,
        image_width: int,
        image_height: int,
        use_wbf: bool,
    ) -> List[DetectionPrediction]:
        tta = HorizontalFlipTTA()
        predictions_per_view: List[List[DetectionPrediction]] = []

        for view_name, view_image in tta.create_views(image_bgr):
            view_predictions = self.detector.predict(view_image)

            restored_predictions = tta.restore_predictions(
                view_name=view_name,
                predictions=view_predictions,
                image_width=image_width,
                image_height=image_height,
            )

            predictions_per_view.append(restored_predictions)

        if use_wbf:
            return WeightedBoxesFusionPostProcessor.fuse(
                predictions_per_view=predictions_per_view,
                image_width=image_width,
                image_height=image_height,
                iou_thr=self.wbf_iou_threshold,
                skip_box_thr=self.wbf_skip_box_threshold,
            )

        flattened_predictions = [
            prediction
            for predictions in predictions_per_view
            for prediction in predictions
        ]

        flattened_predictions.sort(
            key=lambda prediction: prediction.confidence,
            reverse=True,
        )

        return flattened_predictions
        
    def _crop_from_detection(
        self,
        image_bgr: np.ndarray,
        detection: DetectionPrediction,
        image_width: int,
        image_height: int,
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = detection.xyxy

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            return None

        margin_x = box_width * self.crop_margin
        margin_y = box_height * self.crop_margin

        crop_x1 = int(max(0, x1 - margin_x))
        crop_y1 = int(max(0, y1 - margin_y))
        crop_x2 = int(min(image_width, x2 + margin_x))
        crop_y2 = int(min(image_height, y2 + margin_y))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0:
            return None

        return crop

    def _compute_final_confidence(
        self,
        detector_confidence: float,
        classifier_confidence: float,
    ) -> float:
        if self.final_confidence_rule == "multiply":
            return float(detector_confidence * classifier_confidence)

        if self.final_confidence_rule == "classifier":
            return float(classifier_confidence)

        if self.final_confidence_rule == "detector":
            return float(detector_confidence)

        if self.final_confidence_rule == "average":
            return float((detector_confidence + classifier_confidence) / 2.0)

        raise ValueError(
            f"final_confidence_rule không hợp lệ: {self.final_confidence_rule}"
        )
