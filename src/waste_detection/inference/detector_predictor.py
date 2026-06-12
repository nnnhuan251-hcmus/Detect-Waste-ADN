from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from waste_detection.inference.predictor_base import PredictorBase
from waste_detection.models.detector_base import DetectorBase
from waste_detection.models.rtdetr_detector import RTDETRDetector
from waste_detection.models.yolov8_detector import YOLOv8Detector


logger = logging.getLogger("DetectorPredictor")


@dataclass
class DetectionPrediction:
    xyxy: List[float]
    confidence: float
    class_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xyxy": [float(value) for value in self.xyxy],
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
        }


class DetectorPredictor(PredictorBase):
    """
    Predictor cho detector Ultralytics.

    Dùng được cho:
    - YOLOv8n binary detector
    - YOLOv8s detector-only
    - RT-DETR detector-only
    """

    def __init__(
        self,
        weights_path: str | Path,
        detector_name: str = "yolov8n",
        family: str | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        max_detections: int = 300,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.detector_name = str(detector_name).lower()
        self.family = str(family or "").lower()
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)

        self.detector = self._build_detector()

    def predict(self, source, **kwargs) -> List[DetectionPrediction]:
        results = self.detector.predict(
            source=source,
            conf=kwargs.get("conf", self.confidence_threshold),
            iou=kwargs.get("iou", self.iou_threshold),
            max_det=kwargs.get("max_det", self.max_detections),
            verbose=kwargs.get("verbose", False),
        )

        raw_predictions = DetectorBase.extract_xyxy_predictions(results)

        predictions = [
            DetectionPrediction(
                xyxy=[float(value) for value in prediction["xyxy"]],
                confidence=float(prediction["confidence"]),
                class_id=int(prediction["class_id"]),
            )
            for prediction in raw_predictions
        ]

        logger.info(
            "Detector predicted %d boxes for source: %s",
            len(predictions),
            type(source),
        )

        return predictions

    def _build_detector(self):
        if "rtdetr" in self.detector_name or "rt-detr" in self.family:
            return RTDETRDetector(weights=self.weights_path)

        return YOLOv8Detector(weights=self.weights_path)
