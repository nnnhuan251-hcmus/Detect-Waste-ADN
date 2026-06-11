from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from waste_detection.models.detector_base import DetectorBase


logger = logging.getLogger("YOLOv8Detector")


class YOLOv8Detector(DetectorBase):
    """
    Wrapper cho Ultralytics YOLO detector.

    Dùng cho:
    - YOLOv8n binary detector trong hybrid pipeline.
    - YOLOv8s detector-only 7 class.
    """

    def __init__(self, weights: str | Path) -> None:
        self.weights = str(weights)
        logger.info("Đang load YOLO model từ: %s", self.weights)
        self.model = YOLO(self.weights)

    def train(self, **kwargs) -> Any:
        logger.info("Bắt đầu train YOLO với args: %s", kwargs)
        return self.model.train(**kwargs)

    def val(self, **kwargs) -> Any:
        logger.info("Bắt đầu validate YOLO với args: %s", kwargs)
        return self.model.val(**kwargs)

    def predict(self, source: str | Path, **kwargs) -> Any:
        logger.info("Bắt đầu predict YOLO. Source: %s", source)
        return self.model.predict(source=str(source), **kwargs)
