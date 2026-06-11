from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ultralytics import RTDETR

from waste_detection.models.detector_base import DetectorBase


logger = logging.getLogger("RTDETRDetector")


class RTDETRDetector(DetectorBase):
    """
    Wrapper cho Ultralytics RT-DETR detector.

    Lưu ý:
    - Trọng số cụ thể cho RT-DETR-R18 cần được xác thực trước khi train chính thức.
    - Nếu dùng weight Ultralytics có sẵn, cần điền đúng vào configs/models/rtdetr_r18.yaml.
    """

    def __init__(self, weights: str | Path | None) -> None:
        if weights is None or str(weights).strip() == "":
            raise ValueError(
                "RT-DETR weights đang để trống. "
                "Hãy cập nhật 'weights' trong configs/models/rtdetr_r18.yaml "
                "sau khi xác thực weight/backend phù hợp."
            )

        self.weights = str(weights)
        logger.info("Đang load RT-DETR model từ: %s", self.weights)
        self.model = RTDETR(self.weights)

    def train(self, **kwargs) -> Any:
        logger.info("Bắt đầu train RT-DETR với args: %s", kwargs)
        return self.model.train(**kwargs)

    def val(self, **kwargs) -> Any:
        logger.info("Bắt đầu validate RT-DETR với args: %s", kwargs)
        return self.model.val(**kwargs)

    def predict(self, source: str | Path, **kwargs) -> Any:
        logger.info("Bắt đầu predict RT-DETR. Source: %s", source)
        return self.model.predict(source=str(source), **kwargs)
