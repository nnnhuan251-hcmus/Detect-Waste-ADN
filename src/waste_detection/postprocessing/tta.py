from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from waste_detection.inference.detector_predictor import DetectionPrediction
from waste_detection.postprocessing.bbox_utils import BBoxUtils


class HorizontalFlipTTA:
    """
    Test-time augmentation đơn giản cho detector:
    - original image
    - horizontal flipped image

    Sau predict trên ảnh flip, bbox được restore về hệ tọa độ ảnh gốc.
    """

    ORIGINAL = "original"
    HFLIP = "hflip"

    def create_views(self, image_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        return [
            (self.ORIGINAL, image_bgr),
            (self.HFLIP, cv2.flip(image_bgr, 1)),
        ]

    def restore_predictions(
        self,
        view_name: str,
        predictions: List[DetectionPrediction],
        image_width: int,
        image_height: int,
    ) -> List[DetectionPrediction]:
        restored_predictions: List[DetectionPrediction] = []

        for prediction in predictions:
            if view_name == self.ORIGINAL:
                restored_box = prediction.xyxy

            elif view_name == self.HFLIP:
                restored_box = BBoxUtils.flip_xyxy_horizontal(
                    box=prediction.xyxy,
                    image_width=image_width,
                )

            else:
                raise ValueError(f"TTA view không hợp lệ: {view_name}")

            restored_box = BBoxUtils.clip_xyxy(
                box=restored_box,
                image_width=image_width,
                image_height=image_height,
            )

            restored_predictions.append(
                DetectionPrediction(
                    xyxy=restored_box,
                    confidence=prediction.confidence,
                    class_id=prediction.class_id,
                )
            )

        return restored_predictions
