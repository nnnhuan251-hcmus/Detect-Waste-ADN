from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class DetectorBase(ABC):
    """
    Base interface cho detector.

    Các detector cụ thể như YOLOv8 hoặc RT-DETR phải triển khai:
    - train
    - val
    - predict
    """

    @abstractmethod
    def train(self, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def val(self, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self, source: Any, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def extract_xyxy_predictions(results: Any) -> List[Dict[str, Any]]:
        """
        Chuyển output Ultralytics predict thành list dict đơn giản.

        Output:
        [
            {
                "xyxy": [x1, y1, x2, y2],
                "confidence": 0.91,
                "class_id": 0
            }
        ]
        """
        predictions: List[Dict[str, Any]] = []

        for result in results:
            boxes = getattr(result, "boxes", None)

            if boxes is None:
                continue

            xyxy_list = boxes.xyxy.detach().cpu().numpy().tolist()
            conf_list = boxes.conf.detach().cpu().numpy().tolist()
            cls_list = boxes.cls.detach().cpu().numpy().astype(int).tolist()

            for xyxy, confidence, class_id in zip(xyxy_list, conf_list, cls_list):
                predictions.append(
                    {
                        "xyxy": [float(value) for value in xyxy],
                        "confidence": float(confidence),
                        "class_id": int(class_id),
                    }
                )

        return predictions
