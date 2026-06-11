from waste_detection.inference.predictor_base import PredictorBase

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

from waste_detection.data.transforms import ClassifierTransformFactory
from waste_detection.models.efficientnet_classifier import EfficientNetB0Classifier


logger = logging.getLogger("ClassifierPredictor")


@dataclass
class ClassificationPrediction:
    class_id: int
    class_name: str
    confidence: float
    probabilities: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class ClassifierPredictor(PredictorBase):
    """
    Predictor cho EfficientNet-B0 classifier.

    Input:
    - crop image dạng numpy array BGR hoặc RGB.

    Output:
    - class_id
    - class_name
    - confidence
    """

    def __init__(
        self,
        weights_path: str | Path,
        class_names: List[str],
        image_size: int = 224,
        device: torch.device | str = "cpu",
        input_is_bgr: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.class_names = list(class_names)
        self.image_size = image_size
        self.device = torch.device(device)
        self.input_is_bgr = input_is_bgr

        self.model, checkpoint_class_names = EfficientNetB0Classifier.load_from_checkpoint(
            self.weights_path,
            map_location=self.device,
        )

        if checkpoint_class_names and checkpoint_class_names != self.class_names:
            logger.warning(
                "Class names trong checkpoint khác config. checkpoint=%s, config=%s",
                checkpoint_class_names,
                self.class_names,
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = ClassifierTransformFactory.build_eval_transform(
            image_size=self.image_size
        )

    def predict_one(self, crop_image: np.ndarray) -> ClassificationPrediction:
        image = self._numpy_to_pil(crop_image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities_tensor = torch.softmax(logits, dim=1)[0]

        probabilities = probabilities_tensor.detach().cpu().tolist()
        class_id = int(torch.argmax(probabilities_tensor).item())
        confidence = float(probabilities[class_id])

        class_name = (
            self.class_names[class_id]
            if class_id < len(self.class_names)
            else f"class_{class_id}"
        )

        return ClassificationPrediction(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            probabilities=[float(value) for value in probabilities],
        )

    def predict_batch(
        self,
        crop_images: List[np.ndarray],
    ) -> List[ClassificationPrediction]:
        return [self.predict_one(crop_image) for crop_image in crop_images]

    def predict(self, source, **kwargs):
        """
        Cổng predict chung theo PredictorBase.
    
        Hỗ trợ:
        - source là 1 crop numpy array: trả về 1 ClassificationPrediction.
        - source là list crop numpy arrays: trả về list ClassificationPrediction.
        - source là đường dẫn ảnh: đọc ảnh và phân loại ảnh đó như một crop.
        """
        if isinstance(source, list):
            return self.predict_batch(source)
    
        if isinstance(source, np.ndarray):
            return self.predict_one(source)
    
        image_path = Path(source)
        image = cv2.imread(str(image_path))
    
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh crop: {image_path}")
    
        return self.predict_one(image)

    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Crop image phải có shape HxWx3. Nhận được shape={image.shape}"
            )

        if self.input_is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(image)
