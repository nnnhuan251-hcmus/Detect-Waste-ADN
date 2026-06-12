from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from waste_detection.utils.io import IOUtils


@dataclass
class HybridModelInfo:
    """
    Metadata mô tả hybrid model.

    Lưu ý:
    - Hybrid YOLOv8n + EfficientNet-B0 không phải một nn.Module end-to-end train chung.
    - Detector và classifier được train riêng.
    - End-to-end chỉ xảy ra ở inference/evaluation:
        full image -> detector -> predicted crops -> classifier.
    """

    detector_name: str
    classifier_name: str
    detector_mode: str
    num_detector_classes: int
    num_classifier_classes: int
    class_names: List[str]
    final_confidence_rule: str = "multiply"
    crop_margin: float = 0.10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "classifier_name": self.classifier_name,
            "detector_mode": self.detector_mode,
            "num_detector_classes": self.num_detector_classes,
            "num_classifier_classes": self.num_classifier_classes,
            "class_names": self.class_names,
            "final_confidence_rule": self.final_confidence_rule,
            "crop_margin": self.crop_margin,
        }


class HybridWasteModel:
    """
    Mô tả cấp model cho pipeline hybrid.

    Class này không chứa training loop.
    Logic inference thực tế nằm ở:

        src/waste_detection/inference/hybrid_predictor.py

    Mục đích của file này:
    - Làm rõ kiến trúc hybrid trong tầng models.
    - Giúp report/checkpoint/config có object mô tả nhất quán.
    - Tránh để scaffold `hybrid_model.py` trống.
    """

    def __init__(
        self,
        detector_name: str,
        classifier_name: str,
        detector_mode: str,
        class_names: List[str],
        final_confidence_rule: str = "multiply",
        crop_margin: float = 0.10,
    ) -> None:
        self.detector_name = detector_name
        self.classifier_name = classifier_name
        self.detector_mode = detector_mode
        self.class_names = list(class_names)
        self.final_confidence_rule = final_confidence_rule
        self.crop_margin = crop_margin

        if detector_mode == "binary_waste":
            num_detector_classes = 1
        elif detector_mode == "seven_class":
            num_detector_classes = len(class_names)
        else:
            raise ValueError(f"detector_mode không hợp lệ: {detector_mode}")

        self.info = HybridModelInfo(
            detector_name=detector_name,
            classifier_name=classifier_name,
            detector_mode=detector_mode,
            num_detector_classes=num_detector_classes,
            num_classifier_classes=len(class_names),
            class_names=self.class_names,
            final_confidence_rule=final_confidence_rule,
            crop_margin=crop_margin,
        )

    @classmethod
    def from_configs(
        cls,
        model_config: Dict[str, Any],
        class_names: List[str],
    ) -> "HybridWasteModel":
        detector_config = model_config.get("detector", {})
        classifier_config = model_config.get("classifier", {})
        hybrid_config = model_config.get("hybrid_inference", {})

        return cls(
            detector_name=str(detector_config.get("name", "yolov8n")),
            classifier_name=str(classifier_config.get("name", "efficientnet_b0")),
            detector_mode=str(detector_config.get("detector_mode", "binary_waste")),
            class_names=class_names,
            final_confidence_rule=str(
                hybrid_config.get("final_confidence_rule", "multiply")
            ),
            crop_margin=float(hybrid_config.get("crop_margin", 0.10)),
        )

    def describe(self) -> Dict[str, Any]:
        return self.info.to_dict()

    def save_description(self, save_path: str | Path) -> Path:
        save_path = Path(save_path)
    
        IOUtils.save_json(
            dest_path=save_path,
            data=self.describe(),
            indent=2,
        )
    
        return save_path
