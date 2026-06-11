from waste_detection.training.trainer_base import TrainerBase

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from waste_detection.evaluation.detection_metrics import DetectionMetrics
from waste_detection.models.detector_base import DetectorBase
from waste_detection.models.rtdetr_detector import RTDETRDetector
from waste_detection.models.yolov8_detector import YOLOv8Detector
from waste_detection.utils.io import IOUtils


logger = logging.getLogger("DetectorTrainer")


class DetectorTrainer(TrainerBase):
    """
    Trainer wrapper cho detector dùng Ultralytics.

    Hỗ trợ:
    - YOLOv8n binary detector cho hybrid.
    - YOLOv8s detector-only 7 class.
    - RT-DETR-L detector-only 7 class.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        experiment_config: Dict[str, Any],
        data_yaml_path: str | Path,
        output_root: str | Path,
    ) -> None:
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.data_yaml_path = Path(data_yaml_path)
        self.output_root = Path(output_root)

        self.detector_config = model_config.get("detector", {})
        self.model_meta = model_config.get("model", {})
        self.experiment_meta = experiment_config.get("experiment", {})

        self.model_name = self.model_meta.get(
            "name",
            self.detector_config.get("name", "detector"),
        )

        self.run_name = self.experiment_meta.get("name", "run_unknown")

        self.output_dir = self.output_root / self.model_name / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.detector = self._build_detector()

    def train(self) -> Dict[str, Any]:
        train_args = self._build_train_args()

        logger.info("Training detector: model=%s, run=%s", self.model_name, self.run_name)
        logger.info("Ultralytics train args: %s", train_args)

        train_result = self.detector.train(**train_args)

        best_path = self.output_dir / "weights" / "best.pt"
        last_path = self.output_dir / "weights" / "last.pt"

        summary = {
            "model_name": self.model_name,
            "run_name": self.run_name,
            "data_yaml_path": str(self.data_yaml_path),
            "output_dir": str(self.output_dir),
            "best_weight_path": str(best_path),
            "last_weight_path": str(last_path),
            "train_args": train_args,
            "train_result_type": str(type(train_result)),
        }

        IOUtils.save_json(self.output_dir / "train_summary.json", summary)

        logger.info("Training detector hoàn tất.")
        logger.info("Best weight dự kiến: %s", best_path)

        return summary

    def validate(
        self,
        weights_path: str | Path | None = None,
        split: str = "test",
    ) -> Dict[str, Any]:
        if weights_path is not None:
            self.detector = self._build_detector(weights_override=weights_path)

        val_args = self._build_val_args(split=split)

        logger.info("Validating detector với args: %s", val_args)

        metrics = self.detector.val(**val_args)
        metrics_dict = DetectionMetrics.from_ultralytics(metrics)

        output_path = self.output_dir / f"{split}_metrics.json"

        IOUtils.save_json(output_path, metrics_dict)

        logger.info("Validate detector hoàn tất. Metrics: %s", metrics_dict)
        logger.info("Metrics saved to: %s", output_path)

        return metrics_dict

    def _build_detector(
        self,
        weights_override: str | Path | None = None,
    ) -> DetectorBase:
        weights = weights_override or self.detector_config.get("weights")

        detector_name = str(self.detector_config.get("name", "")).lower()
        family = str(self.detector_config.get("family", "")).lower()

        if "rtdetr" in detector_name or "rt-detr" in family:
            return RTDETRDetector(weights=weights)

        return YOLOv8Detector(weights=weights)

    def _build_train_args(self) -> Dict[str, Any]:
        training_config = self.experiment_config.get("training", {})
        optimizer_config = self.experiment_config.get("optimizer", {})
        scheduler_config = self.experiment_config.get("scheduler", {})
        augmentation_config = self.experiment_config.get("augmentation", {})
        fine_tuning_config = self.experiment_config.get("fine_tuning", {})

        epochs = int(training_config.get("epochs", 50))
        batch_size = int(training_config.get("batch_size", 16))
        num_workers = int(training_config.get("num_workers", 2))
        patience = int(training_config.get("patience", 10))
        mixed_precision = bool(training_config.get("mixed_precision", True))

        image_size = int(self.detector_config.get("input_size", 640))

        optimizer_name = self._normalize_optimizer_name(
            optimizer_config.get("name", "auto")
        )

        lr0 = float(optimizer_config.get("lr", 0.001))

        scheduler_name = scheduler_config.get("name", "constant")
        cos_lr = bool(scheduler_config.get("enabled", False)) and (
            scheduler_name == "cosine_annealing"
        )

        warmup_config = scheduler_config.get("warmup", {})
        warmup_epochs = int(warmup_config.get("warmup_epochs", 0))

        train_args = {
            "data": str(self.data_yaml_path),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": image_size,
            "project": str(self.output_root / self.model_name),
            "name": self.run_name,
            "exist_ok": True,
            "plots": True,
            "save": True,
            "save_period": -1,
            "patience": patience,
            "optimizer": optimizer_name,
            "lr0": lr0,
            "cos_lr": cos_lr,
            "warmup_epochs": warmup_epochs,
            "workers": num_workers,
            "amp": mixed_precision,
            "seed": int(self.experiment_meta.get("seed", 42)),
        }

        train_args.update(self._build_augmentation_args(augmentation_config))

        if bool(fine_tuning_config.get("freeze_backbone", False)):
            # Ultralytics freeze nhận số layer đầu cần freeze.
            # Giá trị này chỉ là mặc định an toàn ban đầu, có thể tinh chỉnh sau.
            train_args["freeze"] = int(fine_tuning_config.get("freeze_layers", 10))

        return train_args

    def _build_val_args(self, split: str = "test") -> Dict[str, Any]:
        image_size = int(self.detector_config.get("input_size", 640))

        return {
            "data": str(self.data_yaml_path),
            "imgsz": image_size,
            "split": split,
            "project": str(self.output_root / self.model_name),
            "name": f"{self.run_name}_{split}_eval",
            "exist_ok": True,
            "plots": True,
        }

    def _build_augmentation_args(
        self,
        augmentation_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        mosaic_enabled = bool(augmentation_config.get("mosaic", False))
        color_jitter_enabled = bool(augmentation_config.get("color_jitter", False))
        horizontal_flip_enabled = bool(augmentation_config.get("horizontal_flip", False))
    
        color_params = augmentation_config.get("color_jitter_params", {})
    
        if not any(
            [
                mosaic_enabled,
                color_jitter_enabled,
                horizontal_flip_enabled,
            ]
        ):
            return {
                "mosaic": 0.0,
                "degrees": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
            }
    
        return {
            "mosaic": float(augmentation_config.get("mosaic_prob", 0.5))
            if mosaic_enabled
            else 0.0,
            # Không dùng diagonal rotation cho detector vì bbox là axis-aligned.
            # Nếu cần 90/180 thật, làm offline bbox-aware augmentation riêng.
            "degrees": 0.0,
            "fliplr": 0.5 if horizontal_flip_enabled else 0.0,
            "flipud": 0.0,
            "hsv_h": float(color_params.get("hue", 0.015))
            if color_jitter_enabled
            else 0.0,
            "hsv_s": float(color_params.get("saturation", 0.2))
            if color_jitter_enabled
            else 0.0,
            "hsv_v": float(color_params.get("brightness", 0.2))
            if color_jitter_enabled
            else 0.0,
        }

    @staticmethod
    def _normalize_optimizer_name(name: str) -> str:
        name_lower = str(name).lower()

        mapping = {
            "adamw": "AdamW",
            "adam": "Adam",
            "sgd": "SGD",
            "auto": "auto",
        }

        return mapping.get(name_lower, name)
