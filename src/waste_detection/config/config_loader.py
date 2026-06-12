from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from waste_detection.config.schemas import (
    BBoxConfig,
    ClassConfig,
    CropDatasetConfig,
    DataConfig,
    DataPathsConfig,
    ImageStructureConfig,
    LoadedConfig,
    MappingConfig,
    SplitConfig,
    SystemConfig,
)


class ConfigLoader:
    """
    Loader tập trung cho YAML config.

    Nguyên tắc:
    - PROJECT_ROOT env var được ưu tiên nếu project_root không được truyền.
    - Nếu không có PROJECT_ROOT, dùng current working directory.
    - Mọi relative path trong YAML được resolve theo project_root.
    - Loader chỉ load/validate config, không tạo data folder và không chạy pipeline.
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = self._resolve_project_root(project_root)

    def load_yaml(self, config_path: str | Path) -> Dict[str, Any]:
        resolved_config_path = self._resolve_path(config_path)

        if not resolved_config_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy config file: {resolved_config_path}"
            )

        with resolved_config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if data is None:
            raise ValueError(f"Config file rỗng: {resolved_config_path}")

        if not isinstance(data, dict):
            raise ValueError(
                f"Config file phải có dạng YAML mapping: {resolved_config_path}"
            )

        return data

    def load_system_config(
        self,
        log_dir: str | Path = "outputs/logs",
        log_file_name: str = "pipeline.log",
    ) -> SystemConfig:
        return SystemConfig(
            project_root=self.project_root,
            log_dir=self._resolve_path(log_dir),
            log_file_name=log_file_name,
        )

    def load_data_config(self, data_config_path: str | Path) -> DataConfig:
        raw_config = self.load_yaml(data_config_path)

        required_sections = [
            "dataset",
            "paths",
            "image_structure",
            "classes",
            "mapping",
            "split",
            "bbox",
            "crop_dataset",
        ]

        self._validate_required_sections(
            config=raw_config,
            required_sections=required_sections,
            config_path=data_config_path,
        )

        paths = self._require_mapping(raw_config, "paths", data_config_path)
        image_structure = self._require_mapping(
            raw_config,
            "image_structure",
            data_config_path,
        )
        classes = self._require_mapping(raw_config, "classes", data_config_path)
        mapping = self._require_mapping(raw_config, "mapping", data_config_path)
        split = self._require_mapping(raw_config, "split", data_config_path)
        bbox = self._require_mapping(raw_config, "bbox", data_config_path)
        crop_dataset = self._require_mapping(
            raw_config,
            "crop_dataset",
            data_config_path,
        )

        data_paths = DataPathsConfig(
            raw_root=self._resolve_required_path(paths, "raw_root", data_config_path),
            annotation_file=self._resolve_required_path(
                paths,
                "annotation_file",
                data_config_path,
            ),
            image_root=self._resolve_required_path(paths, "image_root", data_config_path),
            interim_root=self._resolve_required_path(
                paths,
                "interim_root",
                data_config_path,
            ),
            processed_root=self._resolve_required_path(
                paths,
                "processed_root",
                data_config_path,
            ),
            coco_7class_dir=self._resolve_required_path(
                paths,
                "coco_7class_dir",
                data_config_path,
            ),
            yolo_7class_dir=self._resolve_required_path(
                paths,
                "yolo_7class_dir",
                data_config_path,
            ),
            yolo_binary_waste_dir=self._resolve_required_path(
                paths,
                "yolo_binary_waste_dir",
                data_config_path,
            ),
            crops_7class_dir=self._resolve_required_path(
                paths,
                "crops_7class_dir",
                data_config_path,
            ),
            annotations_7class_file=self._resolve_optional_path(
                paths,
                "annotations_7class_file",
            ),
            train_annotations_file=self._resolve_optional_path(
                paths,
                "train_annotations_file",
            ),
            val_annotations_file=self._resolve_optional_path(
                paths,
                "val_annotations_file",
            ),
            test_annotations_file=self._resolve_optional_path(
                paths,
                "test_annotations_file",
            ),
        )

        image_structure_config = ImageStructureConfig(
            batch_dirs=[
                str(item) for item in image_structure.get("batch_dirs", [])
            ],
            file_name_has_batch_prefix=bool(
                image_structure.get("file_name_has_batch_prefix", False)
            ),
            allowed_extensions=[
                str(item) for item in image_structure.get("allowed_extensions", [])
            ],
        )

        class_config = ClassConfig(
            num_classes=int(
                self._require_key(classes, "num_classes", data_config_path)
            ),
            names=[
                str(item)
                for item in self._require_key(classes, "names", data_config_path)
            ],
            name_to_id={
                str(name): int(class_id)
                for name, class_id in dict(classes.get("name_to_id", {})).items()
            },
        )

        mapping_config = MappingConfig(
            file=self._resolve_required_path(mapping, "file", data_config_path),
            ignore_key=str(mapping.get("ignore_key", "ignore")),
            fail_on_unmapped=bool(mapping.get("fail_on_unmapped", True)),
        )

        split_config = SplitConfig(
            strategy=str(self._require_key(split, "strategy", data_config_path)),
            train_ratio=float(
                self._require_key(split, "train_ratio", data_config_path)
            ),
            val_ratio=float(self._require_key(split, "val_ratio", data_config_path)),
            test_ratio=float(self._require_key(split, "test_ratio", data_config_path)),
            seed=int(self._require_key(split, "seed", data_config_path)),
            shuffle=bool(split.get("shuffle", True)),
        )

        bbox_config = BBoxConfig(
            source_format=str(
                self._require_key(bbox, "source_format", data_config_path)
            ),
            min_width=float(self._require_key(bbox, "min_width", data_config_path)),
            min_height=float(self._require_key(bbox, "min_height", data_config_path)),
            clamp_to_image=bool(
                self._require_key(bbox, "clamp_to_image", data_config_path)
            ),
            drop_invalid_bbox=bool(
                self._require_key(bbox, "drop_invalid_bbox", data_config_path)
            ),
        )

        crop_dataset_config = CropDatasetConfig(
            output_size=int(
                self._require_key(crop_dataset, "output_size", data_config_path)
            ),
            save_format=str(
                self._require_key(crop_dataset, "save_format", data_config_path)
            ),
            jpeg_quality=int(
                self._require_key(crop_dataset, "jpeg_quality", data_config_path)
            ),
            include_context_margin=float(
                self._require_key(
                    crop_dataset,
                    "include_context_margin",
                    data_config_path,
                )
            ),
            min_crop_size=int(
                self._require_key(crop_dataset, "min_crop_size", data_config_path)
            ),
            write_metadata_csv=bool(
                self._require_key(
                    crop_dataset,
                    "write_metadata_csv",
                    data_config_path,
                )
            ),
        )

        return DataConfig(
            dataset=dict(raw_config["dataset"]),
            paths=data_paths,
            image_structure=image_structure_config,
            classes=class_config,
            mapping=mapping_config,
            split=split_config,
            bbox=bbox_config,
            crop_dataset=crop_dataset_config,
        )

    def load_model_config(self, model_config_path: str | Path) -> Dict[str, Any]:
        config = self.load_yaml(model_config_path)
        self._validate_model_config(config, model_config_path)
        return config

    def load_experiment_config(
        self,
        experiment_config_path: str | Path,
    ) -> Dict[str, Any]:
        config = self.load_yaml(experiment_config_path)
        self._validate_experiment_config(config, experiment_config_path)
        return config

    def load_all(
        self,
        data_config_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        experiment_config_path: str | Path | None = None,
        log_dir: str | Path = "outputs/logs",
        log_file_name: str = "pipeline.log",
    ) -> LoadedConfig:
        system_config = self.load_system_config(
            log_dir=log_dir,
            log_file_name=log_file_name,
        )

        data_config = (
            self.load_data_config(data_config_path)
            if data_config_path is not None
            else None
        )

        model_config = (
            self.load_model_config(model_config_path)
            if model_config_path is not None
            else None
        )

        experiment_config = (
            self.load_experiment_config(experiment_config_path)
            if experiment_config_path is not None
            else None
        )

        return LoadedConfig(
            system=system_config,
            data=data_config,
            model=model_config,
            experiment=experiment_config,
        )

    def _resolve_project_root(self, project_root: str | Path | None) -> Path:
        if project_root is not None:
            return Path(project_root).expanduser().resolve()

        env_project_root = os.getenv("PROJECT_ROOT")

        if env_project_root:
            return Path(env_project_root).expanduser().resolve()

        return Path.cwd().resolve()

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path).expanduser()

        if path.is_absolute():
            return path.resolve()

        return (self.project_root / path).resolve()

    def _resolve_optional_path(
        self,
        mapping: Mapping[str, Any],
        key: str,
    ) -> Path | None:
        value = mapping.get(key)

        if value is None or str(value).strip() == "":
            return None

        return self._resolve_path(value)

    def _resolve_required_path(
        self,
        mapping: Mapping[str, Any],
        key: str,
        config_path: str | Path,
    ) -> Path:
        value = self._require_key(mapping, key, config_path)
        return self._resolve_path(value)

    @staticmethod
    def _validate_required_sections(
        config: Dict[str, Any],
        required_sections: List[str],
        config_path: str | Path,
    ) -> None:
        missing_sections = [
            section for section in required_sections if section not in config
        ]

        if missing_sections:
            raise KeyError(
                f"Config file {config_path} thiếu các section bắt buộc: "
                f"{missing_sections}"
            )

    @staticmethod
    def _require_mapping(
        config: Mapping[str, Any],
        key: str,
        config_path: str | Path,
    ) -> Mapping[str, Any]:
        value = ConfigLoader._require_key(config, key, config_path)

        if not isinstance(value, Mapping):
            raise TypeError(
                f"Section '{key}' trong {config_path} phải là YAML mapping."
            )

        return value

    @staticmethod
    def _require_key(
        config: Mapping[str, Any],
        key: str,
        config_path: str | Path,
    ) -> Any:
        if key not in config:
            raise KeyError(f"Config file {config_path} thiếu key bắt buộc: {key}")

        value = config[key]

        if value is None:
            raise ValueError(
                f"Config file {config_path} có key '{key}' nhưng giá trị là null."
            )

        return value

    @staticmethod
    def _validate_model_config(
        config: Dict[str, Any],
        config_path: str | Path,
    ) -> None:
        ConfigLoader._validate_required_sections(
            config=config,
            required_sections=["model"],
            config_path=config_path,
        )

        model = ConfigLoader._require_mapping(config, "model", config_path)

        model_name = str(ConfigLoader._require_key(model, "name", config_path))
        model_type = str(ConfigLoader._require_key(model, "type", config_path))

        if not model_name.strip():
            raise ValueError(f"model.name trong {config_path} không được rỗng.")

        if model_type not in {"hybrid", "detector_only"}:
            raise ValueError(
                f"model.type trong {config_path} không hợp lệ: {model_type}. "
                "Chỉ hỗ trợ: hybrid, detector_only."
            )

        if "detector" in config:
            detector = ConfigLoader._require_mapping(config, "detector", config_path)
            ConfigLoader._require_key(detector, "name", config_path)
            ConfigLoader._require_key(detector, "weights", config_path)
            ConfigLoader._require_key(detector, "detector_mode", config_path)

            detector_mode = str(detector["detector_mode"])

            if detector_mode not in {"binary_waste", "seven_class"}:
                raise ValueError(
                    f"detector.detector_mode không hợp lệ: {detector_mode}. "
                    "Chỉ hỗ trợ: binary_waste, seven_class."
                )

        if model_type == "hybrid":
            ConfigLoader._validate_required_sections(
                config=config,
                required_sections=["detector", "classifier", "hybrid_inference"],
                config_path=config_path,
            )

        if model_type == "detector_only":
            ConfigLoader._validate_required_sections(
                config=config,
                required_sections=["detector"],
                config_path=config_path,
            )

    @staticmethod
    def _validate_experiment_config(
        config: Dict[str, Any],
        config_path: str | Path,
    ) -> None:
        required_sections = [
            "experiment",
            "data",
            "augmentation",
            "loss",
            "optimizer",
            "scheduler",
            "fine_tuning",
            "training",
            "checkpoint",
            "tracking",
        ]

        ConfigLoader._validate_required_sections(
            config=config,
            required_sections=required_sections,
            config_path=config_path,
        )

        experiment = ConfigLoader._require_mapping(config, "experiment", config_path)
        training = ConfigLoader._require_mapping(config, "training", config_path)
        optimizer = ConfigLoader._require_mapping(config, "optimizer", config_path)

        ConfigLoader._require_key(experiment, "name", config_path)

        epochs = int(ConfigLoader._require_key(training, "epochs", config_path))
        batch_size = int(ConfigLoader._require_key(training, "batch_size", config_path))
        patience = int(training.get("patience", 0))

        if epochs <= 0:
            raise ValueError(f"training.epochs trong {config_path} phải > 0.")

        if batch_size <= 0:
            raise ValueError(f"training.batch_size trong {config_path} phải > 0.")

        if patience < 0:
            raise ValueError(f"training.patience trong {config_path} không được âm.")

        lr = float(ConfigLoader._require_key(optimizer, "lr", config_path))

        if lr <= 0:
            raise ValueError(f"optimizer.lr trong {config_path} phải > 0.")

        if "sampler" in config:
            sampler = ConfigLoader._require_mapping(config, "sampler", config_path)

            sampler_name = str(sampler.get("name", "weighted_random"))

            if sampler_name not in {"weighted_random"}:
                raise ValueError(
                    f"sampler.name trong {config_path} không hợp lệ: {sampler_name}. "
                    "Hiện chỉ hỗ trợ: weighted_random."
                )

            num_samples_multiplier = float(
                sampler.get("num_samples_multiplier", 1.0)
            )

            if num_samples_multiplier <= 0:
                raise ValueError(
                    "sampler.num_samples_multiplier phải > 0. "
                    f"Hiện tại={num_samples_multiplier}"
                )
