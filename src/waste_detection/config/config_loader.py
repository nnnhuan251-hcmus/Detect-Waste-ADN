import os
from pathlib import Path
from typing import Any, Dict

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
    Có thể load riêng data/model/experiment hoặc load tất cả.
    Sau này mọi script đều dùng chung loader này.
    
    Nguyên tắc:
    - PROJECT_ROOT env var được ưu tiên nếu có.
    - Nếu không có, dùng current working directory.
    - Mọi relative path trong YAML được resolve theo project_root.
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = self._resolve_project_root(project_root)

    def load_yaml(self, config_path: str | Path) -> Dict[str, Any]:
        config_path = self._resolve_path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy config file: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if data is None:
            raise ValueError(f"Config file rỗng: {config_path}")

        if not isinstance(data, dict):
            raise ValueError(f"Config file phải có dạng YAML mapping: {config_path}")

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

        self._validate_required_sections(raw_config, required_sections, data_config_path)

        paths = raw_config["paths"]
        image_structure = raw_config["image_structure"]
        classes = raw_config["classes"]
        mapping = raw_config["mapping"]
        split = raw_config["split"]
        bbox = raw_config["bbox"]
        crop_dataset = raw_config["crop_dataset"]

        data_paths = DataPathsConfig(
            raw_root=self._resolve_path(paths["raw_root"]),
            annotation_file=self._resolve_path(paths["annotation_file"]),
            image_root=self._resolve_path(paths["image_root"]),
            interim_root=self._resolve_path(paths["interim_root"]),
            processed_root=self._resolve_path(paths["processed_root"]),
            coco_7class_dir=self._resolve_path(paths["coco_7class_dir"]),
            yolo_7class_dir=self._resolve_path(paths["yolo_7class_dir"]),
            yolo_binary_waste_dir=self._resolve_path(paths["yolo_binary_waste_dir"]),
            crops_7class_dir=self._resolve_path(paths["crops_7class_dir"]),
        )

        image_structure_config = ImageStructureConfig(
            batch_dirs=list(image_structure["batch_dirs"]),
            file_name_has_batch_prefix=bool(
                image_structure["file_name_has_batch_prefix"]
            ),
            allowed_extensions=list(image_structure["allowed_extensions"]),
        )

        class_config = ClassConfig(
            num_classes=int(classes["num_classes"]),
            names=list(classes["names"]),
            name_to_id=dict(classes.get("name_to_id", {})),
        )

        mapping_config = MappingConfig(
            file=self._resolve_path(mapping["file"]),
            ignore_key=str(mapping.get("ignore_key", "ignore")),
            fail_on_unmapped=bool(mapping.get("fail_on_unmapped", True)),
        )

        split_config = SplitConfig(
            strategy=str(split["strategy"]),
            train_ratio=float(split["train_ratio"]),
            val_ratio=float(split["val_ratio"]),
            test_ratio=float(split["test_ratio"]),
            seed=int(split["seed"]),
            shuffle=bool(split.get("shuffle", True)),
        )

        bbox_config = BBoxConfig(
            source_format=str(bbox["source_format"]),
            min_width=float(bbox["min_width"]),
            min_height=float(bbox["min_height"]),
            clamp_to_image=bool(bbox["clamp_to_image"]),
            drop_invalid_bbox=bool(bbox["drop_invalid_bbox"]),
        )

        crop_dataset_config = CropDatasetConfig(
            output_size=int(crop_dataset["output_size"]),
            save_format=str(crop_dataset["save_format"]),
            jpeg_quality=int(crop_dataset["jpeg_quality"]),
            include_context_margin=float(crop_dataset["include_context_margin"]),
            min_crop_size=int(crop_dataset["min_crop_size"]),
            write_metadata_csv=bool(crop_dataset["write_metadata_csv"]),
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
        return self.load_yaml(model_config_path)

    def load_experiment_config(
        self,
        experiment_config_path: str | Path,
    ) -> Dict[str, Any]:
        return self.load_yaml(experiment_config_path)

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

    @staticmethod
    def _validate_required_sections(
        config: Dict[str, Any],
        required_sections: list[str],
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
