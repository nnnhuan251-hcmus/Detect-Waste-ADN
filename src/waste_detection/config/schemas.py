from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class SystemConfig:
    """
    Cấu hình vận hành hệ thống.
    """

    project_root: Path
    log_dir: Path
    log_file_name: str = "pipeline.log"

    @property
    def log_file_path(self) -> Path:
        return self.log_dir / self.log_file_name


@dataclass(frozen=True)
class DataPathsConfig:
    """
    Cấu hình đường dẫn dữ liệu.

    Các path *_file là optional trong YAML.
    Nếu YAML không khai báo, property sẽ tự suy ra từ coco_7class_dir.
    """

    raw_root: Path
    annotation_file: Path
    image_root: Path

    interim_root: Path
    processed_root: Path

    coco_7class_dir: Path
    yolo_7class_dir: Path
    yolo_binary_waste_dir: Path
    crops_7class_dir: Path

    annotations_7class_file: Path | None = None
    train_annotations_file: Path | None = None
    val_annotations_file: Path | None = None
    test_annotations_file: Path | None = None

    @property
    def annotations_7class_path(self) -> Path:
        return self.annotations_7class_file or (
            self.coco_7class_dir / "annotations_7class.json"
        )

    @property
    def train_annotations_path(self) -> Path:
        return self.train_annotations_file or (
            self.coco_7class_dir / "train_annotations.json"
        )

    @property
    def val_annotations_path(self) -> Path:
        return self.val_annotations_file or (
            self.coco_7class_dir / "val_annotations.json"
        )

    @property
    def test_annotations_path(self) -> Path:
        return self.test_annotations_file or (
            self.coco_7class_dir / "test_annotations.json"
        )

    @property
    def categories_path(self) -> Path:
        return self.coco_7class_dir / "categories.json"

    @property
    def mapping_report_path(self) -> Path:
        return self.coco_7class_dir / "mapping_report.json"

    @property
    def split_report_path(self) -> Path:
        return self.coco_7class_dir / "split_report.json"

    @property
    def yolo_7class_data_yaml_path(self) -> Path:
        return self.yolo_7class_dir / "data.yaml"

    @property
    def yolo_binary_waste_data_yaml_path(self) -> Path:
        return self.yolo_binary_waste_dir / "data.yaml"


@dataclass(frozen=True)
class ImageStructureConfig:
    """
    Cấu hình cấu trúc ảnh TACO hoặc dataset ảnh tương tự.
    """

    batch_dirs: List[str]
    file_name_has_batch_prefix: bool
    allowed_extensions: List[str]

    def __post_init__(self) -> None:
        if not self.allowed_extensions:
            raise ValueError("allowed_extensions không được rỗng.")

        normalized_extensions = []

        for extension in self.allowed_extensions:
            extension = str(extension).strip()

            if not extension:
                continue

            if not extension.startswith("."):
                extension = f".{extension}"

            normalized_extensions.append(extension)

        if not normalized_extensions:
            raise ValueError("allowed_extensions không có extension hợp lệ.")

        object.__setattr__(self, "allowed_extensions", normalized_extensions)


@dataclass(frozen=True)
class ClassConfig:
    """
    Cấu hình class đích.
    """

    num_classes: int
    names: List[str]
    name_to_id: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes phải > 0.")

        if self.num_classes != len(self.names):
            raise ValueError(
                f"num_classes={self.num_classes} nhưng len(names)={len(self.names)}"
            )

        if len(set(self.names)) != len(self.names):
            raise ValueError(f"Danh sách class bị trùng tên: {self.names}")

        if not self.name_to_id:
            generated = {name: index for index, name in enumerate(self.names)}
            object.__setattr__(self, "name_to_id", generated)

        if set(self.name_to_id.keys()) != set(self.names):
            raise ValueError(
                "name_to_id không khớp với danh sách class names. "
                f"name_to_id keys={list(self.name_to_id.keys())}, names={self.names}"
            )

        ids = list(self.name_to_id.values())

        if len(set(ids)) != len(ids):
            raise ValueError(f"name_to_id có id bị trùng: {self.name_to_id}")

        expected_ids = set(range(self.num_classes))

        if set(ids) != expected_ids:
            raise ValueError(
                "Class ids phải liên tục từ 0 đến num_classes - 1. "
                f"Expected={expected_ids}, got={set(ids)}"
            )

    @property
    def id_to_name(self) -> Dict[int, str]:
        return {class_id: name for name, class_id in self.name_to_id.items()}


@dataclass(frozen=True)
class MappingConfig:
    """
    Cấu hình mapping label.
    """

    file: Path
    ignore_key: str = "ignore"
    fail_on_unmapped: bool = True


@dataclass(frozen=True)
class SplitConfig:
    """
    Cấu hình chia train/val/test.
    """

    strategy: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in {"image_level", "multilabel_stratified", "random"}:
            raise ValueError(
                "split.strategy không hợp lệ. "
                "Chỉ hỗ trợ: image_level, multilabel_stratified, random."
            )

        for name, value in {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }.items():
            if value < 0 or value > 1:
                raise ValueError(f"{name} phải nằm trong [0, 1]. Hiện tại={value}")

        total = self.train_ratio + self.val_ratio + self.test_ratio

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Tổng train_ratio + val_ratio + test_ratio phải bằng 1. "
                f"Hiện tại = {total}"
            )

        if self.train_ratio <= 0:
            raise ValueError("train_ratio phải > 0.")

        if self.val_ratio <= 0:
            raise ValueError("val_ratio phải > 0.")

        if self.test_ratio <= 0:
            raise ValueError("test_ratio phải > 0.")


@dataclass(frozen=True)
class BBoxConfig:
    """
    Cấu hình bbox.
    """

    source_format: str
    min_width: float
    min_height: float
    clamp_to_image: bool
    drop_invalid_bbox: bool

    def __post_init__(self) -> None:
        if self.source_format not in {"coco_xywh"}:
            raise ValueError(
                f"source_format={self.source_format} chưa được hỗ trợ. "
                "Hiện chỉ hỗ trợ coco_xywh."
            )

        if self.min_width <= 0:
            raise ValueError("bbox.min_width phải > 0.")

        if self.min_height <= 0:
            raise ValueError("bbox.min_height phải > 0.")


@dataclass(frozen=True)
class CropDatasetConfig:
    """
    Cấu hình tạo crop dataset cho EfficientNet-B0.
    """

    output_size: int
    save_format: str
    jpeg_quality: int
    include_context_margin: float
    min_crop_size: int
    write_metadata_csv: bool

    def __post_init__(self) -> None:
        if self.output_size <= 0:
            raise ValueError("crop_dataset.output_size phải > 0.")

        normalized_format = self.save_format.lower().lstrip(".")

        if normalized_format not in {"jpg", "jpeg", "png"}:
            raise ValueError(
                "crop_dataset.save_format chỉ hỗ trợ: jpg, jpeg, png. "
                f"Hiện tại={self.save_format}"
            )

        object.__setattr__(self, "save_format", normalized_format)

        if self.jpeg_quality < 1 or self.jpeg_quality > 100:
            raise ValueError("crop_dataset.jpeg_quality phải nằm trong [1, 100].")

        if self.include_context_margin < 0:
            raise ValueError("crop_dataset.include_context_margin không được âm.")

        if self.min_crop_size <= 0:
            raise ValueError("crop_dataset.min_crop_size phải > 0.")


@dataclass(frozen=True)
class DataConfig:
    """
    Config tổng hợp cho data pipeline.
    """

    dataset: Dict[str, Any]
    paths: DataPathsConfig
    image_structure: ImageStructureConfig
    classes: ClassConfig
    mapping: MappingConfig
    split: SplitConfig
    bbox: BBoxConfig
    crop_dataset: CropDatasetConfig

    @property
    def dataset_name(self) -> str:
        return str(self.dataset.get("name", "unknown_dataset"))

    @property
    def annotation_format(self) -> str:
        return str(self.dataset.get("annotation_format", "unknown"))


@dataclass(frozen=True)
class LoadedConfig:
    """
    Object tổng hợp sau khi load config.

    Dùng khi script cần:
    - system config
    - data config
    - model config
    - experiment config
    """

    system: SystemConfig
    data: DataConfig | None = None
    model: Dict[str, Any] | None = None
    experiment: Dict[str, Any] | None = None
