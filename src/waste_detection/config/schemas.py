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

    @property
    def annotations_7class_path(self) -> Path:
        return self.coco_7class_dir / "annotations_7class.json"

    @property
    def train_annotations_path(self) -> Path:
        return self.coco_7class_dir / "train_annotations.json"

    @property
    def val_annotations_path(self) -> Path:
        return self.coco_7class_dir / "val_annotations.json"

    @property
    def test_annotations_path(self) -> Path:
        return self.coco_7class_dir / "test_annotations.json"

    @property
    def categories_path(self) -> Path:
        return self.coco_7class_dir / "categories.json"

    @property
    def mapping_report_path(self) -> Path:
        return self.coco_7class_dir / "mapping_report.json"


@dataclass(frozen=True)
class ImageStructureConfig:
    """
    Cấu hình cấu trúc ảnh TACO.
    """
    batch_dirs: List[str]
    file_name_has_batch_prefix: bool
    allowed_extensions: List[str]


@dataclass(frozen=True)
class ClassConfig:
    """
    Cấu hình class đích.
    """
    num_classes: int
    names: List[str]
    name_to_id: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_classes != len(self.names):
            raise ValueError(
                f"num_classes={self.num_classes} nhưng len(names)={len(self.names)}"
            )

        if not self.name_to_id:
            generated = {name: index for index, name in enumerate(self.names)}
            object.__setattr__(self, "name_to_id", generated)

        if set(self.name_to_id.keys()) != set(self.names):
            raise ValueError("name_to_id không khớp với danh sách class names.")


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
        total = self.train_ratio + self.val_ratio + self.test_ratio

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Tổng train_ratio + val_ratio + test_ratio phải bằng 1. "
                f"Hiện tại = {total}"
            )


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
