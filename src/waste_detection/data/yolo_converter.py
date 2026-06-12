import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm

from waste_detection.utils.io import IOUtils
from waste_detection.data.image_resolver import ImageResolver

logger = logging.getLogger("YoloConverter")


@dataclass
class YoloConversionReport:
    split_name: str
    output_dir: str
    binary: bool
    num_images_total: int = 0
    num_images_copied: int = 0
    num_missing_images: int = 0
    num_label_files: int = 0
    num_boxes_written: int = 0
    num_boxes_skipped: int = 0
    missing_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "output_dir": self.output_dir,
            "binary": self.binary,
            "num_images_total": self.num_images_total,
            "num_images_copied": self.num_images_copied,
            "num_missing_images": self.num_missing_images,
            "num_label_files": self.num_label_files,
            "num_boxes_written": self.num_boxes_written,
            "num_boxes_skipped": self.num_boxes_skipped,
            "missing_files": self.missing_files,
        }


class YoloConverter:
    """
    Convert COCO split sang YOLO format.

    Output chuẩn:

    output_root/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml

    Hỗ trợ:
    - seven-class YOLO dataset
    - binary-waste YOLO dataset cho hybrid detector
    """

    def __init__(
        self,
        output_root: str | Path,
        class_names: List[str],
        binary: bool = False,
        binary_class_name: str = "waste",
        min_box_width: float = 1.0,
        min_box_height: float = 1.0,
    ) -> None:
        self.output_root = Path(output_root)
        self.class_names = class_names
        self.binary = binary
        self.binary_class_name = binary_class_name
        self.min_box_width = min_box_width
        self.min_box_height = min_box_height

    @property
    def yolo_class_names(self) -> List[str]:
        if self.binary:
            return [self.binary_class_name]
        return self.class_names

    def convert_split(
        self,
        dataset: Dict[str, Any],
        split_name: str,
        image_resolver: ImageResolver,
        copy_images: bool = True,
    ) -> YoloConversionReport:
        images_dir = self.output_root / "images" / split_name
        labels_dir = self.output_root / "labels" / split_name

        IOUtils.ensure_dir(images_dir)
        IOUtils.ensure_dir(labels_dir)

        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])
        categories = dataset.get("categories", [])

        category_id_to_yolo_id = self._build_category_id_to_yolo_id(categories)

        annotations_by_image_id = defaultdict(list)

        for annotation in annotations:
            annotations_by_image_id[annotation.get("image_id")].append(annotation)

        report = YoloConversionReport(
            split_name=split_name,
            output_dir=str(self.output_root),
            binary=self.binary,
            num_images_total=len(images),
        )

        for image in tqdm(images, desc=f"Convert YOLO [{split_name}]"):
            file_name = image.get("file_name")
            image_id = image.get("id")
            image_width = image.get("width")
            image_height = image.get("height")

            if not file_name:
                report.num_missing_images += 1
                report.missing_files.append("<missing_file_name>")
                continue

            resolved_image_path = image_resolver.resolve(file_name)

            if resolved_image_path is None:
                report.num_missing_images += 1
                report.missing_files.append(file_name)
                continue

            safe_stem = self._safe_stem(file_name)
            output_image_path = images_dir / f"{safe_stem}{resolved_image_path.suffix}"
            output_label_path = labels_dir / f"{safe_stem}.txt"

            if copy_images:
                IOUtils.copy_file(
                    source_path=resolved_image_path,
                    dest_path=output_image_path,
                    overwrite=True,
                )

            label_lines = []

            for annotation in annotations_by_image_id.get(image_id, []):
                yolo_id = 0 if self.binary else category_id_to_yolo_id.get(
                    annotation.get("category_id")
                )

                if yolo_id is None:
                    report.num_boxes_skipped += 1
                    continue

                bbox = annotation.get("bbox")

                converted = self._coco_bbox_to_yolo(
                    bbox=bbox,
                    image_width=image_width,
                    image_height=image_height,
                )

                if converted is None:
                    report.num_boxes_skipped += 1
                    continue

                x_center, y_center, width, height = converted

                label_lines.append(
                    f"{yolo_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            label_content = "\n".join(label_lines) + "\n" if label_lines else ""

            IOUtils.write_text(
                dest_path=output_label_path,
                content=label_content,
            )

            report.num_images_copied += 1
            report.num_label_files += 1
            report.num_boxes_written += len(label_lines)

        logger.info(
            "Convert YOLO split=%s xong: %d images, %d boxes, %d missing images.",
            split_name,
            report.num_images_copied,
            report.num_boxes_written,
            report.num_missing_images,
        )

        return report

    def write_data_yaml(self) -> Path:
        data_yaml = {
            "path": str(self.output_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.yolo_class_names),
            "names": self.yolo_class_names,
        }

        data_yaml_path = self.output_root / "data.yaml"
        
        IOUtils.ensure_dir(data_yaml_path.parent)

        with data_yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                data_yaml,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

        logger.info("Đã tạo YOLO data.yaml tại: %s", data_yaml_path)

        return data_yaml_path

    def _build_category_id_to_yolo_id(
        self,
        categories: List[Dict[str, Any]],
    ) -> Dict[int, int]:
        if self.binary:
            return {category["id"]: 0 for category in categories}

        class_name_to_yolo_id = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }

        category_id_to_yolo_id = {}

        for category in categories:
            category_id = category.get("id")
            category_name = category.get("name")

            if category_name not in class_name_to_yolo_id:
                logger.warning(
                    "Category name '%s' không nằm trong class_names. Sẽ bỏ qua.",
                    category_name,
                )
                continue

            category_id_to_yolo_id[category_id] = class_name_to_yolo_id[category_name]

        return category_id_to_yolo_id

    def _coco_bbox_to_yolo(
        self,
        bbox: List[float] | None,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float] | None:
        if bbox is None or len(bbox) != 4:
            return None

        x_min, y_min, bbox_width, bbox_height = map(float, bbox)

        if bbox_width < self.min_box_width or bbox_height < self.min_box_height:
            return None

        x1 = max(0.0, x_min)
        y1 = max(0.0, y_min)
        x2 = min(float(image_width), x_min + bbox_width)
        y2 = min(float(image_height), y_min + bbox_height)

        clipped_width = x2 - x1
        clipped_height = y2 - y1

        if clipped_width < self.min_box_width or clipped_height < self.min_box_height:
            return None

        x_center = (x1 + clipped_width / 2.0) / image_width
        y_center = (y1 + clipped_height / 2.0) / image_height
        width = clipped_width / image_width
        height = clipped_height / image_height

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        return x_center, y_center, width, height

    @staticmethod
    def _safe_stem(file_name: str) -> str:
        normalized = str(file_name).replace("\\", "/").strip("/")
        normalized = normalized.replace(":", "_")
        path = Path(normalized)
        stem_without_suffix = str(path.with_suffix("")).replace("\\", "/")
        return stem_without_suffix.replace("/", "__")
