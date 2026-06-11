from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import cv2
from tqdm import tqdm

from waste_detection.data.image_resolver import ImageResolver


logger = logging.getLogger("CropDatasetBuilder")


@dataclass
class CropBuildReport:
    split_name: str
    output_dir: str
    num_images_total: int = 0
    num_images_read: int = 0
    num_missing_images: int = 0
    num_annotations_total: int = 0
    num_crops_saved: int = 0
    num_invalid_bboxes: int = 0
    num_unknown_categories: int = 0
    missing_files: List[str] = field(default_factory=list)
    metadata_path: str | None = None

    @property
    def num_saved_crops(self) -> int:
        """
        Alias để tương thích với test hoặc code dùng tên num_saved_crops.
        """
        return self.num_crops_saved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "output_dir": self.output_dir,
            "num_images_total": self.num_images_total,
            "num_images_read": self.num_images_read,
            "num_missing_images": self.num_missing_images,
            "num_annotations_total": self.num_annotations_total,
            "num_crops_saved": self.num_crops_saved,
            "num_saved_crops": self.num_saved_crops,
            "num_invalid_bboxes": self.num_invalid_bboxes,
            "num_unknown_categories": self.num_unknown_categories,
            "missing_files": self.missing_files,
            "metadata_path": self.metadata_path,
        }


class CropDatasetBuilder:
    """
    Tạo crop dataset cho EfficientNet-B0 từ COCO annotations.

    Lưu ý:
    - Crop từ GT bbox dùng để train/evaluate classifier riêng.
    - Không dùng GT crop để báo cáo final end-to-end metric.
    - Final hybrid evaluation phải chạy từ ảnh gốc qua detector.
    """

    def __init__(
        self,
        output_root: str | Path,
        class_names: List[str] | None = None,
        margin_ratio: float = 0.10,
        min_crop_size: int = 8,
        save_format: str = "jpg",
        jpeg_quality: int = 95,
        write_metadata_csv: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.class_names = list(class_names) if class_names is not None else None
        self.margin_ratio = float(margin_ratio)
        self.min_crop_size = int(min_crop_size)
        self.save_format = save_format.lower().lstrip(".")
        self.jpeg_quality = int(jpeg_quality)
        self.write_metadata_csv = bool(write_metadata_csv)

    def build_from_coco(
        self,
        dataset: Dict[str, Any],
        image_root: str | Path,
        split_name: str,
        image_resolver: ImageResolver | None = None,
    ) -> CropBuildReport:
        """
        Interface tiện dụng cho test và script đơn giản.

        Ví dụ:
            builder = CropDatasetBuilder(...)
            report = builder.build_from_coco(...)
        """
        report, metadata_rows = self._build(
            dataset=dataset,
            split_name=split_name,
            image_root=Path(image_root),
            image_resolver=image_resolver,
            auto_write_metadata=self.write_metadata_csv,
        )

        return report

    def build_split(
        self,
        dataset: Dict[str, Any],
        split_name: str,
        image_resolver: ImageResolver,
    ) -> tuple[CropBuildReport, List[Dict[str, Any]]]:
        """
        Interface tương thích với pipeline đang dùng ImageResolver.

        Trả về:
            report, metadata_rows
        """
        return self._build(
            dataset=dataset,
            split_name=split_name,
            image_root=None,
            image_resolver=image_resolver,
            auto_write_metadata=False,
        )

    def _build(
        self,
        dataset: Dict[str, Any],
        split_name: str,
        image_root: Path | None,
        image_resolver: ImageResolver | None,
        auto_write_metadata: bool,
    ) -> tuple[CropBuildReport, List[Dict[str, Any]]]:
        split_output_dir = self.output_root / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])
        categories = dataset.get("categories", [])

        category_id_to_name = {
            int(category["id"]): str(category["name"])
            for category in categories
        }

        output_class_names = self.class_names or sorted(set(category_id_to_name.values()))

        for class_name in output_class_names:
            (split_output_dir / class_name).mkdir(parents=True, exist_ok=True)

        annotations_by_image_id: Dict[int, List[Dict[str, Any]]] = {}

        for annotation in annotations:
            image_id = annotation.get("image_id")
            annotations_by_image_id.setdefault(image_id, []).append(annotation)

        report = CropBuildReport(
            split_name=split_name,
            output_dir=str(split_output_dir),
            num_images_total=len(images),
            num_annotations_total=len(annotations),
        )

        metadata_rows: List[Dict[str, Any]] = []

        for image in tqdm(images, desc=f"Create crops [{split_name}]"):
            image_id = image.get("id")
            file_name = image.get("file_name")

            image_annotations = annotations_by_image_id.get(image_id, [])

            if not image_annotations:
                continue

            if not file_name:
                report.num_missing_images += 1
                report.missing_files.append("<missing_file_name>")
                continue

            resolved_image_path = self._resolve_image_path(
                file_name=file_name,
                image_root=image_root,
                image_resolver=image_resolver,
            )

            if resolved_image_path is None:
                report.num_missing_images += 1
                report.missing_files.append(str(file_name))
                continue

            image_array = cv2.imread(str(resolved_image_path))

            if image_array is None:
                report.num_missing_images += 1
                report.missing_files.append(str(file_name))
                continue

            report.num_images_read += 1

            image_height, image_width = image_array.shape[:2]
            safe_stem = self._safe_stem(file_name)

            for annotation_index, annotation in enumerate(image_annotations):
                category_id = int(annotation.get("category_id"))
                class_name = category_id_to_name.get(category_id)

                if class_name is None:
                    report.num_unknown_categories += 1
                    continue

                if self.class_names is not None and class_name not in self.class_names:
                    report.num_unknown_categories += 1
                    continue

                crop_bbox = self._compute_crop_bbox(
                    bbox=annotation.get("bbox"),
                    image_width=image_width,
                    image_height=image_height,
                )

                if crop_bbox is None:
                    report.num_invalid_bboxes += 1
                    continue

                x1, y1, x2, y2 = crop_bbox
                crop = image_array[y1:y2, x1:x2]

                if crop.size == 0:
                    report.num_invalid_bboxes += 1
                    continue

                crop_file_name = (
                    f"{safe_stem}__ann_{annotation.get('id', annotation_index)}."
                    f"{self.save_format}"
                )

                crop_path = split_output_dir / class_name / crop_file_name

                self._save_crop(crop_path, crop)

                report.num_crops_saved += 1

                bbox = annotation.get("bbox", [None, None, None, None])

                metadata_rows.append(
                    {
                        "split": split_name,
                        "crop_path": str(crop_path),
                        "original_image_id": image_id,
                        "original_file_name": file_name,
                        "resolved_image_path": str(resolved_image_path),
                        "annotation_id": annotation.get("id"),
                        "category_id": category_id,
                        "class_name": class_name,
                        "bbox_x": bbox[0],
                        "bbox_y": bbox[1],
                        "bbox_w": bbox[2],
                        "bbox_h": bbox[3],
                        "crop_x1": x1,
                        "crop_y1": y1,
                        "crop_x2": x2,
                        "crop_y2": y2,
                    }
                )

        if auto_write_metadata:
            metadata_path = split_output_dir / "metadata.csv"
            self.write_metadata_csv(
                metadata_rows=metadata_rows,
                output_path=metadata_path,
            )
            report.metadata_path = str(metadata_path)

        logger.info(
            "Create crops split=%s xong: %d crops, %d invalid bboxes, %d missing images.",
            split_name,
            report.num_crops_saved,
            report.num_invalid_bboxes,
            report.num_missing_images,
        )

        return report, metadata_rows

    def write_metadata_csv(
        self,
        metadata_rows: List[Dict[str, Any]],
        output_path: str | Path | None = None,
    ) -> Path:
        if output_path is None:
            output_path = self.output_root / "crops_metadata.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not metadata_rows:
            logger.warning("Không có metadata crop để lưu.")
            output_path.touch()
            return output_path

        fieldnames = list(metadata_rows[0].keys())

        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)

        logger.info("Đã lưu crop metadata tại: %s", output_path)

        return output_path

    def _compute_crop_bbox(
        self,
        bbox: List[float] | None,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        if bbox is None or len(bbox) != 4:
            return None

        x_min, y_min, bbox_width, bbox_height = map(float, bbox)

        if bbox_width <= 0 or bbox_height <= 0:
            return None

        margin_x = bbox_width * self.margin_ratio
        margin_y = bbox_height * self.margin_ratio

        x1 = int(max(0, x_min - margin_x))
        y1 = int(max(0, y_min - margin_y))
        x2 = int(min(image_width, x_min + bbox_width + margin_x))
        y2 = int(min(image_height, y_min + bbox_height + margin_y))

        crop_width = x2 - x1
        crop_height = y2 - y1

        if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
            return None

        return x1, y1, x2, y2

    def _resolve_image_path(
        self,
        file_name: str,
        image_root: Path | None,
        image_resolver: ImageResolver | None,
    ) -> Path | None:
        path = Path(file_name)

        if path.is_absolute() and path.exists():
            return path

        if image_resolver is not None:
            resolved = image_resolver.resolve(file_name)

            if resolved is not None:
                return resolved

        if image_root is not None:
            candidate = image_root / file_name

            if candidate.exists():
                return candidate

            candidate = image_root / path.name

            if candidate.exists():
                return candidate

        return None

    def _save_crop(self, crop_path: Path, crop) -> None:
        crop_path.parent.mkdir(parents=True, exist_ok=True)

        if self.save_format in {"jpg", "jpeg"}:
            cv2.imwrite(
                str(crop_path),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
        else:
            cv2.imwrite(str(crop_path), crop)

    @staticmethod
    def _safe_stem(file_name: str) -> str:
        normalized = str(file_name).replace("\\", "/").strip("/")
        normalized = normalized.replace(":", "_")
        path = Path(normalized)
        stem_without_suffix = str(path.with_suffix("")).replace("\\", "/")
        return stem_without_suffix.replace("/", "__")
