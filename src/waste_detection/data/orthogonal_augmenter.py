from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from tqdm import tqdm

from waste_detection.data.image_resolver import ImageResolver


logger = logging.getLogger("OrthogonalAugmenter")


@dataclass
class OrthogonalAugmentationReport:
    angles: List[int]
    output_image_dir: str
    num_original_images: int = 0
    num_original_annotations: int = 0
    num_augmented_images: int = 0
    num_augmented_annotations: int = 0
    num_missing_images: int = 0
    num_invalid_bboxes: int = 0
    missing_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "angles": self.angles,
            "output_image_dir": self.output_image_dir,
            "num_original_images": self.num_original_images,
            "num_original_annotations": self.num_original_annotations,
            "num_augmented_images": self.num_augmented_images,
            "num_augmented_annotations": self.num_augmented_annotations,
            "num_missing_images": self.num_missing_images,
            "num_invalid_bboxes": self.num_invalid_bboxes,
            "missing_files": self.missing_files,
        }


class OrthogonalAugmenter:
    """
    Offline augmentation cho object detection bằng xoay trực giao.

    Hỗ trợ:
    - 90 độ clockwise
    - 180 độ
    - 270 độ clockwise

    Không dùng diagonal rotation để tránh bbox axis-aligned chứa quá nhiều background.
    """

    SUPPORTED_ANGLES = {90, 180, 270}

    def __init__(
        self,
        output_image_dir: str | Path,
        angles: List[int] | None = None,
        use_absolute_file_names: bool = True,
        jpeg_quality: int = 95,
    ) -> None:
        self.output_image_dir = Path(output_image_dir)
        self.angles = angles or [90, 180]
        self.use_absolute_file_names = use_absolute_file_names
        self.jpeg_quality = jpeg_quality

        invalid_angles = set(self.angles) - self.SUPPORTED_ANGLES
        if invalid_angles:
            raise ValueError(
                f"Chỉ hỗ trợ angles={self.SUPPORTED_ANGLES}. "
                f"Nhận được angle không hợp lệ: {invalid_angles}"
            )

    def augment_dataset(
        self,
        dataset: Dict[str, Any],
        image_resolver: ImageResolver,
    ) -> Tuple[Dict[str, Any], OrthogonalAugmentationReport]:
        self.output_image_dir.mkdir(parents=True, exist_ok=True)

        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])

        annotations_by_image_id: Dict[int, List[Dict[str, Any]]] = {}

        for annotation in annotations:
            image_id = annotation.get("image_id")
            annotations_by_image_id.setdefault(image_id, []).append(annotation)

        next_image_id = self._next_id(images)
        next_annotation_id = self._next_id(annotations)

        augmented_dataset = {
            "info": deepcopy(dataset.get("info", {})),
            "licenses": deepcopy(dataset.get("licenses", [])),
            "images": deepcopy(images),
            "annotations": deepcopy(annotations),
            "categories": deepcopy(dataset.get("categories", [])),
        }

        report = OrthogonalAugmentationReport(
            angles=self.angles,
            output_image_dir=str(self.output_image_dir),
            num_original_images=len(images),
            num_original_annotations=len(annotations),
        )

        for image_info in tqdm(images, desc="Orthogonal augmentation"):
            image_id = image_info.get("id")
            file_name = image_info.get("file_name")

            if not file_name:
                report.num_missing_images += 1
                report.missing_files.append("<missing_file_name>")
                continue

            source_path = image_resolver.resolve(file_name)

            if source_path is None:
                report.num_missing_images += 1
                report.missing_files.append(file_name)
                continue

            image_bgr = cv2.imread(str(source_path))

            if image_bgr is None:
                report.num_missing_images += 1
                report.missing_files.append(file_name)
                continue

            original_height, original_width = image_bgr.shape[:2]
            source_annotations = annotations_by_image_id.get(image_id, [])

            if not source_annotations:
                continue

            for angle in self.angles:
                rotated_image = self._rotate_image(image_bgr, angle)
                rotated_height, rotated_width = rotated_image.shape[:2]

                safe_stem = self._safe_stem(file_name)
                output_file_name = f"{safe_stem}__rot{angle}.jpg"
                output_path = self.output_image_dir / output_file_name

                cv2.imwrite(
                    str(output_path),
                    rotated_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
                )

                if self.use_absolute_file_names:
                    new_file_name = str(output_path.resolve())
                else:
                    new_file_name = output_file_name

                new_image = deepcopy(image_info)
                new_image["id"] = next_image_id
                new_image["file_name"] = new_file_name
                new_image["width"] = rotated_width
                new_image["height"] = rotated_height
                new_image["augmentation"] = f"rot{angle}"
                new_image["source_image_id"] = image_id

                augmented_dataset["images"].append(new_image)
                report.num_augmented_images += 1

                for annotation in source_annotations:
                    rotated_bbox = self._rotate_coco_bbox(
                        bbox=annotation.get("bbox"),
                        image_width=original_width,
                        image_height=original_height,
                        angle=angle,
                    )

                    if rotated_bbox is None:
                        report.num_invalid_bboxes += 1
                        continue

                    new_annotation = deepcopy(annotation)
                    new_annotation["id"] = next_annotation_id
                    new_annotation["image_id"] = next_image_id
                    new_annotation["bbox"] = rotated_bbox
                    new_annotation["area"] = rotated_bbox[2] * rotated_bbox[3]
                    new_annotation["augmentation"] = f"rot{angle}"
                    new_annotation["source_annotation_id"] = annotation.get("id")

                    augmented_dataset["annotations"].append(new_annotation)

                    report.num_augmented_annotations += 1
                    next_annotation_id += 1

                next_image_id += 1

        logger.info(
            "Orthogonal augmentation xong: +%d images, +%d annotations.",
            report.num_augmented_images,
            report.num_augmented_annotations,
        )

        return augmented_dataset, report

    @staticmethod
    def _rotate_image(image_bgr, angle: int):
        if angle == 90:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)

        if angle == 180:
            return cv2.rotate(image_bgr, cv2.ROTATE_180)

        if angle == 270:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        raise ValueError(f"Angle không hỗ trợ: {angle}")

    @staticmethod
    def _rotate_coco_bbox(
        bbox: List[float] | None,
        image_width: int,
        image_height: int,
        angle: int,
    ) -> List[float] | None:
        if bbox is None or len(bbox) != 4:
            return None

        x, y, width, height = map(float, bbox)

        if width <= 0 or height <= 0:
            return None

        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height

        corners = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]

        rotated_corners = [
            OrthogonalAugmenter._rotate_point(
                point_x=point_x,
                point_y=point_y,
                image_width=image_width,
                image_height=image_height,
                angle=angle,
            )
            for point_x, point_y in corners
        ]

        xs = [point[0] for point in rotated_corners]
        ys = [point[1] for point in rotated_corners]

        new_x1 = max(0.0, min(xs))
        new_y1 = max(0.0, min(ys))
        new_x2 = max(xs)
        new_y2 = max(ys)

        new_width = new_x2 - new_x1
        new_height = new_y2 - new_y1

        if new_width <= 0 or new_height <= 0:
            return None

        return [
            float(new_x1),
            float(new_y1),
            float(new_width),
            float(new_height),
        ]

    @staticmethod
    def _rotate_point(
        point_x: float,
        point_y: float,
        image_width: int,
        image_height: int,
        angle: int,
    ) -> tuple[float, float]:
        if angle == 90:
            return image_height - point_y, point_x

        if angle == 180:
            return image_width - point_x, image_height - point_y

        if angle == 270:
            return point_y, image_width - point_x

        raise ValueError(f"Angle không hỗ trợ: {angle}")

    @staticmethod
    def _next_id(items: List[Dict[str, Any]]) -> int:
        existing_ids = [item.get("id", 0) for item in items if item.get("id") is not None]

        if not existing_ids:
            return 1

        return max(existing_ids) + 1

    @staticmethod
    def _safe_stem(file_name: str) -> str:
        path = Path(file_name)
        parts = list(path.with_suffix("").parts)
        return "__".join(parts)
