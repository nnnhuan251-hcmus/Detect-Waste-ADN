from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


logger = logging.getLogger("BBoxSanitizer")


@dataclass
class BBoxSanitizerReport:
    num_annotations_input: int = 0
    num_annotations_output: int = 0
    num_bboxes_clamped: int = 0
    num_bboxes_dropped: int = 0
    num_missing_images: int = 0
    clamped_annotation_ids: List[int] = field(default_factory=list)
    dropped_annotation_ids: List[int] = field(default_factory=list)
    missing_image_annotation_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_annotations_input": self.num_annotations_input,
            "num_annotations_output": self.num_annotations_output,
            "num_bboxes_clamped": self.num_bboxes_clamped,
            "num_bboxes_dropped": self.num_bboxes_dropped,
            "num_missing_images": self.num_missing_images,
            "clamped_annotation_ids": self.clamped_annotation_ids,
            "dropped_annotation_ids": self.dropped_annotation_ids,
            "missing_image_annotation_ids": self.missing_image_annotation_ids,
        }


class BBoxSanitizer:
    """
    Chuẩn hóa bbox COCO [x, y, w, h].

    Mục tiêu:
    - Clamp bbox vào trong biên ảnh.
    - Recompute width, height, area sau khi clamp.
    - Drop bbox không hợp lệ nếu width/height quá nhỏ.
    - Giữ lại bbox gốc trong source_bbox để audit.

    Nên chạy sau label mapping và trước split/YOLO/crop.
    """

    def __init__(
        self,
        min_width: float = 1.0,
        min_height: float = 1.0,
        clamp_to_image: bool = True,
        drop_invalid_bbox: bool = True,
    ) -> None:
        self.min_width = float(min_width)
        self.min_height = float(min_height)
        self.clamp_to_image = bool(clamp_to_image)
        self.drop_invalid_bbox = bool(drop_invalid_bbox)

    def sanitize(
        self,
        dataset: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], BBoxSanitizerReport]:
        dataset = deepcopy(dataset)

        images_by_id = {
            image["id"]: image
            for image in dataset.get("images", [])
        }

        annotations = dataset.get("annotations", [])

        report = BBoxSanitizerReport(
            num_annotations_input=len(annotations),
        )

        sanitized_annotations: List[Dict[str, Any]] = []

        for annotation in annotations:
            annotation_id = annotation.get("id")
            image_id = annotation.get("image_id")
            image_info = images_by_id.get(image_id)

            if image_info is None:
                report.num_missing_images += 1
                report.missing_image_annotation_ids.append(annotation_id)

                if not self.drop_invalid_bbox:
                    sanitized_annotations.append(annotation)

                continue

            image_width = float(image_info["width"])
            image_height = float(image_info["height"])

            bbox = annotation.get("bbox")

            sanitized_bbox, was_clamped = self._sanitize_bbox(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
            )

            if sanitized_bbox is None:
                report.num_bboxes_dropped += 1
                report.dropped_annotation_ids.append(annotation_id)

                if not self.drop_invalid_bbox:
                    sanitized_annotations.append(annotation)

                continue

            new_annotation = deepcopy(annotation)

            if was_clamped:
                new_annotation["source_bbox"] = bbox
                new_annotation["bbox_was_clamped"] = True
                report.num_bboxes_clamped += 1
                report.clamped_annotation_ids.append(annotation_id)
            else:
                new_annotation["bbox_was_clamped"] = False

            new_annotation["bbox"] = sanitized_bbox
            new_annotation["area"] = sanitized_bbox[2] * sanitized_bbox[3]

            sanitized_annotations.append(new_annotation)

        dataset["annotations"] = sanitized_annotations
        report.num_annotations_output = len(sanitized_annotations)

        logger.info(
            "BBox sanitize xong: input=%d, output=%d, clamped=%d, dropped=%d, missing_images=%d.",
            report.num_annotations_input,
            report.num_annotations_output,
            report.num_bboxes_clamped,
            report.num_bboxes_dropped,
            report.num_missing_images,
        )

        return dataset, report

    def _sanitize_bbox(
        self,
        bbox: List[float] | None,
        image_width: float,
        image_height: float,
    ) -> tuple[List[float] | None, bool]:
        if bbox is None or len(bbox) != 4:
            return None, False

        x, y, width, height = map(float, bbox)

        if width <= 0 or height <= 0:
            return None, False

        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height

        original = (x1, y1, x2, y2)

        if self.clamp_to_image:
            x1 = max(0.0, min(x1, image_width))
            y1 = max(0.0, min(y1, image_height))
            x2 = max(0.0, min(x2, image_width))
            y2 = max(0.0, min(y2, image_height))

        new_width = x2 - x1
        new_height = y2 - y1

        if new_width < self.min_width or new_height < self.min_height:
            return None, original != (x1, y1, x2, y2)

        sanitized_bbox = [
            float(x1),
            float(y1),
            float(new_width),
            float(new_height),
        ]

        was_clamped = original != (x1, y1, x2, y2)

        return sanitized_bbox, was_clamped
