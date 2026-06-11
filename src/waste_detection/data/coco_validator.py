import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("COCOValidator")

@dataclass
class COCOValidationReport:
    """
    Report kiểm tra COCO dataset.
    Raw data có thể dùng strict=False.
    Processed data nên dùng strict=True.
    Bbox vượt biên ảnh được xem là warning vì sau này crop_builder.py có thể clamp.
    """
    num_images: int = 0
    num_annotations: int = 0
    num_categories: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_images": self.num_images,
            "num_annotations": self.num_annotations,
            "num_categories": self.num_categories,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "is_valid": self.is_valid,
        }


class COCOValidator:
    """
    Kiểm tra tính hợp lệ của COCO-style annotation.

    Có 2 mức:
    - strict=True: gặp lỗi thì raise ValueError.
    - strict=False: chỉ log lỗi và trả về False/report.
    """

    @staticmethod
    def validate(
        dataset: Dict[str, Any],
        strict: bool = True,
        check_bbox_inside_image: bool = False,
        max_messages: int = 20,
    ) -> bool:
        report = COCOValidator.validate_with_report(
            dataset=dataset,
            strict=strict,
            check_bbox_inside_image=check_bbox_inside_image,
            max_messages=max_messages,
        )

        return report.is_valid

    @staticmethod
    def validate_with_report(
        dataset: Dict[str, Any],
        strict: bool = True,
        check_bbox_inside_image: bool = False,
        max_messages: int = 20,
    ) -> COCOValidationReport:
        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])
        categories = dataset.get("categories", [])

        report = COCOValidationReport(
            num_images=len(images),
            num_annotations=len(annotations),
            num_categories=len(categories),
        )

        if not isinstance(images, list):
            report.errors.append("Field 'images' phải là list.")

        if not isinstance(annotations, list):
            report.errors.append("Field 'annotations' phải là list.")

        if not isinstance(categories, list):
            report.errors.append("Field 'categories' phải là list.")

        if report.errors:
            COCOValidator._finalize_report(report, strict, max_messages)
            return report

        if not images:
            report.errors.append("Dataset không có images.")

        if not categories:
            report.errors.append("Dataset không có categories.")

        image_ids = []
        category_ids = []
        annotation_ids = []

        image_id_to_image = {}

        for image in images:
            image_id = image.get("id")
            image_ids.append(image_id)

            if image_id is None:
                report.errors.append(f"Image thiếu id: {image}")

            if image_id in image_id_to_image:
                report.errors.append(f"Trùng image id: {image_id}")
            else:
                image_id_to_image[image_id] = image

            file_name = image.get("file_name")
            width = image.get("width")
            height = image.get("height")

            if not file_name:
                report.errors.append(f"Image id={image_id} thiếu file_name.")

            if width is None or height is None:
                report.errors.append(f"Image id={image_id} thiếu width/height.")
            elif width <= 0 or height <= 0:
                report.errors.append(
                    f"Image id={image_id} có width/height không hợp lệ: "
                    f"width={width}, height={height}"
                )

        for category in categories:
            category_id = category.get("id")
            category_name = category.get("name")

            category_ids.append(category_id)

            if category_id is None:
                report.errors.append(f"Category thiếu id: {category}")

            if not category_name:
                report.errors.append(f"Category id={category_id} thiếu name.")

        duplicate_image_ids = COCOValidator._find_duplicates(image_ids)
        duplicate_category_ids = COCOValidator._find_duplicates(category_ids)

        for duplicated_id in duplicate_image_ids:
            report.errors.append(f"Trùng image id: {duplicated_id}")

        for duplicated_id in duplicate_category_ids:
            report.errors.append(f"Trùng category id: {duplicated_id}")

        image_id_set = set(image_ids)
        category_id_set = set(category_ids)

        for annotation in annotations:
            annotation_id = annotation.get("id")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")

            annotation_ids.append(annotation_id)

            if annotation_id is None:
                report.errors.append(f"Annotation thiếu id: {annotation}")

            if image_id not in image_id_set:
                report.errors.append(
                    f"Annotation id={annotation_id} có image_id={image_id} không tồn tại."
                )

            if category_id not in category_id_set:
                report.errors.append(
                    f"Annotation id={annotation_id} có category_id={category_id} không tồn tại."
                )

            if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                report.errors.append(
                    f"Annotation id={annotation_id} có bbox không hợp lệ: {bbox}"
                )
                continue

            x, y, width, height = bbox

            if width <= 0 or height <= 0:
                report.errors.append(
                    f"Annotation id={annotation_id} có bbox width/height <= 0: {bbox}"
                )

            if check_bbox_inside_image and image_id in image_id_to_image:
                image = image_id_to_image[image_id]
                image_width = image.get("width")
                image_height = image.get("height")

                if image_width is not None and image_height is not None:
                    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
                        report.warnings.append(
                            f"Annotation id={annotation_id} có bbox vượt biên ảnh: "
                            f"bbox={bbox}, image_size=({image_width}, {image_height})"
                        )

            area = annotation.get("area")

            if area is not None and area <= 0:
                report.warnings.append(
                    f"Annotation id={annotation_id} có area <= 0: {area}"
                )

        duplicate_annotation_ids = COCOValidator._find_duplicates(annotation_ids)

        for duplicated_id in duplicate_annotation_ids:
            report.errors.append(f"Trùng annotation id: {duplicated_id}")

        COCOValidator._finalize_report(report, strict, max_messages)

        return report

    @staticmethod
    def _find_duplicates(values: List[Any]) -> List[Any]:
        seen = set()
        duplicates = set()

        for value in values:
            if value is None:
                continue

            if value in seen:
                duplicates.add(value)
            else:
                seen.add(value)

        return sorted(duplicates)

    @staticmethod
    def _finalize_report(
        report: COCOValidationReport,
        strict: bool,
        max_messages: int,
    ) -> None:
        if report.errors:
            logger.error(
                "COCO validation phát hiện %d lỗi và %d cảnh báo.",
                len(report.errors),
                len(report.warnings),
            )

            for error in report.errors[:max_messages]:
                logger.error(error)

            if len(report.errors) > max_messages:
                logger.error(
                    "Còn %d lỗi khác chưa hiển thị.",
                    len(report.errors) - max_messages,
                )

            for warning in report.warnings[:max_messages]:
                logger.warning(warning)

            if strict:
                raise ValueError("COCO dataset không hợp lệ. Xem log để biết chi tiết.")

        else:
            logger.info(
                "COCO dataset hợp lệ: %d images, %d annotations, %d categories.",
                report.num_images,
                report.num_annotations,
                report.num_categories,
            )

            if report.warnings:
                logger.warning(
                    "COCO dataset có %d cảnh báo nhưng không có lỗi nghiêm trọng.",
                    len(report.warnings),
                )

                for warning in report.warnings[:max_messages]:
                    logger.warning(warning)
