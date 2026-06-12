import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from waste_detection.data.coco_validator import COCOValidator
from waste_detection.utils.io import IOUtils


logger = logging.getLogger("CocoReader")


class CocoReader:
    """
    Đọc và truy vấn COCO-style annotation.

    File này chỉ chịu trách nhiệm:
    - Load COCO JSON.
    - Validate nếu được yêu cầu.
    - Tạo các index tiện lợi: image_id -> image, category_id -> category_name.
    - Tóm tắt dataset.

    Không xử lý mapping label, split, convert YOLO hoặc crop.
    """

    def __init__(self, annotation_path: str | Path) -> None:
        self.annotation_path = Path(annotation_path)

    def load(self, validate: bool = True, strict: bool = False) -> Dict[str, Any]:
        dataset = IOUtils.load_json(self.annotation_path)

        if not isinstance(dataset, dict):
            raise ValueError(
                f"COCO annotation phải là dictionary, nhận được: {type(dataset)}"
            )

        if validate:
            COCOValidator.validate(
                dataset=dataset,
                strict=strict,
                check_bbox_inside_image=False,
            )

        logger.info(
            "Đã load COCO dataset: %d images, %d annotations, %d categories.",
            len(dataset.get("images", [])),
            len(dataset.get("annotations", [])),
            len(dataset.get("categories", [])),
        )

        return dataset

    @staticmethod
    def summarize(dataset: Dict[str, Any]) -> Dict[str, Any]:
        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])
        categories = dataset.get("categories", [])

        category_id_to_name = CocoReader.category_id_to_name(dataset)

        annotation_count_by_category = defaultdict(int)

        for annotation in annotations:
            category_id = annotation.get("category_id")
            category_name = category_id_to_name.get(category_id, f"unknown_{category_id}")
            annotation_count_by_category[category_name] += 1

        summary = {
            "num_images": len(images),
            "num_annotations": len(annotations),
            "num_categories": len(categories),
            "categories": CocoReader.get_category_names(dataset),
            "annotation_count_by_category": dict(
                sorted(annotation_count_by_category.items())
            ),
        }

        return summary

    @staticmethod
    def get_categories(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        categories = dataset.get("categories", [])
        return sorted(categories, key=lambda category: category.get("id", 0))

    @staticmethod
    def get_category_names(dataset: Dict[str, Any]) -> List[str]:
        return [category.get("name", "") for category in CocoReader.get_categories(dataset)]

    @staticmethod
    def category_id_to_name(dataset: Dict[str, Any]) -> Dict[int, str]:
        return {
            category["id"]: category["name"]
            for category in dataset.get("categories", [])
            if "id" in category and "name" in category
        }

    @staticmethod
    def category_name_to_id(dataset: Dict[str, Any]) -> Dict[str, int]:
        return {
            category["name"]: category["id"]
            for category in dataset.get("categories", [])
            if "id" in category and "name" in category
        }

    @staticmethod
    def image_id_to_image(dataset: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        return {
            image["id"]: image
            for image in dataset.get("images", [])
            if "id" in image
        }

    @staticmethod
    def annotations_by_image_id(
        dataset: Dict[str, Any],
    ) -> Dict[int, List[Dict[str, Any]]]:
        grouped_annotations: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for annotation in dataset.get("annotations", []):
            image_id = annotation.get("image_id")
            
            if image_id is None:
                continue
            
            grouped_annotations[image_id].append(annotation)

        return dict(grouped_annotations)
