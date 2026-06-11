import logging
from copy import deepcopy
from typing import Any, Dict, List

logger = logging.getLogger("DatasetUtils")

class DatasetUtils:
    """
    Các hàm tiện ích stateless để xử lý COCO-style dataset.
    """

    @staticmethod
    def filter_annotations(
        annotations: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Giữ lại các annotation thuộc về danh sách image hợp lệ.
        """
        logger.info("Bắt đầu lọc annotation cho %d ảnh.", len(images))

        valid_image_ids = {image["id"] for image in images}

        filtered_annotations = []
        dropped_count = 0

        for annotation in annotations:
            image_id = annotation.get("image_id")

            if image_id in valid_image_ids:
                filtered_annotations.append(annotation)
            else:
                dropped_count += 1

        logger.info(
            "Lọc annotation xong: giữ %d / %d annotations. Loại %d annotations.",
            len(filtered_annotations),
            len(annotations),
            dropped_count,
        )

        return filtered_annotations

    @staticmethod
    def create_coco_subset(
        dataset: Dict[str, Any],
        images: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Tạo COCO subset từ danh sách images.

        Dùng cho train/val/test split.
        """
        annotations = dataset.get("annotations", [])
        filtered_annotations = DatasetUtils.filter_annotations(
            annotations=annotations,
            images=images,
        )

        subset = {
            "info": deepcopy(dataset.get("info", {})),
            "licenses": deepcopy(dataset.get("licenses", [])),
            "images": deepcopy(images),
            "annotations": deepcopy(filtered_annotations),
            "categories": deepcopy(dataset.get("categories", [])),
        }

        logger.info(
            "Tạo COCO subset xong: %d images, %d annotations, %d categories.",
            len(subset["images"]),
            len(subset["annotations"]),
            len(subset["categories"]),
        )

        return subset

    @staticmethod
    def concatenate_datasets(
        list_of_datasets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Gộp nhiều COCO datasets.

        Lưu ý:
        - Không làm thay đổi dataset gốc.
        - Tự cấp phát lại image_id và annotation_id.
        - Giả định categories giữa các dataset giống nhau.
        """
        if not list_of_datasets:
            logger.warning("Danh sách dataset rỗng. Trả về COCO dataset rỗng.")

            return {
                "info": {},
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [],
            }

        logger.info("Bắt đầu gộp %d datasets.", len(list_of_datasets))

        base_categories = list_of_datasets[0].get("categories", [])

        concat_dataset = {
            "info": deepcopy(list_of_datasets[0].get("info", {})),
            "licenses": deepcopy(list_of_datasets[0].get("licenses", [])),
            "categories": deepcopy(base_categories),
            "images": [],
            "annotations": [],
        }

        next_image_id = 1
        next_annotation_id = 1

        for dataset_index, dataset in enumerate(list_of_datasets):
            logger.info("Đang xử lý dataset #%d", dataset_index + 1)

            current_categories = dataset.get("categories", [])

            if current_categories != base_categories:
                logger.warning(
                    "Dataset #%d có categories khác dataset đầu tiên. "
                    "Cần kiểm tra lại trước khi dùng cho training.",
                    dataset_index + 1,
                )

            image_id_mapping = {}

            for image in dataset.get("images", []):
                old_image_id = image["id"]
                new_image = deepcopy(image)

                new_image["id"] = next_image_id
                image_id_mapping[old_image_id] = next_image_id

                concat_dataset["images"].append(new_image)
                next_image_id += 1

            dropped_orphan_annotations = 0

            for annotation in dataset.get("annotations", []):
                old_image_id = annotation.get("image_id")

                if old_image_id not in image_id_mapping:
                    dropped_orphan_annotations += 1
                    continue

                new_annotation = deepcopy(annotation)
                new_annotation["id"] = next_annotation_id
                new_annotation["image_id"] = image_id_mapping[old_image_id]

                concat_dataset["annotations"].append(new_annotation)
                next_annotation_id += 1

            if dropped_orphan_annotations > 0:
                logger.warning(
                    "Dataset #%d: bỏ %d annotations vì image_id không tồn tại.",
                    dataset_index + 1,
                    dropped_orphan_annotations,
                )

        logger.info(
            "Gộp dataset xong: %d images, %d annotations.",
            len(concat_dataset["images"]),
            len(concat_dataset["annotations"]),
        )

        return concat_dataset
