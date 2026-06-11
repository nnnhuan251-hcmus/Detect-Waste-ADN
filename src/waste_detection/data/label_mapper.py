from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


logger = logging.getLogger("TacoLabelMapper")


@dataclass
class LabelMappingReport:
    """
    Report sau khi map nhãn COCO gốc về 7 class đích.
    """

    num_original_categories: int = 0
    num_target_categories: int = 0
    num_original_annotations: int = 0
    num_mapped_annotations: int = 0
    num_ignored_annotations: int = 0
    num_unmapped_annotations: int = 0

    mapped_category_names: Dict[str, str] = field(default_factory=dict)
    ignored_category_names: List[str] = field(default_factory=list)
    unmapped_category_names: List[str] = field(default_factory=list)

    old_category_id_to_new_category_id: Dict[int, int] = field(default_factory=dict)

    duplicate_original_annotation_ids: List[int] = field(default_factory=list)
    annotation_id_reindexed: bool = True

    @property
    def num_unmapped_categories(self) -> int:
        return len(set(self.unmapped_category_names))

    @property
    def num_ignored_categories(self) -> int:
        return len(set(self.ignored_category_names))

    @property
    def num_mapped_categories(self) -> int:
        return len(set(self.mapped_category_names.keys()))

    @property
    def num_duplicate_original_annotation_ids(self) -> int:
        return len(set(self.duplicate_original_annotation_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_original_categories": self.num_original_categories,
            "num_target_categories": self.num_target_categories,
            "num_original_annotations": self.num_original_annotations,
            "num_mapped_annotations": self.num_mapped_annotations,
            "num_ignored_annotations": self.num_ignored_annotations,
            "num_unmapped_annotations": self.num_unmapped_annotations,
            "num_mapped_categories": self.num_mapped_categories,
            "num_ignored_categories": self.num_ignored_categories,
            "num_unmapped_categories": self.num_unmapped_categories,
            "mapped_category_names": self.mapped_category_names,
            "ignored_category_names": sorted(set(self.ignored_category_names)),
            "unmapped_category_names": sorted(set(self.unmapped_category_names)),
            "old_category_id_to_new_category_id": self.old_category_id_to_new_category_id,
            "annotation_id_reindexed": self.annotation_id_reindexed,
            "num_duplicate_original_annotation_ids": self.num_duplicate_original_annotation_ids,
            "duplicate_original_annotation_ids": sorted(
                set(self.duplicate_original_annotation_ids)
            ),
        }


class TacoLabelMapper:
    """
    Map category gốc của TACO/COCO về class đích.

    mapping_dict format:

    {
        "plastic": ["Clear plastic bottle", ...],
        "paper": ["Normal paper", ...],
        ...
        "ignore": [...]
    }

    Output:
    - categories được thay bằng 7 class target id 0..6.
    - annotations được đổi category_id tương ứng.
    - annotation thuộc ignore sẽ bị bỏ.
    - annotation id được re-index để tránh lỗi duplicate id từ file COCO gốc.
    """

    def __init__(
        self,
        target_class_names: List[str],
        target_name_to_id: Dict[str, int],
        ignore_key: str = "ignore",
        fail_on_unmapped: bool = True,
    ) -> None:
        self.target_class_names = list(target_class_names)
        self.target_name_to_id = dict(target_name_to_id)
        self.ignore_key = ignore_key
        self.fail_on_unmapped = fail_on_unmapped

        self._validate_target_classes()

    def transform(
        self,
        dataset: Dict[str, Any],
        mapping_dict: Dict[str, List[str]],
    ) -> Tuple[Dict[str, Any], LabelMappingReport]:
        dataset = deepcopy(dataset)

        categories = dataset.get("categories", [])
        annotations = dataset.get("annotations", [])

        report = LabelMappingReport(
            num_original_categories=len(categories),
            num_target_categories=len(self.target_class_names),
            num_original_annotations=len(annotations),
        )

        report.duplicate_original_annotation_ids = self._find_duplicate_annotation_ids(
            annotations
        )

        original_name_to_target_name, ignored_names = self._build_reverse_mapping(
            mapping_dict=mapping_dict,
        )

        old_category_id_to_name = {
            int(category["id"]): str(category["name"])
            for category in categories
        }

        old_category_id_to_new_category_id: Dict[int, int] = {}
        ignored_category_ids = set()
        unmapped_category_names = []

        for old_category_id, original_name in old_category_id_to_name.items():
            if original_name in ignored_names:
                ignored_category_ids.add(old_category_id)
                report.ignored_category_names.append(original_name)
                continue

            target_name = original_name_to_target_name.get(original_name)

            if target_name is None:
                unmapped_category_names.append(original_name)
                report.unmapped_category_names.append(original_name)
                continue

            new_category_id = self.target_name_to_id[target_name]
            old_category_id_to_new_category_id[old_category_id] = new_category_id
            report.mapped_category_names[original_name] = target_name
            report.old_category_id_to_new_category_id[old_category_id] = new_category_id

        if unmapped_category_names and self.fail_on_unmapped:
            raise KeyError(
                "Có category chưa được map. "
                f"Unmapped categories: {sorted(set(unmapped_category_names))}"
            )

        new_annotations = []
        next_annotation_id = 1

        for annotation in annotations:
            old_category_id = int(annotation.get("category_id"))

            if old_category_id in ignored_category_ids:
                report.num_ignored_annotations += 1
                continue

            if old_category_id not in old_category_id_to_new_category_id:
                report.num_unmapped_annotations += 1

                if self.fail_on_unmapped:
                    original_name = old_category_id_to_name.get(
                        old_category_id,
                        f"category_id={old_category_id}",
                    )
                    raise KeyError(
                        f"Annotation id={annotation.get('id')} có category chưa map: "
                        f"{original_name}"
                    )

                continue

            old_annotation_id = annotation.get("id")

            new_annotation = deepcopy(annotation)

            # Quan trọng:
            # TACO official annotations.json có thể có duplicate annotation id.
            # Processed COCO phải có id unique để split/convert/train không lỗi.
            new_annotation["source_annotation_id"] = old_annotation_id
            new_annotation["id"] = next_annotation_id
            new_annotation["category_id"] = old_category_id_to_new_category_id[
                old_category_id
            ]

            new_annotations.append(new_annotation)

            next_annotation_id += 1
            report.num_mapped_annotations += 1

        dataset["categories"] = self._build_target_categories()
        dataset["annotations"] = new_annotations

        logger.info(
            "Label mapping xong: %d mapped annotations, %d ignored, %d unmapped, %d duplicate original annotation ids.",
            report.num_mapped_annotations,
            report.num_ignored_annotations,
            report.num_unmapped_annotations,
            report.num_duplicate_original_annotation_ids,
        )

        return dataset, report

    def _build_reverse_mapping(
        self,
        mapping_dict: Dict[str, List[str]],
    ) -> tuple[Dict[str, str], set[str]]:
        original_name_to_target_name: Dict[str, str] = {}
        ignored_names = set()

        for target_name, original_names in mapping_dict.items():
            if target_name == self.ignore_key:
                ignored_names.update(str(name) for name in original_names)
                continue

            if target_name not in self.target_name_to_id:
                raise KeyError(
                    f"Mapping chứa target class không hợp lệ: {target_name}. "
                    f"Target classes hợp lệ: {self.target_class_names}"
                )

            for original_name in original_names:
                original_name = str(original_name)

                if original_name in original_name_to_target_name:
                    previous_target = original_name_to_target_name[original_name]
                    raise ValueError(
                        f"Category '{original_name}' bị map vào nhiều class: "
                        f"{previous_target} và {target_name}"
                    )

                original_name_to_target_name[original_name] = target_name

        return original_name_to_target_name, ignored_names

    def _build_target_categories(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": self.target_name_to_id[class_name],
                "name": class_name,
                "supercategory": "waste",
            }
            for class_name in self.target_class_names
        ]

    def _validate_target_classes(self) -> None:
        if len(set(self.target_class_names)) != len(self.target_class_names):
            raise ValueError("target_class_names bị trùng.")

        if set(self.target_class_names) != set(self.target_name_to_id.keys()):
            raise ValueError(
                "target_name_to_id không khớp target_class_names. "
                f"names={self.target_class_names}, mapping={self.target_name_to_id}"
            )

        expected_ids = set(range(len(self.target_class_names)))
        actual_ids = set(self.target_name_to_id.values())

        if actual_ids != expected_ids:
            raise ValueError(
                "target ids phải liên tục từ 0 đến num_classes - 1. "
                f"Expected={expected_ids}, got={actual_ids}"
            )

    @staticmethod
    def _find_duplicate_annotation_ids(
        annotations: List[Dict[str, Any]],
    ) -> List[int]:
        seen_ids = set()
        duplicate_ids = []

        for annotation in annotations:
            annotation_id = annotation.get("id")

            if annotation_id is None:
                continue

            try:
                annotation_id = int(annotation_id)
            except (TypeError, ValueError):
                continue

            if annotation_id in seen_ids:
                duplicate_ids.append(annotation_id)
            else:
                seen_ids.add(annotation_id)

        return duplicate_ids
