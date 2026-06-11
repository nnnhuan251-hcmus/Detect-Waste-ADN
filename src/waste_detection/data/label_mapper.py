import logging
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


logger = logging.getLogger("TacoLabelMapper")


@dataclass
class LabelMappingReport:
    original_num_categories: int = 0
    target_num_categories: int = 0
    original_categories: List[str] = field(default_factory=list)
    target_categories: List[str] = field(default_factory=list)
    mapped_original_categories: List[str] = field(default_factory=list)
    ignored_original_categories: List[str] = field(default_factory=list)
    unmapped_original_categories: List[str] = field(default_factory=list)
    categories_in_mapping_but_not_dataset: List[str] = field(default_factory=list)
    num_annotations_before: int = 0
    num_annotations_after: int = 0
    num_ignored_annotations: int = 0
    distribution_before: Dict[str, int] = field(default_factory=dict)
    distribution_after: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_num_categories": self.original_num_categories,
            "target_num_categories": self.target_num_categories,
            "original_categories": self.original_categories,
            "target_categories": self.target_categories,
            "mapped_original_categories": self.mapped_original_categories,
            "ignored_original_categories": self.ignored_original_categories,
            "unmapped_original_categories": self.unmapped_original_categories,
            "categories_in_mapping_but_not_dataset": self.categories_in_mapping_but_not_dataset,
            "num_annotations_before": self.num_annotations_before,
            "num_annotations_after": self.num_annotations_after,
            "num_ignored_annotations": self.num_ignored_annotations,
            "distribution_before": self.distribution_before,
            "distribution_after": self.distribution_after,
        }


class TacoLabelMapper:
    """
    Map category gốc của TACO sang 7 class đích.

    Input:
    - COCO dataset gốc.
    - mapping_label.json.

    Output:
    - COCO dataset mới với categories = 7 class.
    - mapping report để đưa vào báo cáo và debug.
    """

    def __init__(
        self,
        target_class_names: List[str],
        target_name_to_id: Dict[str, int],
        ignore_key: str = "ignore",
        fail_on_unmapped: bool = True,
    ) -> None:
        self.target_class_names = target_class_names
        self.target_name_to_id = target_name_to_id
        self.ignore_key = ignore_key
        self.fail_on_unmapped = fail_on_unmapped

    def transform(
        self,
        dataset: Dict[str, Any],
        mapping_dict: Dict[str, List[str]],
    ) -> Tuple[Dict[str, Any], LabelMappingReport]:
        original_categories = dataset.get("categories", [])
        original_annotations = dataset.get("annotations", [])

        category_id_to_name = {
            category["id"]: category["name"]
            for category in original_categories
            if "id" in category and "name" in category
        }

        reverse_mapping, ignored_names = self._build_reverse_mapping(mapping_dict)

        original_category_names = set(category_id_to_name.values())
        mapping_category_names = set(reverse_mapping.keys()).union(ignored_names)

        categories_in_mapping_but_not_dataset = sorted(
            mapping_category_names - original_category_names
        )

        report = LabelMappingReport(
            original_num_categories=len(original_categories),
            target_num_categories=len(self.target_class_names),
            original_categories=sorted(original_category_names),
            target_categories=list(self.target_class_names),
            categories_in_mapping_but_not_dataset=categories_in_mapping_but_not_dataset,
            num_annotations_before=len(original_annotations),
        )

        distribution_before = Counter()

        for annotation in original_annotations:
            category_name = category_id_to_name.get(annotation.get("category_id"))
            if category_name is not None:
                distribution_before[category_name] += 1

        report.distribution_before = dict(sorted(distribution_before.items()))

        target_categories = [
            {
                "id": self.target_name_to_id[class_name],
                "name": class_name,
                "supercategory": "waste",
            }
            for class_name in self.target_class_names
        ]

        new_annotations = []
        distribution_after = Counter()
        ignored_annotation_count = 0
        unmapped_category_names = set()
        mapped_category_names = set()
        ignored_category_names = set()

        next_annotation_id = 1

        for annotation in original_annotations:
            old_category_id = annotation.get("category_id")
            old_category_name = category_id_to_name.get(old_category_id)

            if old_category_name is None:
                unmapped_category_names.add(f"<unknown_category_id_{old_category_id}>")
                continue

            if old_category_name in ignored_names:
                ignored_annotation_count += 1
                ignored_category_names.add(old_category_name)
                continue

            target_class_name = reverse_mapping.get(old_category_name)

            if target_class_name is None:
                unmapped_category_names.add(old_category_name)
                continue

            if target_class_name not in self.target_name_to_id:
                raise KeyError(
                    f"Target class '{target_class_name}' không nằm trong target_name_to_id."
                )

            new_annotation = deepcopy(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["category_id"] = self.target_name_to_id[target_class_name]

            new_annotations.append(new_annotation)
            distribution_after[target_class_name] += 1
            mapped_category_names.add(old_category_name)
            next_annotation_id += 1

        report.num_annotations_after = len(new_annotations)
        report.num_ignored_annotations = ignored_annotation_count
        report.distribution_after = dict(sorted(distribution_after.items()))
        report.mapped_original_categories = sorted(mapped_category_names)
        report.ignored_original_categories = sorted(ignored_category_names)
        report.unmapped_original_categories = sorted(unmapped_category_names)

        if report.unmapped_original_categories:
            logger.warning(
                "Có %d category gốc chưa được map.",
                len(report.unmapped_original_categories),
            )

            for category_name in report.unmapped_original_categories:
                logger.warning("Unmapped category: %s", category_name)

            if self.fail_on_unmapped:
                raise KeyError(
                    "Có category gốc chưa được map. "
                    "Hãy cập nhật configs/data/mapping_label.json. "
                    f"Unmapped: {report.unmapped_original_categories}"
                )

        processed_dataset = {
            "info": deepcopy(dataset.get("info", {})),
            "licenses": deepcopy(dataset.get("licenses", [])),
            "images": deepcopy(dataset.get("images", [])),
            "annotations": new_annotations,
            "categories": target_categories,
        }

        logger.info(
            "Mapping label xong: %d annotations -> %d annotations, %d ignored.",
            report.num_annotations_before,
            report.num_annotations_after,
            report.num_ignored_annotations,
        )

        return processed_dataset, report

    def _build_reverse_mapping(
        self,
        mapping_dict: Dict[str, List[str]],
    ) -> Tuple[Dict[str, str], set[str]]:
        reverse_mapping: Dict[str, str] = {}
        ignored_names: set[str] = set()

        for target_class_name, original_names in mapping_dict.items():
            if target_class_name == self.ignore_key:
                ignored_names.update(original_names)
                continue

            normalized_target_class_name = target_class_name.strip().lower()

            if normalized_target_class_name not in self.target_name_to_id:
                raise KeyError(
                    f"Class '{target_class_name}' trong mapping không nằm trong "
                    f"danh sách class đích: {list(self.target_name_to_id.keys())}"
                )

            for original_name in original_names:
                if original_name in reverse_mapping:
                    raise ValueError(
                        f"Category gốc '{original_name}' bị map nhiều hơn một lần."
                    )

                reverse_mapping[original_name] = normalized_target_class_name

        return reverse_mapping, ignored_names
