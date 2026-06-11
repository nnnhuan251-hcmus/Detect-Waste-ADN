from __future__ import annotations

import logging
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from waste_detection.data.adapters.base_dataset_adapter import BaseDatasetAdapter
from waste_detection.data.coco_validator import COCOValidator
from waste_detection.data.label_mapper import TacoLabelMapper
from waste_detection.utils.io import IOUtils


logger = logging.getLogger("RoboflowCocoAdapter")


@dataclass
class RoboflowCocoImportReport:
    dataset_dir: str
    output_coco_dir: str
    output_image_dir: str
    imported_splits: List[str] = field(default_factory=list)
    skipped_splits: List[str] = field(default_factory=list)
    num_images: Dict[str, int] = field(default_factory=dict)
    num_annotations: Dict[str, int] = field(default_factory=dict)
    num_categories: Dict[str, int] = field(default_factory=dict)
    mapping_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "output_coco_dir": self.output_coco_dir,
            "output_image_dir": self.output_image_dir,
            "imported_splits": self.imported_splits,
            "skipped_splits": self.skipped_splits,
            "num_images": self.num_images,
            "num_annotations": self.num_annotations,
            "num_categories": self.num_categories,
            "mapping_reports": self.mapping_reports,
        }


class RoboflowCocoAdapter(BaseDatasetAdapter):
    """
    Import Roboflow COCO dataset về format COCO split chuẩn của repo.

    Mục tiêu:
    - Không phá pipeline TACO chính thức.
    - Cho phép dùng lại yolo_converter.py và crop_builder.py.
    - Hỗ trợ optional mapping về 7 class.
    """

    SPLIT_NAME_MAP = {
        "train": "train",
        "valid": "val",
        "val": "val",
        "test": "test",
    }

    def __init__(
        self,
        dataset_dir: str | Path,
        output_coco_dir: str | Path,
        output_image_dir: str | Path,
        target_class_names: List[str],
        target_name_to_id: Dict[str, int],
        mapping_label_path: str | Path | None = None,
        fail_on_unmapped: bool = True,
        copy_images: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_coco_dir = Path(output_coco_dir)
        self.output_image_dir = Path(output_image_dir)
        self.target_class_names = target_class_names
        self.target_name_to_id = target_name_to_id
        self.mapping_label_path = Path(mapping_label_path) if mapping_label_path else None
        self.fail_on_unmapped = fail_on_unmapped
        self.copy_images = copy_images

        self.output_coco_dir.mkdir(parents=True, exist_ok=True)
        self.output_image_dir.mkdir(parents=True, exist_ok=True)

    def import_dataset(self) -> Dict[str, Any]:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy Roboflow dataset dir: {self.dataset_dir}")

        mapping_dict = None

        if self.mapping_label_path is not None:
            if not self.mapping_label_path.exists():
                raise FileNotFoundError(f"Không tìm thấy mapping file: {self.mapping_label_path}")
            mapping_dict = IOUtils.load_json(self.mapping_label_path)

        report = RoboflowCocoImportReport(
            dataset_dir=str(self.dataset_dir),
            output_coco_dir=str(self.output_coco_dir),
            output_image_dir=str(self.output_image_dir),
        )

        for source_split, target_split in self.SPLIT_NAME_MAP.items():
            split_dir = self.dataset_dir / source_split
            annotation_path = split_dir / "_annotations.coco.json"

            if not annotation_path.exists():
                report.skipped_splits.append(source_split)
                logger.warning("Bỏ qua split %s vì không có %s", source_split, annotation_path)
                continue

            logger.info("Import Roboflow COCO split: %s -> %s", source_split, target_split)

            dataset = IOUtils.load_json(annotation_path)

            dataset = self._normalize_image_file_names(
                dataset=dataset,
                split_dir=split_dir,
                target_split=target_split,
            )

            if mapping_dict is not None:
                mapper = TacoLabelMapper(
                    target_class_names=self.target_class_names,
                    target_name_to_id=self.target_name_to_id,
                    ignore_key="ignore",
                    fail_on_unmapped=self.fail_on_unmapped,
                )

                dataset, mapping_report = mapper.transform(
                    dataset=dataset,
                    mapping_dict=mapping_dict,
                )

                report.mapping_reports[target_split] = mapping_report.to_dict()
            else:
                dataset = self._normalize_direct_7class_categories(dataset)

            COCOValidator.validate(
                dataset=dataset,
                strict=True,
                check_bbox_inside_image=False,
            )

            output_annotation_path = self.output_coco_dir / f"{target_split}_annotations.json"

            IOUtils.save_coco_json(
                dest_path=output_annotation_path,
                dataset=dataset,
            )

            report.imported_splits.append(target_split)
            report.num_images[target_split] = len(dataset.get("images", []))
            report.num_annotations[target_split] = len(dataset.get("annotations", []))
            report.num_categories[target_split] = len(dataset.get("categories", []))

        report_path = self.output_coco_dir / "roboflow_coco_import_report.json"

        IOUtils.save_json(
            dest_path=report_path,
            data=report.to_dict(),
        )

        logger.info("Import Roboflow COCO hoàn tất. Report: %s", report_path)

        return report.to_dict()

    def _normalize_image_file_names(
        self,
        dataset: Dict[str, Any],
        split_dir: Path,
        target_split: str,
    ) -> Dict[str, Any]:
        dataset = deepcopy(dataset)

        split_output_image_dir = self.output_image_dir / target_split
        split_output_image_dir.mkdir(parents=True, exist_ok=True)

        for image in dataset.get("images", []):
            original_file_name = image.get("file_name")

            if not original_file_name:
                continue

            source_image_path = split_dir / original_file_name

            if not source_image_path.exists():
                source_image_path = split_dir / Path(original_file_name).name

            if not source_image_path.exists():
                logger.warning(
                    "Không tìm thấy ảnh Roboflow: split=%s, file_name=%s",
                    split_dir,
                    original_file_name,
                )
                continue

            safe_name = self._safe_file_name(
                split_name=target_split,
                file_name=original_file_name,
            )

            target_image_path = split_output_image_dir / safe_name

            if self.copy_images:
                shutil.copy2(source_image_path, target_image_path)
            else:
                target_image_path = source_image_path

            image["file_name"] = str(target_image_path.resolve())

        return dataset

    def _normalize_direct_7class_categories(
        self,
        dataset: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Dùng khi Roboflow dataset đã có sẵn đúng 7 class.

        Hàm này map category name theo lower-case về class order chuẩn.
        """
        dataset = deepcopy(dataset)

        old_categories = dataset.get("categories", [])

        old_id_to_new_id: Dict[int, int] = {}

        for category in old_categories:
            old_id = category.get("id")
            old_name = str(category.get("name", "")).strip().lower()

            if old_name not in self.target_name_to_id:
                raise KeyError(
                    f"Category '{category.get('name')}' không nằm trong 7 class chuẩn. "
                    "Hãy truyền --mapping-label để map nhãn."
                )

            old_id_to_new_id[old_id] = self.target_name_to_id[old_name]

        dataset["categories"] = [
            {
                "id": self.target_name_to_id[class_name],
                "name": class_name,
                "supercategory": "waste",
            }
            for class_name in self.target_class_names
        ]

        new_annotations = []

        for index, annotation in enumerate(dataset.get("annotations", []), start=1):
            old_category_id = annotation.get("category_id")

            if old_category_id not in old_id_to_new_id:
                continue

            new_annotation = deepcopy(annotation)
            new_annotation["id"] = index
            new_annotation["category_id"] = old_id_to_new_id[old_category_id]
            new_annotations.append(new_annotation)

        dataset["annotations"] = new_annotations

        return dataset

    @staticmethod
    def _safe_file_name(split_name: str, file_name: str) -> str:
        path = Path(file_name)
        parts = list(path.parts)
        base_name = "__".join(parts)
        return f"{split_name}__{base_name}"
