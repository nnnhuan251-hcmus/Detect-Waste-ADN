from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from waste_detection.config.schemas import DataConfig, SystemConfig
from waste_detection.data.bbox_sanitizer import BBoxSanitizer
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.coco_splitter import CocoSplitter
from waste_detection.data.coco_validator import COCOValidator
from waste_detection.data.crop_builder import CropDatasetBuilder
from waste_detection.data.image_resolver import ImageResolver
from waste_detection.data.label_mapper import TacoLabelMapper
from waste_detection.data.yolo_converter import YoloConverter
from waste_detection.pipelines.base_pipeline import BasePipeline
from waste_detection.utils.io import IOUtils


logger = logging.getLogger("TacoPreparationPipeline")


class TacoPreparationPipeline(BasePipeline):
    """
    Pipeline tiền xử lý chính cho TACO dataset.

    Nhiệm vụ:
    1. Load annotations.json gốc.
    2. Validate COCO raw.
    3. Kiểm tra ảnh có resolve được trong batch_1 ... batch_15 hay không.
    4. Map nhãn TACO gốc về 7 class.
    5. Sanitize bbox.
    6. Save annotations_7class.json.
    7. Split train/val/test bằng strategy trong YAML.
    8. Convert sang YOLO 7-class.
    9. Convert sang YOLO binary-waste.
    10. Tạo crop dataset cho EfficientNet-B0.
    11. Save reports.

    Pipeline này chỉ điều phối.
    Logic nghiệp vụ nằm trong các class chuyên trách:
    - TacoLabelMapper
    - BBoxSanitizer
    - CocoSplitter
    - YoloConverter
    - CropDatasetBuilder
    """

    def __init__(
        self,
        data_config: DataConfig,
        system_config: SystemConfig,
        copy_images_to_yolo: bool = True,
    ) -> None:
        self.data_config = data_config
        self.system_config = system_config
        self.copy_images_to_yolo = bool(copy_images_to_yolo)

        self.paths = data_config.paths
        self.class_names = list(data_config.classes.names)

        self.image_resolver = ImageResolver(
            image_root=self.paths.image_root,
            batch_dirs=self.data_config.image_structure.batch_dirs,
            allowed_extensions=self.data_config.image_structure.allowed_extensions,
        )

    def execute(self) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("BẮT ĐẦU TACO PREPARATION PIPELINE")
        logger.info("Dataset: %s", self.data_config.dataset_name)
        logger.info("Annotation format: %s", self.data_config.annotation_format)
        logger.info("=" * 80)

        self._prepare_output_directories()

        raw_dataset = self._load_raw_dataset()
        raw_summary = CocoReader.summarize(raw_dataset)

        image_resolution_report = self._build_image_resolution_report(raw_dataset)

        mapped_dataset, mapping_report = self._map_labels(raw_dataset)
        sanitized_dataset, bbox_report = self._sanitize_bboxes(mapped_dataset)

        self._validate_processed_dataset(sanitized_dataset)

        self._save_processed_coco_dataset(
            dataset=sanitized_dataset,
            mapping_report=mapping_report,
            bbox_report=bbox_report,
        )

        splits, split_report = self._split_dataset(sanitized_dataset)
        self._save_split_datasets(splits=splits, split_report=split_report)

        yolo_7class_reports = self._convert_to_yolo(
            splits=splits,
            output_root=self.paths.yolo_7class_dir,
            binary=False,
        )

        yolo_binary_reports = self._convert_to_yolo(
            splits=splits,
            output_root=self.paths.yolo_binary_waste_dir,
            binary=True,
        )

        crop_reports = self._build_crop_dataset(splits=splits)

        final_report = {
            "dataset_name": self.data_config.dataset_name,
            "raw_summary": raw_summary,
            "image_resolution_report": image_resolution_report.to_dict(),
            "mapping_report": self._report_to_dict(mapping_report),
            "bbox_report": self._report_to_dict(bbox_report),
            "split_report": self._report_to_dict(split_report),
            "yolo_7class_reports": [
                self._report_to_dict(report) for report in yolo_7class_reports
            ],
            "yolo_binary_waste_reports": [
                self._report_to_dict(report) for report in yolo_binary_reports
            ],
            "crop_reports": [
                self._report_to_dict(report) for report in crop_reports
            ],
            "outputs": {
                "annotations_7class": str(self.paths.annotations_7class_path),
                "train_annotations": str(self.paths.train_annotations_path),
                "val_annotations": str(self.paths.val_annotations_path),
                "test_annotations": str(self.paths.test_annotations_path),
                "yolo_7class_dir": str(self.paths.yolo_7class_dir),
                "yolo_binary_waste_dir": str(self.paths.yolo_binary_waste_dir),
                "crops_7class_dir": str(self.paths.crops_7class_dir),
            },
        }

        final_report_path = self.paths.coco_7class_dir / "preparation_report.json"
        IOUtils.save_json(final_report_path, final_report)

        logger.info("=" * 80)
        logger.info("TACO PREPARATION PIPELINE HOÀN TẤT")
        logger.info("Final report: %s", final_report_path)
        logger.info("=" * 80)

        return final_report

    def _prepare_output_directories(self) -> None:
        output_dirs = [
            self.paths.interim_root,
            self.paths.processed_root,
            self.paths.coco_7class_dir,
            self.paths.yolo_7class_dir,
            self.paths.yolo_binary_waste_dir,
            self.paths.crops_7class_dir,
        ]

        for output_dir in output_dirs:
            IOUtils.ensure_dir(output_dir)

        logger.info("Đã chuẩn bị output directories.")

    def _load_raw_dataset(self) -> Dict[str, Any]:
        reader = CocoReader(annotation_path=self.paths.annotation_file)

        raw_dataset = reader.load(
            validate=True,
            strict=False,
        )

        logger.info(
            "Raw COCO loaded: %d images, %d annotations, %d categories.",
            len(raw_dataset.get("images", [])),
            len(raw_dataset.get("annotations", [])),
            len(raw_dataset.get("categories", [])),
        )

        return raw_dataset

    def _build_image_resolution_report(self, dataset: Dict[str, Any]):
        report = self.image_resolver.build_report(dataset)

        report_path = self.paths.coco_7class_dir / "image_resolution_report.json"
        IOUtils.save_json(report_path, report.to_dict())

        if report.missing_images > 0:
            logger.warning(
                "Có %d ảnh không resolve được. Xem report: %s",
                report.missing_images,
                report_path,
            )

        return report

    def _map_labels(
        self,
        raw_dataset: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any]:
        mapping_dict = IOUtils.load_json(self.data_config.mapping.file)

        mapper = TacoLabelMapper(
            target_class_names=self.class_names,
            target_name_to_id=self.data_config.classes.name_to_id,
            ignore_key=self.data_config.mapping.ignore_key,
            fail_on_unmapped=self.data_config.mapping.fail_on_unmapped,
        )

        mapped_dataset, mapping_report = mapper.transform(
            dataset=raw_dataset,
            mapping_dict=mapping_dict,
        )

        logger.info(
            "Map label xong: %d images, %d annotations, %d categories.",
            len(mapped_dataset.get("images", [])),
            len(mapped_dataset.get("annotations", [])),
            len(mapped_dataset.get("categories", [])),
        )

        return mapped_dataset, mapping_report

    def _sanitize_bboxes(
        self,
        dataset: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any]:
        sanitizer = BBoxSanitizer(
            min_width=self.data_config.bbox.min_width,
            min_height=self.data_config.bbox.min_height,
            clamp_to_image=self.data_config.bbox.clamp_to_image,
            drop_invalid_bbox=self.data_config.bbox.drop_invalid_bbox,
        )

        sanitized_dataset, bbox_report = sanitizer.sanitize(dataset)

        logger.info(
            "Sanitize bbox xong: %d images, %d annotations.",
            len(sanitized_dataset.get("images", [])),
            len(sanitized_dataset.get("annotations", [])),
        )

        return sanitized_dataset, bbox_report

    def _validate_processed_dataset(self, dataset: Dict[str, Any]) -> None:
        COCOValidator.validate(
            dataset=dataset,
            strict=True,
            check_bbox_inside_image=True,
        )

        logger.info("Processed COCO dataset hợp lệ.")

    def _save_processed_coco_dataset(
        self,
        dataset: Dict[str, Any],
        mapping_report: Any,
        bbox_report: Any,
    ) -> None:
        IOUtils.save_coco_json(
            dest_path=self.paths.annotations_7class_path,
            dataset=dataset,
        )

        IOUtils.save_json(
            dest_path=self.paths.categories_path,
            data=dataset.get("categories", []),
        )

        IOUtils.save_json(
            dest_path=self.paths.mapping_report_path,
            data=self._report_to_dict(mapping_report),
        )

        bbox_report_path = self.paths.coco_7class_dir / "bbox_sanitizer_report.json"
        IOUtils.save_json(
            dest_path=bbox_report_path,
            data=self._report_to_dict(bbox_report),
        )

        logger.info("Đã lưu processed COCO dataset và reports.")

    def _split_dataset(
        self,
        dataset: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, Any]], Any]:
        splitter = CocoSplitter(
            train_ratio=self.data_config.split.train_ratio,
            val_ratio=self.data_config.split.val_ratio,
            test_ratio=self.data_config.split.test_ratio,
            seed=self.data_config.split.seed,
            shuffle=self.data_config.split.shuffle,
            strategy=self.data_config.split.strategy,
        )

        splits, split_report = splitter.split(dataset)

        logger.info(
            "Split xong: train=%d, val=%d, test=%d images.",
            len(splits["train"].get("images", [])),
            len(splits["val"].get("images", [])),
            len(splits["test"].get("images", [])),
        )

        return splits, split_report

    def _save_split_datasets(
        self,
        splits: Dict[str, Dict[str, Any]],
        split_report: Any,
    ) -> None:
        IOUtils.save_coco_json(
            dest_path=self.paths.train_annotations_path,
            dataset=splits["train"],
        )

        IOUtils.save_coco_json(
            dest_path=self.paths.val_annotations_path,
            dataset=splits["val"],
        )

        IOUtils.save_coco_json(
            dest_path=self.paths.test_annotations_path,
            dataset=splits["test"],
        )

        IOUtils.save_json(
            dest_path=self.paths.split_report_path,
            data=self._report_to_dict(split_report),
        )

        logger.info("Đã lưu train/val/test annotations và split report.")

    def _convert_to_yolo(
        self,
        splits: Dict[str, Dict[str, Any]],
        output_root: str | Path,
        binary: bool,
    ) -> List[Any]:
        output_root = Path(output_root)

        converter = YoloConverter(
            output_root=output_root,
            class_names=self.class_names,
            binary=binary,
        )

        reports = []

        for split_name in ["train", "val", "test"]:
            report = converter.convert_split(
                dataset=splits[split_name],
                split_name=split_name,
                image_resolver=self.image_resolver,
                copy_images=self.copy_images_to_yolo,
            )
            reports.append(report)

        converter.write_data_yaml()

        logger.info(
            "Convert YOLO xong: output_root=%s | binary=%s",
            output_root,
            binary,
        )

        return reports

    def _build_crop_dataset(
        self,
        splits: Dict[str, Dict[str, Any]],
    ) -> List[Any]:
        crop_config = self.data_config.crop_dataset

        builder = CropDatasetBuilder(
            output_root=self.paths.crops_7class_dir,
            class_names=self.class_names,
            margin_ratio=crop_config.include_context_margin,
            min_crop_size=crop_config.min_crop_size,
            save_format=crop_config.save_format,
            jpeg_quality=crop_config.jpeg_quality,
            write_metadata_csv=False,
        )

        crop_reports = []
        all_metadata_rows: List[Dict[str, Any]] = []

        for split_name in ["train", "val", "test"]:
            report, metadata_rows = builder.build_split(
                dataset=splits[split_name],
                split_name=split_name,
                image_resolver=self.image_resolver,
            )

            crop_reports.append(report)
            all_metadata_rows.extend(metadata_rows)

        if crop_config.write_metadata_csv:
            metadata_path = self.paths.crops_7class_dir / "crops_metadata.csv"
            builder.write_metadata_csv(
                metadata_rows=all_metadata_rows,
                output_path=metadata_path,
            )

            for report in crop_reports:
                report.metadata_path = str(metadata_path)

        crop_report_path = self.paths.crops_7class_dir / "crop_build_report.json"

        IOUtils.save_json(
            dest_path=crop_report_path,
            data=[self._report_to_dict(report) for report in crop_reports],
        )

        logger.info(
            "Build crop dataset xong: output_root=%s",
            self.paths.crops_7class_dir,
        )

        return crop_reports

    @staticmethod
    def _report_to_dict(report: Any) -> Any:
        if report is None:
            return None

        if isinstance(report, dict):
            return report

        if hasattr(report, "to_dict"):
            return report.to_dict()

        if hasattr(report, "__dict__"):
            return dict(report.__dict__)

        return str(report)
