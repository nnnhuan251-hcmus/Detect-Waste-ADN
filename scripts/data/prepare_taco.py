import argparse
import logging

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.coco_splitter import CocoSplitter
from waste_detection.data.coco_validator import COCOValidator
from waste_detection.data.crop_builder import CropDatasetBuilder
from waste_detection.data.image_resolver import ImageResolver
from waste_detection.data.label_mapper import TacoLabelMapper
from waste_detection.data.yolo_converter import YoloConverter
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("prepare_taco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy toàn bộ pipeline chuẩn bị dữ liệu TACO."
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/taco_7class.yaml",
        help="Đường dẫn tới data config YAML.",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root. Nếu không truyền, dùng PROJECT_ROOT env hoặc cwd.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = ConfigLoader(project_root=args.project_root)
    loaded_config = loader.load_all(
        data_config_path=args.data_config,
        log_file_name="prepare_taco.log",
    )

    data_config = loaded_config.data

    if data_config is None:
        raise RuntimeError("Không load được data config.")

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    logger.info("BẮT ĐẦU FULL DATA PIPELINE TACO.")

    resolver = ImageResolver(
        image_root=data_config.paths.image_root,
        batch_dirs=data_config.image_structure.batch_dirs,
        allowed_extensions=data_config.image_structure.allowed_extensions,
    )

    logger.info("Bước 1: Load và kiểm tra annotations.json gốc.")

    raw_dataset = CocoReader(data_config.paths.annotation_file).load(
        validate=False,
    )

    validation_report = COCOValidator.validate_with_report(
        dataset=raw_dataset,
        strict=False,
        check_bbox_inside_image=True,
    )

    image_report = resolver.build_report(raw_dataset)
    raw_summary = CocoReader.summarize(raw_dataset)

    IOUtils.save_json(
        data_config.paths.processed_root / "raw_check_summary.json",
        raw_summary,
    )

    IOUtils.save_json(
        data_config.paths.processed_root / "raw_validation_report.json",
        validation_report.to_dict(),
    )

    IOUtils.save_json(
        data_config.paths.processed_root / "image_resolution_report.json",
        image_report.to_dict(),
    )

    logger.info("Bước 2: Map nhãn TACO gốc sang 7 class.")

    mapping_dict = IOUtils.load_json(data_config.mapping.file)

    mapper = TacoLabelMapper(
        target_class_names=data_config.classes.names,
        target_name_to_id=data_config.classes.name_to_id,
        ignore_key=data_config.mapping.ignore_key,
        fail_on_unmapped=data_config.mapping.fail_on_unmapped,
    )

    processed_dataset, mapping_report = mapper.transform(
        dataset=raw_dataset,
        mapping_dict=mapping_dict,
    )

    COCOValidator.validate(
        dataset=processed_dataset,
        strict=True,
        check_bbox_inside_image=False,
    )

    IOUtils.save_coco_json(
        data_config.paths.annotations_7class_path,
        processed_dataset,
    )

    IOUtils.save_json(
        data_config.paths.categories_path,
        processed_dataset["categories"],
    )

    IOUtils.save_json(
        data_config.paths.mapping_report_path,
        mapping_report.to_dict(),
    )

    logger.info("Bước 3: Split train/val/test.")

    splitter = CocoSplitter(
        train_ratio=data_config.split.train_ratio,
        val_ratio=data_config.split.val_ratio,
        test_ratio=data_config.split.test_ratio,
        seed=data_config.split.seed,
        shuffle=data_config.split.shuffle,
    )

    splits, split_report = splitter.split(processed_dataset)

    IOUtils.save_coco_json(
        data_config.paths.train_annotations_path,
        splits["train"],
    )

    IOUtils.save_coco_json(
        data_config.paths.val_annotations_path,
        splits["val"],
    )

    IOUtils.save_coco_json(
        data_config.paths.test_annotations_path,
        splits["test"],
    )

    IOUtils.save_json(
        data_config.paths.coco_7class_dir / "split_report.json",
        split_report.to_dict(),
    )

    logger.info("Bước 4: Convert sang YOLO 7-class.")

    yolo_7class_converter = YoloConverter(
        output_root=data_config.paths.yolo_7class_dir,
        class_names=data_config.classes.names,
        binary=False,
        min_box_width=data_config.bbox.min_width,
        min_box_height=data_config.bbox.min_height,
    )

    yolo_reports = {"yolo_7class": {}, "yolo_binary_waste": {}}

    for split_name, split_dataset in splits.items():
        report = yolo_7class_converter.convert_split(
            dataset=split_dataset,
            split_name=split_name,
            image_resolver=resolver,
            copy_images=True,
        )
        yolo_reports["yolo_7class"][split_name] = report.to_dict()

    yolo_7class_converter.write_data_yaml()

    logger.info("Bước 5: Convert sang YOLO binary-waste cho hybrid detector.")

    yolo_binary_converter = YoloConverter(
        output_root=data_config.paths.yolo_binary_waste_dir,
        class_names=["waste"],
        binary=True,
        binary_class_name="waste",
        min_box_width=data_config.bbox.min_width,
        min_box_height=data_config.bbox.min_height,
    )

    for split_name, split_dataset in splits.items():
        report = yolo_binary_converter.convert_split(
            dataset=split_dataset,
            split_name=split_name,
            image_resolver=resolver,
            copy_images=True,
        )
        yolo_reports["yolo_binary_waste"][split_name] = report.to_dict()

    yolo_binary_converter.write_data_yaml()

    IOUtils.save_json(
        data_config.paths.processed_root / "yolo_conversion_report.json",
        yolo_reports,
    )

    logger.info("Bước 6: Tạo crop dataset cho EfficientNet-B0 classifier.")

    crop_builder = CropDatasetBuilder(
        output_root=data_config.paths.crops_7class_dir,
        margin_ratio=data_config.crop_dataset.include_context_margin,
        min_crop_size=data_config.crop_dataset.min_crop_size,
        save_format=data_config.crop_dataset.save_format,
        jpeg_quality=data_config.crop_dataset.jpeg_quality,
    )

    crop_reports = {}
    all_metadata_rows = []

    for split_name, split_dataset in splits.items():
        report, metadata_rows = crop_builder.build_split(
            dataset=split_dataset,
            split_name=split_name,
            image_resolver=resolver,
        )

        crop_reports[split_name] = report.to_dict()
        all_metadata_rows.extend(metadata_rows)

    if data_config.crop_dataset.write_metadata_csv:
        crop_builder.write_metadata_csv(all_metadata_rows)

    IOUtils.save_json(
        data_config.paths.crops_7class_dir / "crop_build_report.json",
        crop_reports,
    )

    logger.info("HOÀN THÀNH FULL DATA PIPELINE TACO.")


if __name__ == "__main__":
    main()
