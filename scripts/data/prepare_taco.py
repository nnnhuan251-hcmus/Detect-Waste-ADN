import argparse
import logging
from pathlib import Path

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.coco_validator import COCOValidator
from waste_detection.data.image_resolver import ImageResolver
from waste_detection.data.label_mapper import TacoLabelMapper
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("prepare_taco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy pipeline chuẩn bị dữ liệu TACO theo scaffold hiện tại."
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

    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Bỏ qua bước check COCO và resolve image.",
    )

    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="Bỏ qua bước build mapping 7 class.",
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

    logger.info("BẮT ĐẦU PIPELINE CHUẨN BỊ DỮ LIỆU TACO.")

    raw_dataset = CocoReader(data_config.paths.annotation_file).load(
        validate=False,
    )

    output_metrics_dir = Path(loader.project_root) / "outputs" / "metrics"
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_check:
        logger.info("Bước 1: Kiểm tra COCO annotation và resolve ảnh.")

        validation_report = COCOValidator.validate_with_report(
            dataset=raw_dataset,
            strict=False,
            check_bbox_inside_image=True,
        )

        resolver = ImageResolver(
            image_root=data_config.paths.image_root,
            batch_dirs=data_config.image_structure.batch_dirs,
            allowed_extensions=data_config.image_structure.allowed_extensions,
        )

        image_report = resolver.build_report(raw_dataset)
        summary = CocoReader.summarize(raw_dataset)

        IOUtils.save_json(
            output_metrics_dir / "check_coco_summary.json",
            summary,
        )

        IOUtils.save_json(
            output_metrics_dir / "check_coco_validation_report.json",
            validation_report.to_dict(),
        )

        IOUtils.save_json(
            output_metrics_dir / "check_coco_image_resolution_report.json",
            image_report.to_dict(),
        )

    if not args.skip_mapping:
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
            dest_path=data_config.paths.annotations_7class_path,
            dataset=processed_dataset,
        )

        IOUtils.save_json(
            dest_path=data_config.paths.categories_path,
            data=processed_dataset["categories"],
        )

        IOUtils.save_json(
            dest_path=data_config.paths.mapping_report_path,
            data=mapping_report.to_dict(),
        )

    logger.info("HOÀN THÀNH PIPELINE CHUẨN BỊ DỮ LIỆU TACO GIAI ĐOẠN HIỆN TẠI.")
    logger.info(
        "Lưu ý: split_coco, convert_coco_to_yolo và create_crop_dataset "
        "sẽ được nối vào prepare_taco.py sau khi các module tương ứng được hoàn thiện."
    )


if __name__ == "__main__":
    main()
