import argparse
import logging
from pathlib import Path

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.coco_validator import COCOValidator
from waste_detection.data.image_resolver import ImageResolver
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("check_coco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm tra COCO annotations.json chính thức của TACO."
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
        "--strict",
        action="store_true",
        help="Nếu bật, validator sẽ raise error khi phát hiện lỗi.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = ConfigLoader(project_root=args.project_root)
    loaded_config = loader.load_all(
        data_config_path=args.data_config,
        log_file_name="check_coco.log",
    )

    data_config = loaded_config.data
    if data_config is None:
        raise RuntimeError("Không load được data config.")

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    logger.info("Bắt đầu kiểm tra COCO dataset.")

    reader = CocoReader(data_config.paths.annotation_file)
    dataset = reader.load(validate=False)

    validation_report = COCOValidator.validate_with_report(
        dataset=dataset,
        strict=args.strict,
        check_bbox_inside_image=True,
    )

    resolver = ImageResolver(
        image_root=data_config.paths.image_root,
        batch_dirs=data_config.image_structure.batch_dirs,
        allowed_extensions=data_config.image_structure.allowed_extensions,
    )

    image_report = resolver.build_report(dataset)

    summary = CocoReader.summarize(dataset)

    output_dir = Path(loader.project_root) / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    IOUtils.save_json(
        output_dir / "check_coco_summary.json",
        summary,
    )

    IOUtils.save_json(
        output_dir / "check_coco_validation_report.json",
        validation_report.to_dict(),
    )

    IOUtils.save_json(
        output_dir / "check_coco_image_resolution_report.json",
        image_report.to_dict(),
    )

    logger.info("Kiểm tra COCO dataset hoàn tất.")
    logger.info("Summary lưu tại: %s", output_dir / "check_coco_summary.json")
    logger.info(
        "Validation report lưu tại: %s",
        output_dir / "check_coco_validation_report.json",
    )
    logger.info(
        "Image resolution report lưu tại: %s",
        output_dir / "check_coco_image_resolution_report.json",
    )


if __name__ == "__main__":
    main()
