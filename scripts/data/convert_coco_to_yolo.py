import argparse
import logging

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.image_resolver import ImageResolver
from waste_detection.data.yolo_converter import YoloConverter
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("convert_coco_to_yolo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO train/val/test sang YOLO format."
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
        "--skip-seven-class",
        action="store_true",
        help="Bỏ qua tạo yolo_7class.",
    )

    parser.add_argument(
        "--skip-binary",
        action="store_true",
        help="Bỏ qua tạo yolo_binary_waste.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = ConfigLoader(project_root=args.project_root)
    loaded_config = loader.load_all(
        data_config_path=args.data_config,
        log_file_name="convert_coco_to_yolo.log",
    )

    data_config = loaded_config.data

    if data_config is None:
        raise RuntimeError("Không load được data config.")

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    logger.info("Bắt đầu convert COCO sang YOLO.")

    split_paths = {
        "train": data_config.paths.train_annotations_path,
        "val": data_config.paths.val_annotations_path,
        "test": data_config.paths.test_annotations_path,
    }

    split_datasets = {
        split_name: CocoReader(split_path).load(validate=True, strict=True)
        for split_name, split_path in split_paths.items()
    }

    resolver = ImageResolver(
        image_root=data_config.paths.image_root,
        batch_dirs=data_config.image_structure.batch_dirs,
        allowed_extensions=data_config.image_structure.allowed_extensions,
    )

    all_reports = {}

    if not args.skip_seven_class:
        seven_class_converter = YoloConverter(
            output_root=data_config.paths.yolo_7class_dir,
            class_names=data_config.classes.names,
            binary=False,
            min_box_width=data_config.bbox.min_width,
            min_box_height=data_config.bbox.min_height,
        )

        all_reports["yolo_7class"] = {}

        for split_name, split_dataset in split_datasets.items():
            report = seven_class_converter.convert_split(
                dataset=split_dataset,
                split_name=split_name,
                image_resolver=resolver,
                copy_images=True,
            )
            all_reports["yolo_7class"][split_name] = report.to_dict()

        seven_class_converter.write_data_yaml()

    if not args.skip_binary:
        binary_converter = YoloConverter(
            output_root=data_config.paths.yolo_binary_waste_dir,
            class_names=["waste"],
            binary=True,
            binary_class_name="waste",
            min_box_width=data_config.bbox.min_width,
            min_box_height=data_config.bbox.min_height,
        )

        all_reports["yolo_binary_waste"] = {}

        for split_name, split_dataset in split_datasets.items():
            report = binary_converter.convert_split(
                dataset=split_dataset,
                split_name=split_name,
                image_resolver=resolver,
                copy_images=True,
            )
            all_reports["yolo_binary_waste"][split_name] = report.to_dict()

        binary_converter.write_data_yaml()

    IOUtils.save_json(
        data_config.paths.processed_root / "yolo_conversion_report.json",
        all_reports,
    )

    logger.info("Convert COCO sang YOLO hoàn tất.")


if __name__ == "__main__":
    main()
