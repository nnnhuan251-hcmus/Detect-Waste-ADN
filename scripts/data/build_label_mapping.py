import argparse
import logging

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.coco_validator import COCOValidator
from waste_detection.data.label_mapper import TacoLabelMapper
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("build_label_mapping")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map category gốc TACO sang 7 class đích."
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
        log_file_name="build_label_mapping.log",
    )

    data_config = loaded_config.data
    if data_config is None:
        raise RuntimeError("Không load được data config.")

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    logger.info("Bắt đầu build label mapping.")

    raw_dataset = CocoReader(data_config.paths.annotation_file).load(
        validate=True,
        strict=False,
    )

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

    logger.info("Build label mapping hoàn tất.")
    logger.info("Output annotation: %s", data_config.paths.annotations_7class_path)
    logger.info("Mapping report: %s", data_config.paths.mapping_report_path)


if __name__ == "__main__":
    main()
