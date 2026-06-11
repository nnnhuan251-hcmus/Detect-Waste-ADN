from __future__ import annotations

import argparse
import logging
from pathlib import Path

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.adapters.roboflow_coco_adapter import RoboflowCocoAdapter
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("import_roboflow_coco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Roboflow COCO train/valid/test dataset into project COCO split format."
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Thư mục Roboflow COCO chứa train/valid/test.",
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/taco_7class.yaml",
        help="Dùng config hiện tại để lấy class_names và output paths.",
    )

    parser.add_argument(
        "--mapping-label",
        type=str,
        default=None,
        help="Optional mapping_label.json nếu Roboflow chưa đúng 7 class.",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--fail-on-unmapped",
        action="store_true",
        help="Raise error nếu có category chưa được map.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = ConfigLoader(project_root=args.project_root)

    loaded_config = loader.load_all(
        data_config_path=args.data_config,
        log_file_name="import_roboflow_coco.log",
    )

    if loaded_config.data is None:
        raise RuntimeError("Không load được data config.")

    data_config = loaded_config.data

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    output_coco_dir = data_config.paths.coco_7class_dir
    output_image_dir = data_config.paths.processed_root / "roboflow_coco_images"

    mapping_label_path = args.mapping_label

    adapter = RoboflowCocoAdapter(
        dataset_dir=args.dataset_dir,
        output_coco_dir=output_coco_dir,
        output_image_dir=output_image_dir,
        target_class_names=data_config.classes.names,
        target_name_to_id=data_config.classes.name_to_id,
        mapping_label_path=mapping_label_path,
        fail_on_unmapped=args.fail_on_unmapped,
        copy_images=True,
    )

    report = adapter.import_dataset()

    logger.info("Roboflow COCO import hoàn tất.")
    logger.info("Report: %s", report)


if __name__ == "__main__":
    main()
