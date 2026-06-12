from __future__ import annotations

import argparse
import logging

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.pipelines.taco_preparation_pipeline import TacoPreparationPipeline
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("prepare_taco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare TACO dataset: raw COCO -> 7-class COCO -> "
            "train/val/test -> YOLO format -> crop dataset."
        )
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
        help="Project root. Nếu không truyền, ConfigLoader sẽ tự suy luận.",
    )

    parser.add_argument(
        "--no-copy-yolo-images",
        action="store_true",
        help=(
            "Nếu bật cờ này, YOLO labels vẫn được tạo nhưng ảnh không được copy "
            "sang thư mục YOLO output."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = ConfigLoader(project_root=args.project_root)

    loaded_config = loader.load_all(
        data_config_path=args.data_config,
        log_file_name="prepare_taco.log",
    )

    if loaded_config.data is None:
        raise RuntimeError("Không load được data config.")

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    logger.info("Bắt đầu prepare TACO dataset.")
    logger.info("Data config: %s", args.data_config)

    pipeline = TacoPreparationPipeline(
        data_config=loaded_config.data,
        system_config=loaded_config.system,
        copy_images_to_yolo=not args.no_copy_yolo_images,
    )

    pipeline.execute()

    logger.info("Prepare TACO dataset hoàn tất.")


if __name__ == "__main__":
    main()
