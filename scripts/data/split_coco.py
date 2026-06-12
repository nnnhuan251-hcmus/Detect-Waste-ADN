import argparse
import logging

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.coco_splitter import CocoSplitter
from waste_detection.data.coco_validator import COCOValidator
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("split_coco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chia COCO 7-class dataset thành train/val/test."
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
        log_file_name="split_coco.log",
    )

    data_config = loaded_config.data

    if data_config is None:
        raise RuntimeError("Không load được data config.")

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    logger.info("Bắt đầu split COCO 7-class dataset.")

    dataset = CocoReader(data_config.paths.annotations_7class_path).load(
        validate=True,
        strict=True,
    )

    splitter = CocoSplitter(
        train_ratio=data_config.split.train_ratio,
        val_ratio=data_config.split.val_ratio,
        test_ratio=data_config.split.test_ratio,
        seed=data_config.split.seed,
        shuffle=data_config.split.shuffle,
        strategy=data_config.split.strategy,
    )

    splits, report = splitter.split(dataset)

    for split_name, split_dataset in splits.items():
        COCOValidator.validate(
            dataset=split_dataset,
            strict=True,
            check_bbox_inside_image=False,
        )

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
        report.to_dict(),
    )

    logger.info("Split COCO hoàn tất.")
    logger.info("Train: %s", data_config.paths.train_annotations_path)
    logger.info("Val: %s", data_config.paths.val_annotations_path)
    logger.info("Test: %s", data_config.paths.test_annotations_path)


if __name__ == "__main__":
    main()
