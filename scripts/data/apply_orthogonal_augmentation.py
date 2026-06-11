from __future__ import annotations

import argparse
import logging
import shutil

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.coco_validator import COCOValidator
from waste_detection.data.image_resolver import ImageResolver
from waste_detection.data.orthogonal_augmenter import OrthogonalAugmenter
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("apply_orthogonal_augmentation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply offline orthogonal rotation augmentation to train COCO split."
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/taco_7class.yaml",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--angles",
        nargs="+",
        type=int,
        default=[90, 180],
        help="Angles trực giao. Hỗ trợ: 90 180 270.",
    )

    parser.add_argument(
        "--overwrite-train",
        action="store_true",
        help="Nếu bật, ghi đè train_annotations.json sau khi tạo backup.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = ConfigLoader(project_root=args.project_root)
    loaded_config = loader.load_all(
        data_config_path=args.data_config,
        log_file_name="apply_orthogonal_augmentation.log",
    )

    if loaded_config.data is None:
        raise RuntimeError("Không load được data config.")

    data_config = loaded_config.data

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    train_path = data_config.paths.train_annotations_path
    output_path = data_config.paths.coco_7class_dir / "train_annotations_orthogonal_aug.json"
    output_image_dir = data_config.paths.coco_7class_dir / "orthogonal_augmented_images" / "train"

    logger.info("Load train annotations: %s", train_path)

    train_dataset = CocoReader(train_path).load(validate=True, strict=True)

    resolver = ImageResolver(
        image_root=data_config.paths.image_root,
        batch_dirs=data_config.image_structure.batch_dirs,
        allowed_extensions=data_config.image_structure.allowed_extensions,
    )

    augmenter = OrthogonalAugmenter(
        output_image_dir=output_image_dir,
        angles=args.angles,
        use_absolute_file_names=True,
        jpeg_quality=data_config.crop_dataset.jpeg_quality,
    )

    augmented_dataset, report = augmenter.augment_dataset(
        dataset=train_dataset,
        image_resolver=resolver,
    )

    COCOValidator.validate(
        dataset=augmented_dataset,
        strict=True,
        check_bbox_inside_image=False,
    )

    IOUtils.save_coco_json(
        dest_path=output_path,
        dataset=augmented_dataset,
    )

    IOUtils.save_json(
        dest_path=data_config.paths.coco_7class_dir / "orthogonal_augmentation_report.json",
        data=report.to_dict(),
    )

    logger.info("Saved augmented train annotations: %s", output_path)

    if args.overwrite_train:
        backup_path = train_path.with_suffix(".before_orthogonal_aug.json")
        shutil.copy2(train_path, backup_path)
        shutil.copy2(output_path, train_path)

        logger.info("Backup train annotations: %s", backup_path)
        logger.info("Overwrote train annotations: %s", train_path)


if __name__ == "__main__":
    main()
