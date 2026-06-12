import argparse
import logging

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.crop_builder import CropDatasetBuilder
from waste_detection.data.image_resolver import ImageResolver
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("create_crop_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo crop dataset 7-class cho EfficientNet-B0."
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
        log_file_name="create_crop_dataset.log",
    )

    data_config = loaded_config.data

    if data_config is None:
        raise RuntimeError("Không load được data config.")

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    logger.info("Bắt đầu tạo crop dataset cho EfficientNet-B0.")

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

    crop_builder = CropDatasetBuilder(
        output_root=data_config.paths.crops_7class_dir,
        class_names=data_config.classes.names,
        margin_ratio=data_config.crop_dataset.include_context_margin,
        min_crop_size=data_config.crop_dataset.min_crop_size,
        save_format=data_config.crop_dataset.save_format,
        jpeg_quality=data_config.crop_dataset.jpeg_quality,
        
        # dù một split nào đó thiếu class hiếm, thư mục class vẫn được tạo đầy đủ theo 7 class chuẩn.
        write_metadata_csv=False,
    )

    all_reports = {}
    all_metadata_rows = []

    for split_name, split_dataset in split_datasets.items():
        report, metadata_rows = crop_builder.build_split(
            dataset=split_dataset,
            split_name=split_name,
            image_resolver=resolver,
        )

        all_reports[split_name] = report.to_dict()
        all_metadata_rows.extend(metadata_rows)

    if data_config.crop_dataset.write_metadata_csv:
        metadata_path = data_config.paths.crops_7class_dir / "crops_metadata.csv"

        crop_builder.write_metadata_csv(
            metadata_rows=all_metadata_rows,
            output_path=metadata_path,
        )

        for split_report in all_reports.values():
            split_report["metadata_path"] = str(metadata_path)

    IOUtils.save_json(
        data_config.paths.crops_7class_dir / "crop_build_report.json",
        all_reports,
    )

    logger.info("Tạo crop dataset hoàn tất.")


if __name__ == "__main__":
    main()
