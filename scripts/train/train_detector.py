from __future__ import annotations

import argparse
import logging
from pathlib import Path

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.training.detector_trainer import DetectorTrainer
from waste_detection.utils.logger import LoggerSetup
from waste_detection.utils.seed import set_seed


logger = logging.getLogger("train_detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train detector using Ultralytics backend."
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/taco_7class.yaml",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Ví dụ: configs/models/yolov8s.yaml hoặc configs/models/hybrid_yolov8n_effb0.yaml",
    )

    parser.add_argument(
        "--experiment-config",
        type=str,
        required=True,
        help="Ví dụ: configs/experiments/run1_baseline.yaml",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = ConfigLoader(project_root=args.project_root)
    loaded_config = loader.load_all(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        experiment_config_path=args.experiment_config,
        log_file_name="train_detector.log",
    )

    if loaded_config.data is None:
        raise RuntimeError("Không load được data config.")

    if loaded_config.model is None:
        raise RuntimeError("Không load được model config.")

    if loaded_config.experiment is None:
        raise RuntimeError("Không load được experiment config.")

    data_config = loaded_config.data
    model_config = loaded_config.model
    experiment_config = loaded_config.experiment

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    seed = int(experiment_config.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    detector_config = model_config.get("detector", {})
    detector_mode = detector_config.get("detector_mode", "seven_class")

    if detector_mode == "binary_waste":
        data_yaml_path = data_config.paths.yolo_binary_waste_dir / "data.yaml"
    elif detector_mode == "seven_class":
        data_yaml_path = data_config.paths.yolo_7class_dir / "data.yaml"
    else:
        raise ValueError(f"detector_mode không hợp lệ: {detector_mode}")

    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy YOLO data.yaml: {data_yaml_path}. "
            "Hãy chạy scripts/data/convert_coco_to_yolo.py hoặc prepare_taco.py trước."
        )

    output_root = Path(loaded_config.system.project_root) / "outputs" / "checkpoints"

    trainer = DetectorTrainer(
        model_config=model_config,
        experiment_config=experiment_config,
        data_yaml_path=data_yaml_path,
        output_root=output_root,
    )

    trainer.train()

    logger.info("Train detector hoàn tất.")


if __name__ == "__main__":
    main()
