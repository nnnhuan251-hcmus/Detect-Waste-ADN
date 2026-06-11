from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup


logger = logging.getLogger("run_ablation")


DEFAULT_EXPERIMENTS = [
    "configs/experiments/run1_baseline.yaml",
    "configs/experiments/run2_strong_aug.yaml",
    "configs/experiments/run3_freeze_cosine_warmup.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or execute ablation training plan."
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/taco_7class.yaml",
    )

    parser.add_argument(
        "--model-configs",
        nargs="+",
        default=[
            "configs/models/hybrid_yolov8n_effb0.yaml",
            "configs/models/yolov8s.yaml",
            "configs/models/rtdetr_l.yaml",
        ],
    )

    parser.add_argument(
        "--experiment-configs",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["detector", "classifier"],
        choices=["detector", "classifier"],
        help=(
            "detector: train detector cho từng model-config. "
            "classifier: train EfficientNet-B0 cho hybrid model."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Nếu bật, chạy thật các command. Nếu không, chỉ lưu plan.",
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
        log_file_name="run_ablation.log",
    )

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    project_root = Path(loaded_config.system.project_root)

    commands = build_ablation_commands(
        project_root=project_root,
        data_config=args.data_config,
        model_configs=args.model_configs,
        experiment_configs=args.experiment_configs,
        tasks=args.tasks,
    )

    output_path = (
        project_root
        / "outputs"
        / "metrics"
        / "ablation_plan.json"
    )

    IOUtils.save_json(
        dest_path=output_path,
        data={
            "execute": args.execute,
            "num_commands": len(commands),
            "commands": commands,
        },
    )

    logger.info("Đã lưu ablation plan tại: %s", output_path)

    for command in commands:
        logger.info("PLAN: %s", " ".join(command))

    if not args.execute:
        logger.info("Chưa chạy training vì bạn chưa bật --execute.")
        return

    for command in commands:
        logger.info("RUN: %s", " ".join(command))

        subprocess.run(
            command,
            cwd=project_root,
            check=True,
        )

    logger.info("Hoàn tất ablation run.")


def build_ablation_commands(
    project_root: Path,
    data_config: str,
    model_configs: List[str],
    experiment_configs: List[str],
    tasks: List[str],
) -> List[List[str]]:
    commands: List[List[str]] = []

    for model_config_path in model_configs:
        model_config = _load_yaml(project_root / model_config_path)
        model_type = model_config.get("model", {}).get("type", "")
        model_name = model_config.get("model", {}).get("name", Path(model_config_path).stem)

        for experiment_config_path in experiment_configs:
            if "detector" in tasks:
                commands.append(
                    [
                        sys.executable,
                        "scripts/train/train_detector.py",
                        "--data-config",
                        data_config,
                        "--model-config",
                        model_config_path,
                        "--experiment-config",
                        experiment_config_path,
                    ]
                )

            if "classifier" in tasks and model_type == "hybrid":
                commands.append(
                    [
                        sys.executable,
                        "scripts/train/train_classifier.py",
                        "--data-config",
                        data_config,
                        "--model-config",
                        model_config_path,
                        "--experiment-config",
                        experiment_config_path,
                    ]
                )

            logger.info(
                "Prepared commands for model=%s, experiment=%s",
                model_name,
                experiment_config_path,
            )

    return commands


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy YAML config: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data or {}


if __name__ == "__main__":
    main()
