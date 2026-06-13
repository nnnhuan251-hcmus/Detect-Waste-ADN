from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.tracking.experiment_registry import ExperimentRegistry
from waste_detection.tracking.wandb_logger import WandbLogger, to_serializable
from waste_detection.training.detector_trainer import DetectorTrainer
from waste_detection.utils.logger import LoggerSetup
from waste_detection.utils.seed import set_seed


logger = logging.getLogger("train_detector")

def infer_detector_wandb_group(model_name: str) -> str:
    if model_name == "hybrid_yolov8n_effb0":
        return "hybrid_detector"

    if model_name == "yolov8s":
        return "yolov8s_detector"

    if model_name == "rtdetr_l":
        return "rtdetr_l_detector"

    return "detector"
    
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
        help=(
            "Ví dụ: configs/models/yolov8s.yaml hoặc "
            "configs/models/hybrid_yolov8n_effb0.yaml"
        ),
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

    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Bật W&B tracking, override tracking.use_wandb trong YAML.",
    )

    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Override W&B mode.",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/team/user. Có thể bỏ trống.",
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

    tracking_config = experiment_config.setdefault("tracking", {})

    if args.use_wandb:
        tracking_config["use_wandb"] = True

        if tracking_config.get("mode") in {None, "disabled"}:
            tracking_config["mode"] = "online"

    if args.wandb_mode is not None:
        tracking_config["mode"] = args.wandb_mode

        if args.wandb_mode == "disabled":
            tracking_config["use_wandb"] = False

    if args.wandb_entity is not None:
        tracking_config["entity"] = args.wandb_entity

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

    model_name = model_config.get("model", {}).get(
        "name",
        detector_config.get("name", "detector"),
    )
    run_name = experiment_config.get("experiment", {}).get("name", "run_unknown")

    output_root = Path(loaded_config.system.project_root) / "outputs" / "checkpoints"
    output_dir = output_root / model_name / run_name

    wandb_group = tracking_config.get("group")
    if wandb_group in {None, "", "ablation", "ablation_study"}:
        wandb_group = infer_detector_wandb_group(model_name)

    wandb_config = {
        "data": data_config,
        "model": model_config,
        "experiment": experiment_config,
        "system": loaded_config.system,
        "cli_args": vars(args),
    }

    wandb_logger = WandbLogger(
        enabled=bool(tracking_config.get("use_wandb", False)),
        project=str(tracking_config.get("project", "waste-detection-adn")),
        entity=tracking_config.get("entity"),
        run_name=f"{run_name}__{model_name}__detector",
        config=to_serializable(wandb_config),
        group=wandb_group,
        tags=tracking_config.get("tags", []),
        mode=tracking_config.get("mode"),
        job_type="train_detector",
        notes=experiment_config.get("experiment", {}).get("description"),
        output_dir=output_dir,
        log_code=bool(tracking_config.get("log_code", False)),
    )

    with wandb_logger:
        trainer = DetectorTrainer(
            model_config=model_config,
            experiment_config=experiment_config,
            data_yaml_path=data_yaml_path,
            output_root=output_root,
        )

        summary = trainer.train()

        best_weight_path = output_dir / "weights" / "best.pt"
        last_weight_path = output_dir / "weights" / "last.pt"
        train_summary_path = output_dir / "train_summary.json"
        results_csv_path = output_dir / "results.csv"

        wandb_logger.log_summary(
            {
                "model_name": model_name,
                "run_name": run_name,
                "detector_mode": detector_mode,
                "best_weight_path": str(best_weight_path),
                "last_weight_path": str(last_weight_path),
                "output_dir": str(output_dir),
            }
        )

        if results_csv_path.exists():
            wandb_logger.log_csv_history(
                csv_path=results_csv_path,
                prefix="detector/",
                step_column="epoch",
            )

        if bool(tracking_config.get("log_model", True)):
            if best_weight_path.exists():
                wandb_logger.log_artifact(
                    file_path=best_weight_path,
                    artifact_name=f"{run_name}-{model_name}-detector-best",
                    artifact_type="model",
                    aliases=["best", run_name],
                    metadata={
                        "task": "object_detection",
                        "model": model_name,
                        "detector_mode": detector_mode,
                        "data_yaml_path": str(data_yaml_path),
                    },
                )
            else:
                logger.warning("Không tìm thấy best detector checkpoint: %s", best_weight_path)

            if last_weight_path.exists():
                wandb_logger.log_artifact(
                    file_path=last_weight_path,
                    artifact_name=f"{run_name}-{model_name}-detector-last",
                    artifact_type="model",
                    aliases=["last", run_name],
                    metadata={
                        "task": "object_detection",
                        "model": model_name,
                        "detector_mode": detector_mode,
                        "data_yaml_path": str(data_yaml_path),
                    },
                )
            else:
                logger.warning("Không tìm thấy last detector checkpoint: %s", last_weight_path)

        if train_summary_path.exists():
            wandb_logger.log_artifact(
                file_path=train_summary_path,
                artifact_name=f"{run_name}-{model_name}-detector-summary",
                artifact_type="metadata",
                aliases=[run_name],
                metadata=summary,
            )
        else:
            logger.warning("Không tìm thấy detector train summary: %s", train_summary_path)

    registry = ExperimentRegistry(
        Path(loaded_config.system.project_root)
        / "outputs"
        / "metrics"
        / "experiment_registry.json"
    )

    registry.add(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task": "train_detector",
            "model_name": model_name,
            "run_name": run_name,
            "detector_mode": detector_mode,
            "data_yaml_path": str(data_yaml_path),
            "output_dir": str(output_dir),
            "best_checkpoint": str(output_dir / "weights" / "best.pt"),
            "wandb_enabled": bool(tracking_config.get("use_wandb", False)),
            "wandb_mode": tracking_config.get("mode"),
        }
    )

    logger.info("Train detector hoàn tất.")
    logger.info("Best checkpoint: %s", output_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
