from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from torch.utils.data import DataLoader

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.crop_dataset import CropClassificationDataset
from waste_detection.data.samplers import SamplerFactory
from waste_detection.data.transforms import ClassifierTransformFactory
from waste_detection.models.efficientnet_classifier import EfficientNetB0Classifier
from waste_detection.tracking.experiment_registry import ExperimentRegistry
from waste_detection.tracking.wandb_logger import WandbLogger, to_serializable
from waste_detection.training.classifier_trainer import ClassifierTrainer
from waste_detection.utils.device import get_device
from waste_detection.utils.logger import LoggerSetup
from waste_detection.utils.seed import set_seed


logger = logging.getLogger("train_classifier")

DEFAULT_WANDB_PROJECT = "waste-detection-adn"
LEGACY_WANDB_PROJECT = "waste-detection-taco"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện EfficientNet-B0 crop classifier."
    )

    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/taco_7class.yaml",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/models/hybrid_yolov8n_effb0.yaml",
    )

    parser.add_argument(
        "--experiment-config",
        type=str,
        default="configs/experiments/run1_baseline.yaml",
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

    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override W&B project.",
    )

    return parser.parse_args()


def unique_tags(tags: Iterable[str]) -> List[str]:
    seen = set()
    result = []

    for tag in tags:
        tag = str(tag)

        if tag not in seen:
            result.append(tag)
            seen.add(tag)

    return result


def normalize_wandb_tracking_config(
    tracking_config: dict,
    args: argparse.Namespace,
) -> dict:
    """
    Apply CLI overrides and protect against the common case:
    --use-wandb is passed but YAML still has mode: disabled.
    """
    if args.use_wandb:
        tracking_config["use_wandb"] = True

    if args.wandb_project is not None:
        tracking_config["project"] = args.wandb_project

    if args.wandb_entity is not None:
        tracking_config["entity"] = args.wandb_entity

    if args.wandb_mode is not None:
        tracking_config["mode"] = args.wandb_mode

        if args.wandb_mode == "disabled":
            tracking_config["use_wandb"] = False

    if bool(tracking_config.get("use_wandb", False)):
        if tracking_config.get("mode") in {None, "", "disabled"}:
            tracking_config["mode"] = "online"

    project = str(tracking_config.get("project") or DEFAULT_WANDB_PROJECT)

    if project == LEGACY_WANDB_PROJECT:
        logger.warning(
            "W&B project đang là legacy '%s'. Tự động đổi sang '%s'.",
            LEGACY_WANDB_PROJECT,
            DEFAULT_WANDB_PROJECT,
        )
        project = DEFAULT_WANDB_PROJECT

    tracking_config["project"] = project

    return tracking_config


def main() -> None:
    args = parse_args()

    loader = ConfigLoader(project_root=args.project_root)
    loaded_config = loader.load_all(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        experiment_config_path=args.experiment_config,
        log_file_name="train_classifier.log",
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
    tracking_config = normalize_wandb_tracking_config(
        tracking_config=tracking_config,
        args=args,
    )

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    seed = int(experiment_config.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    device = get_device()

    logger.info("Device: %s", device)

    classifier_config = model_config.get("classifier", {})
    training_config = experiment_config.get("training", {})
    data_experiment_config = experiment_config.get("data", {})

    image_size = int(classifier_config.get("input_size", 224))
    augmentation_level = data_experiment_config.get("augmentation_level", "minimal")

    train_transform = ClassifierTransformFactory.build_train_transform(
        image_size=image_size,
        augmentation_level=augmentation_level,
    )

    eval_transform = ClassifierTransformFactory.build_eval_transform(
        image_size=image_size,
    )

    train_dataset = CropClassificationDataset(
        root_dir=data_config.paths.crops_7class_dir / "train",
        class_names=data_config.classes.names,
        transform=train_transform,
    )

    val_dataset = CropClassificationDataset(
        root_dir=data_config.paths.crops_7class_dir / "val",
        class_names=data_config.classes.names,
        transform=eval_transform,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"Train crop dataset rỗng: {data_config.paths.crops_7class_dir / 'train'}. "
            "Hãy chạy scripts/data/create_crop_dataset.py hoặc prepare_taco.py trước."
        )

    if len(val_dataset) == 0:
        raise RuntimeError(
            f"Val crop dataset rỗng: {data_config.paths.crops_7class_dir / 'val'}. "
            "Hãy chạy scripts/data/create_crop_dataset.py hoặc prepare_taco.py trước."
        )

    batch_size = int(training_config.get("batch_size", 16))
    num_workers = int(training_config.get("num_workers", 2))
    pin_memory = device.type == "cuda"

    sampler_config = experiment_config.get("sampler", {})

    train_sampler = SamplerFactory.build_classification_sampler(
        dataset=train_dataset,
        sampler_config=sampler_config,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    use_class_weighting = bool(
        experiment_config.get("loss", {}).get("class_weighting", False)
    )

    sampler_enabled = bool(sampler_config.get("enabled", False))

    if sampler_enabled and use_class_weighting:
        logger.warning(
            "Bạn đang bật cả weighted sampler và class weighting. "
            "Điều này có thể over-correct class imbalance. "
            "Khuyến nghị: thử từng kỹ thuật riêng trước."
        )

    class_weights = train_dataset.get_class_weights() if use_class_weighting else None

    model = EfficientNetB0Classifier(
        num_classes=data_config.classes.num_classes,
        dropout=float(classifier_config.get("dropout", 0.2)),
        pretrained=bool(classifier_config.get("pretrained", True)),
    )

    logger.info(
        "EfficientNet-B0 parameters: total=%d, trainable=%d",
        model.count_total_parameters(),
        model.count_trainable_parameters(),
    )

    model_config_name = model_config.get("model", {}).get("name", "hybrid_yolov8n_effb0")
    classifier_name = classifier_config.get("name", "efficientnet_b0")
    run_name = experiment_config.get("experiment", {}).get("name", "run_unknown")

    output_dir = (
        Path(loaded_config.system.project_root)
        / "outputs"
        / "checkpoints"
        / "efficientnet_b0"
        / run_name
    )

    wandb_group = tracking_config.get("group")

    if wandb_group in {None, "", "ablation", "ablation_study"}:
        wandb_group = "hybrid_classifier"

    wandb_tags = unique_tags(
        list(tracking_config.get("tags", []))
        + [
            "classifier",
            "train_classifier",
            classifier_name,
            model_config_name,
            run_name,
        ]
    )

    wandb_config = {
        "data": data_config,
        "model": model_config,
        "experiment": experiment_config,
        "system": loaded_config.system,
        "cli_args": vars(args),
        "train_class_counts": train_dataset.get_class_counts(),
        "val_class_counts": val_dataset.get_class_counts(),
    }

    wandb_logger = WandbLogger(
        enabled=bool(tracking_config.get("use_wandb", False)),
        project=str(tracking_config.get("project", DEFAULT_WANDB_PROJECT)),
        entity=tracking_config.get("entity"),
        run_name=f"{run_name}__{classifier_name}__train_classifier",
        config=to_serializable(wandb_config),
        group=wandb_group,
        tags=wandb_tags,
        mode=tracking_config.get("mode"),
        job_type="train_classifier",
        notes=experiment_config.get("experiment", {}).get("description"),
        output_dir=output_dir,
        log_code=bool(tracking_config.get("log_code", False)),
    )

    with wandb_logger:
        if bool(tracking_config.get("watch_model", False)):
            wandb_logger.watch_model(model)

        trainer = ClassifierTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=data_config.classes.names,
            experiment_config=experiment_config,
            output_dir=output_dir,
            device=device,
            class_weights=class_weights,
            wandb_run=wandb_logger.run,
        )

        history = trainer.train()

        best_checkpoint_path = output_dir / "best.pth"
        history_path = output_dir / "training_history.json"

        wandb_logger.log_summary(
            {
                "best_metric": trainer.best_metric,
                "best_checkpoint": str(best_checkpoint_path),
                "best_checkpoint_exists": best_checkpoint_path.exists(),
                "training_history_path": str(history_path),
                "training_history_exists": history_path.exists(),
                "num_epochs_ran": len(history),
                "model_config_name": model_config_name,
                "classifier_name": classifier_name,
                "run_name": run_name,
                "use_class_weighting": use_class_weighting,
                "sampler_enabled": sampler_enabled,
            }
        )

        if bool(tracking_config.get("log_model", True)):
            if best_checkpoint_path.exists():
                wandb_logger.log_artifact(
                    file_path=best_checkpoint_path,
                    artifact_name=f"{run_name}-{classifier_name}-classifier-best",
                    artifact_type="model",
                    aliases=["best", run_name],
                    metadata={
                        "task": "crop_classification",
                        "model": classifier_name,
                        "model_config": model_config_name,
                        "num_classes": data_config.classes.num_classes,
                        "class_names": data_config.classes.names,
                        "best_metric": trainer.best_metric,
                    },
                )
            else:
                logger.warning(
                    "Không tìm thấy classifier best checkpoint: %s",
                    best_checkpoint_path,
                )

        if history_path.exists():
            wandb_logger.log_artifact(
                file_path=history_path,
                artifact_name=f"{run_name}-{classifier_name}-training-history",
                artifact_type="metadata",
                aliases=[run_name],
                metadata={
                    "task": "crop_classification",
                    "run_name": run_name,
                    "classifier": classifier_name,
                },
            )

    registry = ExperimentRegistry(
        Path(loaded_config.system.project_root)
        / "outputs"
        / "metrics"
        / "experiment_registry.json"
    )

    registry.add(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task": "train_classifier",
            "model_name": classifier_name,
            "model_config_name": model_config_name,
            "run_name": run_name,
            "output_dir": str(output_dir),
            "best_checkpoint": str(output_dir / "best.pth"),
            "wandb_enabled": bool(tracking_config.get("use_wandb", False)),
            "wandb_mode": tracking_config.get("mode"),
            "wandb_project": tracking_config.get("project"),
            "wandb_entity": tracking_config.get("entity"),
            "wandb_group": wandb_group,
            "best_metric": getattr(trainer, "best_metric", None),
            "sampler_enabled": sampler_enabled,
            "use_class_weighting": use_class_weighting,
        }
    )

    logger.info("Huấn luyện classifier hoàn tất.")
    logger.info("Best checkpoint: %s", output_dir / "best.pth")


if __name__ == "__main__":
    main()
