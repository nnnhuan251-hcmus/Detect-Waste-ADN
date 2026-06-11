from __future__ import annotations

import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.crop_dataset import CropClassificationDataset
from waste_detection.data.transforms import ClassifierTransformFactory
from waste_detection.models.efficientnet_classifier import EfficientNetB0Classifier
from waste_detection.training.classifier_trainer import ClassifierTrainer
from waste_detection.utils.device import get_device
from waste_detection.utils.logger import LoggerSetup
from waste_detection.utils.seed import set_seed


logger = logging.getLogger("train_classifier")


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

    return parser.parse_args()


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

    batch_size = int(training_config.get("batch_size", 16))
    num_workers = int(training_config.get("num_workers", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_weights = train_dataset.get_class_weights()

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

    model_name = model_config.get("model", {}).get("name", "hybrid_yolov8n_effb0")
    run_name = experiment_config.get("experiment", {}).get("name", "run_unknown")

    output_dir = (
        Path(loaded_config.system.project_root)
        / "outputs"
        / "checkpoints"
        / "efficientnet_b0"
        / run_name
    )

    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=data_config.classes.names,
        experiment_config=experiment_config,
        output_dir=output_dir,
        device=device,
        class_weights=class_weights,
        wandb_run=None,
    )

    trainer.train()

    logger.info("Huấn luyện classifier hoàn tất.")
    logger.info("Best checkpoint: %s", output_dir / "best.pth")


if __name__ == "__main__":
    main()
