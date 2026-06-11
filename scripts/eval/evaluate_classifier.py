from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.crop_dataset import CropClassificationDataset
from waste_detection.data.transforms import ClassifierTransformFactory
from waste_detection.evaluation.classification_metrics import ClassificationMetrics
from waste_detection.models.efficientnet_classifier import EfficientNetB0Classifier
from waste_detection.utils.device import get_device
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup

from waste_detection.evaluation.error_analysis import ClassificationErrorAnalyzer
from waste_detection.visualization.visualize_errors import ErrorVisualizer

logger = logging.getLogger("evaluate_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá EfficientNet-B0 crop classifier."
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
        "--weights",
        type=str,
        required=True,
        help="Đường dẫn tới checkpoint best.pth.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
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
        log_file_name="evaluate_classifier.log",
    )

    if loaded_config.data is None:
        raise RuntimeError("Không load được data config.")

    if loaded_config.model is None:
        raise RuntimeError("Không load được model config.")

    data_config = loaded_config.data
    model_config = loaded_config.model

    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    device = get_device()

    classifier_config = model_config.get("classifier", {})
    image_size = int(classifier_config.get("input_size", 224))

    eval_transform = ClassifierTransformFactory.build_eval_transform(
        image_size=image_size,
    )

    dataset = CropClassificationDataset(
        root_dir=data_config.paths.crops_7class_dir / args.split,
        class_names=data_config.classes.names,
        transform=eval_transform,
        return_path=True,
    )

    batch_size = int(classifier_config.get("batch_size", 32))

    loader_eval = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model, checkpoint_class_names = EfficientNetB0Classifier.load_from_checkpoint(
        args.weights,
        map_location=device,
    )

    if checkpoint_class_names and checkpoint_class_names != data_config.classes.names:
        logger.warning(
            "Class names trong checkpoint khác data config. checkpoint=%s, config=%s",
            checkpoint_class_names,
            data_config.classes.names,
        )

    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    prediction_rows = []

    with torch.no_grad():
        for batch in tqdm(loader_eval, desc=f"Evaluate classifier [{args.split}]"):
            inputs, labels, paths = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

            for path, label, pred, prob_vector in zip(
                paths,
                labels.cpu().tolist(),
                predictions.cpu().tolist(),
                probabilities.cpu().tolist(),
            ):
                prediction_rows.append(
                    {
                        "image_path": path,
                        "true_id": label,
                        "true_class": data_config.classes.names[label],
                        "pred_id": pred,
                        "pred_class": data_config.classes.names[pred],
                        "confidence": float(max(prob_vector)),
                        "is_correct": bool(label == pred),
                    }
                )

    metrics = ClassificationMetrics.compute(
        y_true=y_true,
        y_pred=y_pred,
        class_names=data_config.classes.names,
    )

    output_dir = (
        Path(loaded_config.system.project_root)
        / "outputs"
        / "metrics"
        / "classifier"
        / args.split
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    IOUtils.save_json(output_dir / "metrics.json", metrics)
    IOUtils.save_json(output_dir / "predictions.json", prediction_rows)

    # --------------------------------
    error_report = ClassificationErrorAnalyzer.analyze(prediction_rows)

    IOUtils.save_json(
        output_dir / "error_analysis.json",
        error_report.to_dict(),
    )
    
    figures_dir = (
        Path(loaded_config.system.project_root)
        / "outputs"
        / "figures"
        / "classifier"
        / args.split
    )
    
    ErrorVisualizer.plot_confusion_matrix(
        confusion_matrix=metrics["confusion_matrix"],
        class_names=data_config.classes.names,
        save_path=figures_dir / "confusion_matrix.png",
    )
    
    ErrorVisualizer.plot_wrong_predictions(
        dataset=dataset,
        y_true=y_true,
        y_pred=y_pred,
        class_names=data_config.classes.names,
        save_path=figures_dir / "wrong_predictions.png",
        max_samples=9,
    )
    # --------------------------------

    logger.info("Evaluation hoàn tất.")
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro F1: %.4f", metrics["macro_f1"])
    logger.info("Metrics saved to: %s", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
