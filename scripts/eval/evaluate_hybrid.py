from __future__ import annotations

import torch
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.coco_reader import CocoReader
from waste_detection.data.image_resolver import ImageResolver
from waste_detection.evaluation.hybrid_metrics import HybridMetrics
from waste_detection.inference.hybrid_predictor import HybridPredictor
from waste_detection.models.hybrid_model import HybridWasteModel
from waste_detection.utils.device import get_device
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup
from waste_detection.utils.timer import Timer
from waste_detection.visualization.draw_boxes import BoxDrawer


logger = logging.getLogger("evaluate_hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid YOLOv8n + EfficientNet-B0 end-to-end on full images."
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
        "--detector-weights",
        type=str,
        required=True,
        help="Đường dẫn tới YOLOv8n binary detector best.pt.",
    )

    parser.add_argument(
        "--classifier-weights",
        type=str,
        required=True,
        help="Đường dẫn tới EfficientNet-B0 classifier best.pth.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override detector confidence threshold.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Giới hạn số ảnh để debug nhanh. Không truyền khi chạy chính thức.",
    )

    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Lưu một số ảnh đã vẽ prediction.",
    )

    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Bật test-time augmentation cho detector.",
    )
    
    parser.add_argument(
        "--use-wbf",
        action="store_true",
        help="Bật Weighted Boxes Fusion khi dùng TTA.",
    )
    
    parser.add_argument(
        "--wbf-iou-threshold",
        type=float,
        default=0.55,
    )
    
    parser.add_argument(
        "--wbf-skip-box-threshold",
        type=float,
        default=0.001,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = ConfigLoader(project_root=args.project_root)

    loaded_config = loader.load_all(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        log_file_name="evaluate_hybrid.log",
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

    # Thêm cảnh báo logic TTA và WBF
    if args.use_wbf and not args.use_tta:
        logger.warning(
            "--use-wbf được bật nhưng --use-tta không bật. "
            "WBF sẽ không có tác dụng trong evaluate_hybrid hiện tại."
        )
    
    detector_weights = Path(args.detector_weights)
    classifier_weights = Path(args.classifier_weights)

    if not detector_weights.exists():
        raise FileNotFoundError(f"Không tìm thấy detector weights: {detector_weights}")

    if not classifier_weights.exists():
        raise FileNotFoundError(
            f"Không tìm thấy classifier weights: {classifier_weights}"
        )
        
    annotation_path = _get_split_annotation_path(data_config, args.split)

    dataset = CocoReader(annotation_path).load(
        validate=True,
        strict=True,
    )

    resolver = ImageResolver(
        image_root=data_config.paths.image_root,
        batch_dirs=data_config.image_structure.batch_dirs,
        allowed_extensions=data_config.image_structure.allowed_extensions,
    )

    device = get_device()

    detector_config = model_config.get("detector", {})
    classifier_config = model_config.get("classifier", {})
    hybrid_config = model_config.get("hybrid_inference", {})

    detector_confidence_threshold = (
        float(args.conf)
        if args.conf is not None
        else float(detector_config.get("confidence_threshold", 0.25))
    )

    predictor = HybridPredictor(
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        class_names=data_config.classes.names,
        detector_name=str(detector_config.get("name", "yolov8n")),
        detector_family=detector_config.get("family", None),
        detector_confidence_threshold=detector_confidence_threshold,
        detector_iou_threshold=float(detector_config.get("iou_threshold", 0.50)),
        max_detections=int(detector_config.get("max_detections", 300)),
        classifier_image_size=int(classifier_config.get("input_size", 224)),
        crop_margin=float(hybrid_config.get("crop_margin", 0.10)),
        final_confidence_rule=str(
            hybrid_config.get("final_confidence_rule", "multiply")
        ),
        final_confidence_threshold=float(
            hybrid_config.get("final_confidence_threshold", 0.25)
        ),
        use_tta=args.use_tta,
        use_wbf=args.use_wbf,
        wbf_iou_threshold=args.wbf_iou_threshold,
        wbf_skip_box_threshold=args.wbf_skip_box_threshold,
        device=device,
    )

    images = dataset.get("images", [])

    if args.max_images is not None:
        images = images[: args.max_images]

    annotations = dataset.get("annotations", [])
    categories = dataset.get("categories", [])

    category_id_to_name = {
        category["id"]: category["name"]
        for category in categories
    }

    image_id_set = {image["id"] for image in images}

    ground_truths = _build_ground_truths(
        annotations=annotations,
        image_id_set=image_id_set,
        category_id_to_name=category_id_to_name,
    )

    predictions: List[Dict[str, Any]] = []
    image_reports: List[Dict[str, Any]] = []
    inference_times = []

    output_root = (
        Path(loaded_config.system.project_root)
        / "outputs"
        / "metrics"
        / "hybrid"
        / args.split
    )

    figures_dir = (
        Path(loaded_config.system.project_root)
        / "outputs"
        / "figures"
        / "hybrid"
        / args.split
    )

    IOUtils.ensure_dir(output_root)
    IOUtils.ensure_dir(figures_dir)

    visualization_count = 0

    for image in tqdm(images, desc=f"Evaluate hybrid [{args.split}]"):
        image_id = image["id"]
        file_name = image["file_name"]

        resolved_path = resolver.resolve(file_name)

        if resolved_path is None:
            logger.warning("Không resolve được ảnh: %s", file_name)
            image_reports.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "status": "missing_image",
                    "num_predictions": 0,
                }
            )
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        
        with Timer() as timer:
            hybrid_predictions = predictor.predict(resolved_path)
        
        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_time = timer.elapsed_seconds or 0.0
        inference_times.append(inference_time)

        prediction_dicts = []

        for prediction in hybrid_predictions:
            row = prediction.to_dict()
            row["image_id"] = image_id
            row["file_name"] = file_name
            prediction_dicts.append(row)
            predictions.append(row)

        image_reports.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "resolved_path": str(resolved_path),
                "status": "ok",
                "num_predictions": len(prediction_dicts),
                "inference_time_seconds": inference_time,
            }
        )

        if args.save_visualizations and visualization_count < args.max_visualizations:
            # Tránh khi nhiều ảnh ở nhiều batch có cùng stem, có thể overwrite
            safe_stem = str(file_name).replace("\\", "/").strip("/")
            safe_stem = safe_stem.replace(":", "_").replace("/", "__")
            safe_stem = Path(safe_stem).with_suffix("").name
            
            save_path = figures_dir / f"{image_id}_{Path(file_name).stem}_hybrid_prediction.jpg"

            BoxDrawer.draw_predictions(
                image_path=resolved_path,
                predictions=prediction_dicts,
                save_path=save_path,
            )

            visualization_count += 1

    metrics = HybridMetrics.compute(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=data_config.classes.names,
        iou_threshold=args.iou_threshold,
    )

    metrics["average_inference_time_seconds"] = (
        sum(inference_times) / len(inference_times)
        if inference_times
        else 0.0
    )

    metrics["fps"] = (
        1.0 / metrics["average_inference_time_seconds"]
        if metrics["average_inference_time_seconds"] > 0
        else 0.0
    )

    hybrid_model = HybridWasteModel.from_configs(
        model_config=model_config,
        class_names=data_config.classes.names,
    )

    hybrid_model.save_description(output_root / "hybrid_model_description.json")

    IOUtils.save_json(output_root / "metrics.json", metrics)
    IOUtils.save_json(output_root / "predictions.json", predictions)
    IOUtils.save_json(output_root / "ground_truths.json", ground_truths)
    IOUtils.save_json(output_root / "image_reports.json", image_reports)

    logger.info("Evaluate hybrid hoàn tất.")
    logger.info("Metrics: %s", metrics)
    logger.info("Saved metrics: %s", output_root / "metrics.json")


def _get_split_annotation_path(data_config, split: str) -> Path:
    if split == "train":
        return data_config.paths.train_annotations_path

    if split == "val":
        return data_config.paths.val_annotations_path

    if split == "test":
        return data_config.paths.test_annotations_path

    raise ValueError(f"Split không hợp lệ: {split}")


def _build_ground_truths(
    annotations: List[Dict[str, Any]],
    image_id_set: set[int],
    category_id_to_name: Dict[int, str],
) -> List[Dict[str, Any]]:
    ground_truths = []

    for annotation in annotations:
        image_id = annotation.get("image_id")

        if image_id not in image_id_set:
            continue

        bbox = annotation.get("bbox")

        if bbox is None or len(bbox) != 4:
            continue

        category_id = int(annotation["category_id"])
        class_name = category_id_to_name.get(category_id, f"class_{category_id}")

        ground_truths.append(
            {
                "image_id": image_id,
                "annotation_id": annotation.get("id"),
                "xyxy": HybridMetrics.coco_bbox_to_xyxy(bbox),
                "class_id": category_id,
                "class_name": class_name,
            }
        )

    return ground_truths


if __name__ == "__main__":
    main()
