from __future__ import annotations

import torch
import time
import argparse
import logging
from pathlib import Path

from waste_detection.config.config_loader import ConfigLoader
from waste_detection.inference.hybrid_predictor import HybridPredictor
from waste_detection.utils.device import get_device
from waste_detection.utils.io import IOUtils
from waste_detection.utils.logger import LoggerSetup
from waste_detection.visualization.draw_boxes import BoxDrawer


logger = logging.getLogger("infer_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid inference on a single image."
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
        help="Đường dẫn tới detector best.pt.",
    )

    parser.add_argument(
        "--classifier-weights",
        type=str,
        required=True,
        help="Đường dẫn tới EfficientNet-B0 best.pth.",
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Đường dẫn ảnh cần inference.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/predictions",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override detector confidence threshold nếu muốn.",
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
        help="Bật Weighted Boxes Fusion để gộp bbox từ TTA views.",
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
        log_file_name="infer_image.log",
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

    if args.use_wbf and not args.use_tta:
        logger.warning(
            "--use-wbf được bật nhưng --use-tta không bật. "
            "WBF sẽ không có tác dụng trong inference hiện tại."
        )

    detector_weights = Path(args.detector_weights)
    classifier_weights = Path(args.classifier_weights)
    image_path = Path(args.image)

    if not detector_weights.exists():
        raise FileNotFoundError(f"Không tìm thấy detector weights: {detector_weights}")

    if not classifier_weights.exists():
        raise FileNotFoundError(f"Không tìm thấy classifier weights: {classifier_weights}")

    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh inference: {image_path}")
    
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

    logger.info("Đang tiến hành nhận diện ảnh: %s", image_path)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    predictions = predictor.predict(image_path)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time_seconds = time.perf_counter() - start_time
    
    # Bạn có thể log thêm thời gian ra console để dễ theo dõi
    logger.info("Hoàn tất nhận diện! Thời gian xử lý: %.4f giây", inference_time_seconds)
    
    prediction_dicts = [prediction.to_dict() for prediction in predictions]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    output_image_path = save_dir / f"{image_path.stem}_prediction.jpg"
    output_json_path = save_dir / f"{image_path.stem}_prediction.json"

    BoxDrawer.draw_predictions(
        image_path=args.image,
        predictions=prediction_dicts,
        save_path=output_image_path,
    )

    IOUtils.save_json(
        dest_path=output_json_path,
        data=prediction_dicts,
    )

    logger.info("Inference hoàn tất.")
    logger.info("Saved image: %s", output_image_path)
    logger.info("Saved predictions: %s", output_json_path)


if __name__ == "__main__":
    main()
