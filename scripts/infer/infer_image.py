from __future__ import annotations

import argparse
import logging
import time
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
        description="Run hybrid inference on a single image with optional TTA and WBF."
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

    # 1. Nạp và kiểm tra tính toàn vẹn của cấu hình
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

    # 2. Khởi tạo Logger tập trung
    LoggerSetup.initialize(
        log_file=loaded_config.system.log_file_path,
        clear_old_logs=True,
    )

    device = get_device()
    logger.info("Thiết bị tính toán được sử dụng: %s", device)

    # 3. Trích xuất tham số cấu hình từ mô hình
    detector_config = model_config.get("detector", {})
    classifier_config = model_config.get("classifier", {})
    hybrid_config = model_config.get("hybrid_inference", {})

    detector_confidence_threshold = (
        float(args.conf)
        if args.conf is not None
        else float(detector_config.get("confidence_threshold", 0.25))
    )

    # 4. Khởi tạo thực thể Dự đoán Hybrid (Dependency Injection)
    predictor = HybridPredictor(
        detector_weights=args.detector_weights,
        classifier_weights=args.classifier_weights,
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

    # 5. Thực thi xử lý ảnh và đo lường thời gian (KPI đánh giá)
    logger.info("Đang tiến hành nhận diện ảnh: %s", args.image)
    start_time = time.perf_counter()
    predictions = predictor.predict(args.image)
    elapsed_time = time.perf_counter() - start_time
    
    logger.info("Thời gian xử lý nhận diện (Inference Time): %.4f giây", elapsed_time)

    # 6. Chuẩn hóa cấu trúc dữ liệu đầu ra và xuất bản kết quả qua IOUtils
    prediction_dicts = [prediction.to_dict() for prediction in predictions]

    save_dir = Path(args.save_dir)
    IOUtils.ensure_dir(save_dir)  # Chuẩn hóa việc phân lớp I/O cấp thấp

    image_path = Path(args.image)
    output_image_path = save_dir / f"{image_path.stem}_prediction.jpg"
    output_json_path = save_dir / f"{image_path.stem}_prediction.json"

    # Vẽ bounding boxes lên ảnh kết quả
    BoxDrawer.draw_predictions(
        image_path=args.image,
        predictions=prediction_dicts,
        save_path=output_image_path,
    )

    # Lưu thông tin tọa độ chi tiết
    IOUtils.save_json(
        dest_path=output_json_path,
        data={
            "inference_time_seconds": elapsed_time,
            "num_detections": len(prediction_dicts),
            "predictions": prediction_dicts
        },
    )

    logger.info("Quy trình Inference hoàn tất thành công.")
    logger.info("Ảnh kết quả được lưu tại: %s", output_image_path)
    logger.info("File JSON dự đoán được lưu tại: %s", output_json_path)


if __name__ == "__main__":
    main()
