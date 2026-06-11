import argparse
import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waste_detection.inference.hybrid_predictor import HybridPredictor

def run_easy_inference(image_path, detector_weights, classifier_weights, conf_threshold):
    print("=== KHỞI ĐỘNG DÂY CHUYỀN XỬ LÝ RÁC (ADN PIPELINE) ===")
    
    class_names = ['Aluminium foil', 'Battery', 'Blister pack', 'Bottle', 'Bottle cap', 'Broken glass', 'Can']
    
    predictor = HybridPredictor(
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        class_names=class_names,
        detector_confidence_threshold=conf_threshold
    )
    
    print(f"-> Đang xử lý bức ảnh: {image_path}")
    predictions = predictor.predict(image_path)
    
    if not predictions:
        print("Không tìm thấy rác trong ảnh này!")
        return

    print(f"[Pipeline] Đã phát hiện và phân loại thành công {len(predictions)} mảnh rác.")
    
    img = cv2.imread(image_path)
    
    print("\n=== KẾT QUẢ CUỐI CÙNG ===")
    for i, pred in enumerate(predictions):
        x_min, y_min, x_max, y_max = map(int, pred.xyxy)
        
        display_text = f"{pred.class_name.upper()} ({pred.classifier_confidence*100:.1f}%)"
        print(f"Cục rác thứ {i+1} ở tọa độ {[x_min, y_min, x_max, y_max]} là: {display_text}")
        
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x_min, y_min - 30), (x_min + w, y_min), (0, 255, 0), -1)
        cv2.putText(img, display_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    print("=========================")
    
    output_path = "inference_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"\n[Visualizer] TADA! Đã vẽ xong khung, nhãn và độ tin cậy.")
    print(f"[Visualizer] Ảnh trực quan đã được lưu tại: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công cụ chạy Inference siêu tốc cho ADN Pipeline")
    parser.add_argument('--image', type=str, required=True, help='Đường dẫn ảnh test')
    parser.add_argument('--detector', type=str, default='weights/detector.pt', help='Đường dẫn mô hình YOLO')
    parser.add_argument('--classifier_weights', type=str, default='weights/classifier.pth', help='Đường dẫn file trọng số classifier')
    parser.add_argument('--conf', type=float, default=0.25, help='Ngưỡng tin cậy của YOLO')
    
    args = parser.parse_args()
    run_easy_inference(args.image, args.detector, args.classifier_weights, args.conf)
