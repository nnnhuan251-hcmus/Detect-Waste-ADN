import argparse
import cv2
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from waste_detection.inference.hybrid_predictor import HybridPredictor
from waste_detection.inference.classifier_predictor import ClassifierPredictor

def run_ablation_comparison(image_path, detector_weights, classifier_weights):
    print(f"BẮT ĐẦU THỰC NGHIỆM ABLATION STUDY TRÊN ẢNH: {image_path}")
    # Class names mặc định của đồ án (7 Macro-classes)
    class_names = ['plastic', 'paper', 'metal', 'glass', 'organic', 'cigarette', 'other']
    # Nạp mô hình
    classifier = ClassifierPredictor(weights_path=classifier_weights, class_names=class_names)
    hybrid = HybridPredictor(
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        class_names=class_names,
        detector_confidence_threshold=0.1
    )
    
    # Đọc ảnh gốc
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Lỗi: Không thể đọc được ảnh {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # -------------------------------------------------------------------------
    # PHƯƠNG PHÁP 1: TRUYỀN THẲNG ẢNH GỐC VÀO CLASSIFIER (BASELINE)
    # -------------------------------------------------------------------------
    print("\n--- PHƯƠNG PHÁP 1: CLASSIFIER THUẦN (KHÔNG DETECT) ---")
    baseline_res = classifier.predict_one(img_rgb)
    baseline_pred = baseline_res.class_name
    baseline_prob = baseline_res.confidence
    print(f"Kết quả dự đoán toàn cảnh: {baseline_pred} ({baseline_prob*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # PHƯƠNG PHÁP 2: DETECT -> CROP -> CLASSIFY (PROPOSED)
    # -------------------------------------------------------------------------
    print("\n--- PHƯƠNG PHÁP 2: PIPELINE ĐỀ XUẤT (DETECT + CROP + CLASSIFY) ---")
    proposed_preds = hybrid.predict(image_path)
    if len(proposed_preds) == 0:
        print("YOLO không tìm thấy cục rác nào trong ảnh này.")

    # -------------------------------------------------------------------------
    # VẼ BIỂU ĐỒ SO SÁNH TRỰC QUAN
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Khung 1: Baseline
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"[Baseline] Truyền thẳng ảnh gốc vào Classifier\nDự đoán: {baseline_pred} ({baseline_prob*100:.1f}%)", fontsize=14, color='red')
    axes[0].axis('off')
    
    # Khung 2: Proposed
    img_boxes = img_rgb.copy()
    for pred in proposed_preds:
        x_min, y_min, x_max, y_max = map(int, pred.xyxy)
        cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
        
        label = f"{pred.class_name} ({pred.classifier_confidence*100:.1f}%)"
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img_boxes, (x_min, y_min - 30), (x_min + w, y_min), (0, 255, 0), -1)
        cv2.putText(img_boxes, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
    axes[1].imshow(img_boxes)
    if len(proposed_preds) > 0:
        axes[1].set_title(f"[Đề xuất] Hybrid Pipeline (YOLO + Classifier)\nTìm thấy {len(proposed_preds)} đối tượng rác", fontsize=14, color='green')
    else:
        axes[1].set_title(f"[Đề xuất] Hybrid Pipeline (YOLO + Classifier)\nKhông thấy rác", fontsize=14, color='green')
    axes[1].axis('off')
    
    plt.tight_layout()
    output_filename = "ablation_comparison_result.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\n[HOÀN TẤT] Đã lưu ảnh so sánh cực kỳ trực quan tại: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh hiệu quả của Hybrid Pipeline vs Classifier thuần")
    parser.add_argument('--image', type=str, required=True, help="Đường dẫn tới bức ảnh test")
    parser.add_argument('--detector', type=str, default='weights/detector.pt', help="File weights của YOLO")
    parser.add_argument('--classifier_weights', type=str, default='weights/classifier.pth', help="File weights của Classifier")
    
    args = parser.parse_args()
    run_ablation_comparison(args.image, args.detector, args.classifier_weights)
