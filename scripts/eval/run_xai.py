import argparse
import os
import sys
import urllib.request
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms

# Thêm đường dẫn src để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.waste_detection.inference.hybrid_predictor import HybridPredictor
from src.waste_detection.visualization.gradcam import ErrorAnalyzer

def download_image(url, save_path):
    print(f"Đang tải ảnh thử nghiệm từ Internet...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
        out_file.write(response.read())
    print("Tải xong ảnh!")

def run_test(image_path, detector_weights, classifier_weights):
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh {image_path}. Sẽ thử tải một ảnh test mặc định...")
        image_path = 'test_trash.jpg'
        img_url = "https://images.unsplash.com/photo-1595278069441-2cf29f8005a4?q=80&w=800&auto=format&fit=crop"
        if not os.path.exists(image_path):
            download_image(img_url, image_path)

    print("\n--- BƯỚC 1 & 2: CHẠY NHẬN DIỆN VÀ PHÂN LOẠI (HYBRID PIPELINE) ---")
    
    # Class names mặc định (TACO 7 classes)
    class_names = ['Aluminium foil', 'Battery', 'Blister pack', 'Bottle', 'Bottle cap', 'Broken glass', 'Can']
    
    predictor = HybridPredictor(
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        class_names=class_names,
        detector_confidence_threshold=0.25
    )
    
    try:
        predictions = predictor.predict(image_path)
    except Exception as e:
        print(f"Lỗi khi chạy Model: {e}")
        return

    print("\n--- BƯỚC 3: PHÂN TÍCH LỖI VÀ VẼ BẢN ĐỒ NHIỆT (GRAD-CAM) ---")
    if predictions:
        best_pred = predictions[0]
        x1, y1, x2, y2 = map(int, best_pred.xyxy)
        
        # 1. Vẽ Bản đồ nhiệt Grad-CAM
        print("Đang vẽ Bản đồ nhiệt Grad-CAM cho cục rác vừa phát hiện...")
        try:
            # Lấy model CNN từ pipeline
            efficientnet = predictor.classifier.model.model
            
            # Target layer cho torchvision efficientnet
            target_layer = efficientnet.features[-1]
            
            explainer = ErrorAnalyzer(model=efficientnet, target_layer=target_layer)
            
            # Cắt ảnh
            original_img = cv2.imread(image_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            cropped_rgb = original_img[y1:y2, x1:x2]
            
            if cropped_rgb.size == 0:
                print("Lỗi cắt ảnh.")
                return

            # Tiền xử lý
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(cropped_rgb).unsqueeze(0).to(predictor.device)
            
            cropped_resized_rgb = cv2.resize(cropped_rgb, (224, 224))
            
            explainer.explain_prediction(
                input_tensor=input_tensor,
                original_image_rgb=cropped_resized_rgb,
                save_path="test_heatmap_gradcam.png"
            )
            print("Hoàn thành! Hãy mở file test_heatmap_gradcam.png để xem kết quả.")
        except Exception as e:
            print(f"Lỗi chạy Grad-CAM: {e}")
    else:
        print("Không tìm thấy rác nên không vẽ biểu đồ được!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khối XAI: Vẽ Heatmap Grad-CAM cho ADN Pipeline")
    parser.add_argument('--image', type=str, default='test_trash.jpg', help='Đường dẫn ảnh test')
    parser.add_argument('--detector', type=str, default='weights/detector.pt', help='Đường dẫn mô hình YOLO')
    parser.add_argument('--classifier_weights', type=str, default='weights/classifier.pth', help='Đường dẫn file trọng số classifier')
    
    args = parser.parse_args()
    run_test(args.image, args.detector, args.classifier_weights)
