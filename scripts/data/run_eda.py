import os
import json
import yaml
import argparse
from pathlib import Path
import sys

# Đảm bảo có thể import được src
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.waste_detection.data.eda_analyzer import CocoEDAAnalyzer
from src.waste_detection.data.label_mapper import TacoLabelMapper

def load_mapping_dict(mapping_file: str):
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_label_mapper(classes_info: dict) -> dict:
    """Tạo dictionary mapping từ original name sang target name"""
    # Mặc dù hệ thống có TacoLabelMapper, nhưng ở bước EDA (dữ liệu thô),
    # ta có thể đơn giản tạo một hàm mapper nhanh gọn để truyền vào CocoEDAAnalyzer
    # vì CocoEDAAnalyzer yêu cầu một callable `label_mapper(cat_id, cat_name) -> mapped_name`
    pass

def main():
    parser = argparse.ArgumentParser(description="Run Full OOP EDA Pipeline")
    parser.add_argument('--dataset', type=str, default='taco', choices=['taco', 'roboflow', 'all'],
                        help="Chọn dataset để chạy EDA")
    args = parser.parse_args()
    
    output_dir = os.path.join(project_root, 'outputs', 'figures', 'eda')
    os.makedirs(output_dir, exist_ok=True)
    
    # === 1. CHẠY EDA CHO TACO ===
    if args.dataset in ['taco', 'all']:
        taco_ann_path = os.path.join(project_root, 'data', 'raw', 'TACO', 'annotations.json')
        taco_img_dir = os.path.join(project_root, 'data', 'raw', 'TACO')
        
        if not os.path.exists(taco_ann_path):
            print(f"Không tìm thấy {taco_ann_path}. Vui lòng tải dữ liệu TACO trước!")
        else:
            # Nạp mapping config để ánh xạ 60 nhãn về 7 nhãn
            mapping_file = os.path.join(project_root, 'configs', 'data', 'mapping_label.json')
            mapping_dict = load_mapping_dict(mapping_file)
            
            # Xây dựng hàm mapping ngược: từ original_name -> target_name
            original_to_target = {}
            for target_name, original_names in mapping_dict.items():
                for orig in original_names:
                    original_to_target[orig] = target_name
                    
            def taco_mapper(cat_id, cat_name):
                mapped = original_to_target.get(cat_name)
                if mapped == "ignore" or mapped is None:
                    return None
                return mapped

            print("\n" + "="*50)
            print("🚀 BẮT ĐẦU EDA CHO TACO DATASET (ĐÃ ÁNH XẠ 7 NHÃN)")
            print("="*50)
            taco_analyzer = CocoEDAAnalyzer(
                annotation_file=taco_ann_path,
                image_dir=taco_img_dir,
                dataset_name="TACO_7Class",
                label_mapper=taco_mapper
            )
            taco_analyzer.run_full_eda(save_dir=output_dir)

    # === 2. CHẠY EDA CHO ROBOFLOW (Nếu có) ===
    if args.dataset in ['roboflow', 'all']:
        # Dữ liệu Roboflow thường tải về thư mục data/raw/Roboflow
        # Bạn có thể điều chỉnh đường dẫn theo thực tế
        robo_ann_path = os.path.join(project_root, 'data', 'raw', 'Roboflow', '_annotations.coco.json')
        robo_img_dir = os.path.join(project_root, 'data', 'raw', 'Roboflow')
        
        if not os.path.exists(robo_ann_path):
            print(f"\nKhông tìm thấy {robo_ann_path}. Bỏ qua EDA cho Roboflow.")
        else:
            # Roboflow tải về thường đã gán sẵn 7 nhãn, nên không cần mapper
            print("\n" + "="*50)
            print("🚀 BẮT ĐẦU EDA CHO ROBOFLOW DATASET")
            print("="*50)
            robo_analyzer = CocoEDAAnalyzer(
                annotation_file=robo_ann_path,
                image_dir=robo_img_dir,
                dataset_name="Roboflow",
                label_mapper=None
            )
            robo_analyzer.run_full_eda(save_dir=output_dir)
            
    print(f"\n✅ Hoàn tất! Tất cả biểu đồ EDA đã được lưu tại: {output_dir}")

if __name__ == '__main__':
    main()
