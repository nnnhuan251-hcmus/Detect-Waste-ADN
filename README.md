# Waste Detection with Transfer Learning and Fine-tuning

This repository contains the source code for a Deep Learning final project on waste detection and classification using transfer learning and fine-tuning.

The project focuses on building and comparing three model pipelines:

1. **Hybrid YOLOv8n + EfficientNet-B0**

   * YOLOv8n detects waste objects as binary `waste`.
   * EfficientNet-B0 classifies each detected crop into 7 waste categories.
   * Final confidence is computed from detector confidence and classifier confidence.

2. **YOLOv8s detector-only**

   * YOLOv8s is trained directly on 7 waste classes.

3. **RT-DETR-L detector-only**

   * RT-DETR-L is trained directly on 7 waste classes as a transformer-based detector comparison.

The main dataset is the official TACO dataset using the official COCO-format `annotations.json`. The pipeline maps original TACO categories into 7 target waste classes.

---

## 1. Target Classes

The project uses exactly 7 target classes:

```text
plastic
paper
metal
glass
organic
cigarette
other
```

Class ids are 0-based:

```text
plastic: 0
paper: 1
metal: 2
glass: 3
organic: 4
cigarette: 5
other: 6
```

---

## 2. Main Dataset

Primary dataset:

```text
TACO official COCO annotations
```

Expected raw dataset structure:

```text
data/raw/TACO/
├── annotations.json
├── batch_1/
├── batch_2/
├── batch_3/
├── ...
└── batch_15/
```

Important notes:

* Use the official `annotations.json`.
* Do not use `annotations_unofficial.json`.
* Original TACO categories are mapped into 7 target classes using:

```text
configs/data/mapping_label.json
```

---

## 3. Additional Dataset Support

The main experiment uses TACO official COCO annotations.

The repository also includes an adapter for Roboflow COCO datasets. This allows a Roboflow COCO dataset with `train`, `valid`, and `test` splits to be imported into the same internal COCO split format used by this project.

Roboflow support is optional and does not change the main TACO experiment.

---

## 4. Project Structure

```text
our_pipeline/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── data/
│   │   ├── taco_7class.yaml
│   │   └── mapping_label.json
│   │
│   ├── models/
│   │   ├── hybrid_yolov8n_effb0.yaml
│   │   ├── yolov8s.yaml
│   │   └── rtdetr_l.yaml
│   │
│   └── experiments/
│       ├── run1_baseline.yaml
│       ├── run2_strong_aug.yaml
│       └── run3_freeze_cosine_warmup.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   └── kaggle_runner.ipynb
│
├── scripts/
│   ├── data/
│   │   ├── check_coco.py
│   │   ├── build_label_mapping.py
│   │   ├── prepare_taco.py
│   │   ├── split_coco.py
│   │   ├── convert_coco_to_yolo.py
│   │   ├── create_crop_dataset.py
│   │   ├── apply_orthogonal_augmentation.py
│   │   └── import_roboflow_coco.py
│   │
│   ├── train/
│   │   ├── train_detector.py
│   │   ├── train_classifier.py
│   │   └── run_ablation.py
│   │
│   ├── eval/
│   │   ├── evaluate_detector.py
│   │   ├── evaluate_classifier.py
│   │   └── evaluate_hybrid.py
│   │
│   └── infer/
│       └── infer_image.py
│
├── src/
│   └── waste_detection/
│       ├── config/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       ├── visualization/
│       ├── tracking/
│       └── utils/
│
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   └── logs/
│
└── tests/
```

---

## 5. Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the repository as an editable Python package:

```bash
pip install -e .
```

After installation, project modules can be imported as:

```python
from waste_detection.config.config_loader import ConfigLoader
from waste_detection.data.crop_builder import CropDatasetBuilder
from waste_detection.inference.hybrid_predictor import HybridPredictor
```

---

## 6. Data Preparation Pipeline

The TACO preparation pipeline performs the following steps:

```text
Official TACO annotations.json
→ Validate COCO format
→ Map original TACO categories into 7 target classes
→ Split into train/val/test at image level
→ Convert COCO into YOLO 7-class format
→ Convert COCO into YOLO binary-waste format
→ Create crop dataset for EfficientNet-B0
```

Generated data:

```text
data/processed/coco_7class/
├── annotations_7class.json
├── train_annotations.json
├── val_annotations.json
└── test_annotations.json
```

```text
data/processed/yolo_7class/
├── data.yaml
├── images/
└── labels/
```

```text
data/processed/yolo_binary_waste/
├── data.yaml
├── images/
└── labels/
```

```text
data/processed/crops_7class/
├── train/
├── val/
└── test/
```

---

## 7. Model Pipelines

### 7.1 Hybrid YOLOv8n + EfficientNet-B0

The hybrid model is trained in two independent stages:

```text
Stage 1:
YOLOv8n binary detector
Input: full original images
Output: bounding boxes for waste objects

Stage 2:
EfficientNet-B0 classifier
Input: ground-truth object crops from train/val/test
Output: 7-class waste prediction
```

End-to-end evaluation is performed differently:

```text
Full image
→ YOLOv8n predicts bounding boxes
→ Predicted boxes are cropped
→ EfficientNet-B0 classifies each predicted crop
→ Final hybrid prediction is evaluated against ground truth
```

Important:

```text
Ground-truth crops are used only for isolated classifier training/evaluation.
Final hybrid test metrics must use full images, not ground-truth crops.
```

---

### 7.2 YOLOv8s Detector-only

YOLOv8s is trained directly on the 7 waste classes.

Input:

```text
data/processed/yolo_7class/data.yaml
```

Output:

```text
7-class bounding box prediction
```

---

### 7.3 RT-DETR-L Detector-only

RT-DETR-L is used as the transformer-based detector comparison.

Input:

```text
data/processed/yolo_7class/data.yaml
```

Output:

```text
7-class bounding box prediction
```

---

## 8. Experiment Configurations

The repository contains three main experiment configurations:

```text
configs/experiments/run1_baseline.yaml
configs/experiments/run2_strong_aug.yaml
configs/experiments/run3_freeze_cosine_warmup.yaml
```

### Run 1: Baseline

```text
Resize
Normalize
Pretrained initialization
Constant learning rate
No strong augmentation
No focal loss
No staged freeze/unfreeze
```

### Run 2: Strong Augmentation + Focal Loss

```text
Horizontal flip
Exact 90/180-degree classifier rotation
Mosaic for detector
Color jitter
Focal loss for classifier
Class weighting
Constant learning rate
```

Detector diagonal rotation is intentionally disabled to avoid distorted axis-aligned bounding boxes.

### Run 3: Strong Augmentation + Freeze/Cosine/Warmup

```text
Strong augmentation
Focal loss
Class weighting
Backbone freezing
Gradual unfreezing
Cosine annealing scheduler
Warm-up
Gradient clipping
```

---

## 9. Training Scripts

### 9.1 Train detector

```bash
python scripts/train/train_detector.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/yolov8s.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml
```

For RT-DETR-L:

```bash
python scripts/train/train_detector.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/rtdetr_l.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml
```

For hybrid YOLOv8n binary detector:

```bash
python scripts/train/train_detector.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml
```

---

### 9.2 Train classifier

The classifier is used only for the hybrid pipeline.

```bash
python scripts/train/train_classifier.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml
```

---

## 10. Evaluation Scripts

### 10.1 Evaluate detector

```bash
python scripts/eval/evaluate_detector.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/yolov8s.yaml \
  --weights outputs/checkpoints/path_to_detector_best.pt \
  --split test
```

### 10.2 Evaluate classifier

```bash
python scripts/eval/evaluate_classifier.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --weights outputs/checkpoints/path_to_classifier_best.pth \
  --split test
```

### 10.3 Evaluate hybrid end-to-end

```bash
python scripts/eval/evaluate_hybrid.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --detector-weights outputs/checkpoints/path_to_yolov8n_binary_best.pt \
  --classifier-weights outputs/checkpoints/path_to_classifier_best.pth \
  --split test
```

---

## 11. Các Công Cụ Phân Tích & Chạy Nhanh (Được kế thừa từ our_pipeline)

### 11.1 Chạy Nhận Diện Siêu Tốc (Easy Inference)

Bạn không cần truyền các file cấu hình YAML phức tạp. Chỉ cần dùng kịch bản sau để chạy nhanh một bức ảnh:

```bash
python scripts/easy_infer.py \
  --image path/to/test_image.jpg \
  --detector weights/detector.pt \
  --classifier_weights weights/classifier.pth \
  --conf 0.25
```
Kết quả trực quan (Bounding Box + Nhãn + Độ tin cậy) sẽ được lưu thẳng thành `inference_result.jpg`.

### 11.2 Bản Đồ Nhiệt Giải Thích (XAI - Grad-CAM)

Đây là công cụ giúp bạn ghi điểm tuyệt đối trong báo cáo. Nó sẽ vẽ bản đồ nhiệt cho thấy mạng CNN đang "nhìn" vào điểm nào trên cục rác để phân loại.

```bash
python scripts/eval/run_xai.py \
  --image path/to/test_image.jpg \
  --detector weights/detector.pt \
  --classifier_weights weights/classifier.pth
```
Hệ thống sẽ tự cắt ảnh và lưu biểu đồ vào `test_heatmap_gradcam.png`.

### 11.3 So Sánh Kiến Trúc (Ablation Comparison)

Hiển thị trực quan sự khác biệt giữa việc: Truyền thẳng ảnh vào CNN (bị nhiễu hậu cảnh) VS Việc dùng YOLO cắt rác rồi mới phân loại.

```bash
python scripts/eval/ablation_comparison.py \
  --image path/to/test_image.jpg \
  --detector weights/detector.pt \
  --classifier_weights weights/classifier.pth
```
Kết quả được trình bày cạnh nhau, siêu trực quan tại `ablation_comparison_result.png`.

---

## 12. Metrics

### Detection metrics

```text
mAP@50
mAP@50:95
Precision
Recall
F1
Per-class AP
Inference time
FPS
```

### Classification metrics

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted F1
Confusion matrix
Wrong prediction visualization
```

### Hybrid metrics

```text
End-to-end precision
End-to-end recall
End-to-end F1
Classification accuracy on matched boxes
mAP@50
mAP@50:95
False positives
Missed objects
Classification errors
Localization errors
```

---

## 13. Outputs

Training and evaluation outputs are saved under:

```text
outputs/
├── checkpoints/
├── metrics/
├── predictions/
├── figures/
└── logs/
```

These files are ignored by Git and should not be committed.

---

## 14. Testing

Run unit tests:

```bash
pytest
```

Tests cover:

```text
Label mapping
BBox conversion and IoU
Crop dataset builder
Orthogonal bbox-aware augmentation
```

---

## 15. Git Notes

Do not commit:

```text
data/raw/
data/processed/
outputs/
runs/
wandb/
*.pt
*.pth
```

Commit only:

```text
source code
configs
scripts
tests
README
requirements
pyproject.toml
```

---

## 16. Project Scope

This project does not perform segmentation.

The main scope is:

```text
Object detection
Crop-based image classification
Hybrid detector-classifier inference
Transfer learning
Fine-tuning
Ablation study
Error analysis
```

The final report should clearly distinguish:

```text
Classifier-only evaluation on ground-truth crops
```

from:

```text
Hybrid end-to-end evaluation on full original images
```
