# Waste Detection with Transfer Learning and Fine-tuning

## 1. Project Overview

This repository implements a deep learning pipeline for **waste detection and classification** using transfer learning and fine-tuning.

The project focuses on detecting waste objects in images and predicting their waste category. The input is a natural image, and the output is an image with bounding boxes, class labels, and confidence scores.

The project uses two main data sources:

1. **TACO dataset**
   Official COCO-format annotation file: `annotations.json`.

2. **Roboflow COCO-format dataset**
   External COCO-format dataset used for extended experiments and robustness comparison.

The target label space is standardized into **7 waste classes**:

```text
plastic, paper, metal, glass, organic, cigarette, other
```

---

## 2. Main Objectives

The project is designed to answer the following questions:

* Can transfer learning improve waste detection performance on a small dataset?
* How does a hybrid detector-classifier architecture compare with detector-only models?
* How do data augmentation, focal loss, staged fine-tuning, cosine learning rate scheduling, and balanced sampling affect performance?
* What types of waste are commonly misclassified?
* How well does the trained model generalize to new unseen images?

---

## 3. Model Architectures

This project compares three model directions.

### 3.1 Hybrid YOLOv8n + EfficientNet-B0

The hybrid model is the main architecture of the project.

```text
Input image
→ YOLOv8n binary detector
→ detected waste bounding boxes
→ crop each detected object
→ EfficientNet-B0 classifier
→ final waste class and confidence score
```

In this architecture:

* YOLOv8n detects whether an object is waste or not.
* EfficientNet-B0 classifies each detected crop into one of the 7 target classes.
* The final confidence score is computed from detector confidence and classifier confidence.

This architecture separates localization and classification, which is useful when the detector is lightweight and the classifier can focus on cropped object regions.

---

### 3.2 YOLOv8s Detector-only

YOLOv8s is trained directly as a 7-class object detector.

```text
Input image
→ YOLOv8s
→ bounding boxes + 7-class labels + confidence scores
```

This model acts as a detector-only baseline for comparison with the hybrid approach.

---

### 3.3 RT-DETR-L Detector-only

RT-DETR-L is also trained directly as a 7-class detector.

```text
Input image
→ RT-DETR-L
→ bounding boxes + 7-class labels + confidence scores
```

RT-DETR-L is included as a transformer-based detector comparison model. Due to GPU memory requirements, it may need a smaller batch size than YOLO-based models.

---

## 4. Ablation Studies

The project includes four experiment configurations.

### Run 1: Baseline

```text
Minimal augmentation
Cross entropy loss
No class weighting
No balanced sampler
Constant learning rate
End-to-end fine-tuning
```

### Run 2: Strong Augmentation + Focal Loss

```text
Strong augmentation
Detector-side mosaic and color jitter
Classifier-side stronger crop augmentation
Focal loss for classifier
Class weighting enabled
```

### Run 3: Freeze/Unfreeze + Cosine Warm-up

```text
Strong augmentation
Focal loss
Staged fine-tuning
Backbone freeze/unfreeze schedule
Cosine annealing scheduler
Warm-up
Gradient clipping
```

### Run 4: Balanced Sampler

```text
Strong augmentation
Focal loss
Weighted random sampler for classifier crops
No class weighting
```

Run 4 is mainly designed for the EfficientNet-B0 classifier. It is not essential for detector-only models because the custom sampler is not used by the Ultralytics detector training pipeline.

---

## 5. Repository Structure

```text
our_pipeline/
│
├── README.md
├── requirements.txt
├── requirements-optional.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── data/
│   │   ├── taco_7class.yaml
│   │   ├── taco_7class_kaggle.yaml
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
│       ├── run3_freeze_cosine_warmup.yaml
│       └── run4_balanced_sampler.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       ├── coco_7class/
│       ├── yolo_7class/
│       ├── yolo_binary_waste/
│       └── crops_7class/
│
├── notebooks/
│   ├── 01_preprocess_taco_roboflow.ipynb
│   ├── 02_train_hybrid_yolov8n_effb0.ipynb
│   ├── 03_train_yolov8s_detector.ipynb
│   └── 04_train_rtdetr_l_detector.ipynb
│
├── scripts/
│   ├── data/
│   │   ├── check_coco.py
│   │   ├── prepare_taco.py
│   │   ├── build_label_mapping.py
│   │   ├── split_coco.py
│   │   ├── convert_coco_to_yolo.py
│   │   ├── create_crop_dataset.py
│   │   ├── import_roboflow_coco.py
│   │   └── apply_orthogonal_augmentation.py
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
│       ├── evaluation/
│       ├── inference/
│       ├── models/
│       ├── pipelines/
│       ├── postprocessing/
│       ├── tracking/
│       ├── training/
│       ├── utils/
│       └── visualization/
│
└── outputs/
    ├── checkpoints/
    ├── figures/
    ├── logs/
    ├── metrics/
    └── predictions/
```

---

## 6. Data Processing Pipeline

The preprocessing pipeline converts raw COCO annotations into formats required by the training pipelines.

### 6.1 TACO Preprocessing

The TACO preprocessing pipeline performs:

1. Load official TACO `annotations.json`.
2. Validate COCO structure.
3. Map original TACO categories into 7 target classes.
4. Sanitize bounding boxes.
5. Split into train/validation/test using multilabel stratification.
6. Convert COCO to YOLO format:

   * 7-class YOLO format for detector-only models.
   * binary waste YOLO format for the hybrid detector.
7. Create crop dataset for EfficientNet-B0 classifier.
8. Save preprocessing reports.

Main command:

```bash
python scripts/data/prepare_taco.py \
  --data-config configs/data/taco_7class.yaml
```

---

### 6.2 Roboflow COCO Preprocessing

The Roboflow dataset is imported through a COCO adapter.

The pipeline:

1. Read Roboflow COCO annotations.
2. Normalize image paths.
3. Standardize labels into the 7-class target label space.
4. Convert to the same processed structure as TACO.
5. Optionally combine with the processed TACO dataset.

Roboflow is used as an external dataset for additional experiments and robustness analysis.

---

## 7. Training Pipeline

### 7.1 Train Hybrid YOLOv8n Binary Detector

```bash
python scripts/train/train_detector.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml
```

The binary detector is trained on:

```text
data/processed/yolo_binary_waste/
```

---

### 7.2 Train EfficientNet-B0 Classifier

```bash
python scripts/train/train_classifier.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml
```

The classifier is trained on:

```text
data/processed/crops_7class/
```

---

### 7.3 Train YOLOv8s Detector-only Model

```bash
python scripts/train/train_detector.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/yolov8s.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml
```

The detector is trained on:

```text
data/processed/yolo_7class/
```

---

### 7.4 Train RT-DETR-L Detector-only Model

```bash
python scripts/train/train_detector.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/rtdetr_l.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml
```

RT-DETR-L may require smaller batch size due to higher GPU memory consumption.

---

## 8. Evaluation Pipeline

### 8.1 Detector Evaluation

```bash
python scripts/eval/evaluate_detector.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/yolov8s.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml \
  --weights outputs/checkpoints/yolov8s/run1_baseline/weights/best.pt \
  --split test
```

Expected metrics:

```text
mAP50
mAP50-95
mAP75
precision
recall
F1 score
per-class mAP
```

---

### 8.2 Classifier Evaluation

```bash
python scripts/eval/evaluate_classifier.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --weights outputs/checkpoints/efficientnet_b0/run1_baseline/best.pth \
  --split test
```

Expected metrics:

```text
accuracy
macro F1
weighted F1
macro precision
macro recall
classification report
confusion matrix
wrong predictions
error analysis
```

---

### 8.3 Hybrid End-to-End Evaluation

```bash
python scripts/eval/evaluate_hybrid.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --detector-weights outputs/checkpoints/hybrid_yolov8n_effb0/run1_baseline/weights/best.pt \
  --classifier-weights outputs/checkpoints/efficientnet_b0/run1_baseline/best.pth \
  --split test \
  --run-name H1_run1_baseline \
  --save-visualizations
```

Expected metrics:

```text
detection TP / FP / FN
precision
recall
F1
classification accuracy on matched objects
hybrid mAP50
hybrid mAP50-95
average inference time
FPS
```

---

## 9. Inference on New Images

Run inference on a single image:

```bash
python scripts/infer/infer_image.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --detector-weights outputs/checkpoints/hybrid_yolov8n_effb0/run1_baseline/weights/best.pt \
  --classifier-weights outputs/checkpoints/efficientnet_b0/run1_baseline/best.pth \
  --image path/to/new_image.jpg \
  --save-dir outputs/predictions
```

With test-time augmentation and weighted boxes fusion:

```bash
python scripts/infer/infer_image.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --detector-weights outputs/checkpoints/hybrid_yolov8n_effb0/run1_baseline/weights/best.pt \
  --classifier-weights outputs/checkpoints/efficientnet_b0/run1_baseline/best.pth \
  --image path/to/new_image.jpg \
  --save-dir outputs/predictions \
  --use-tta \
  --use-wbf
```

The output includes:

```text
prediction image
prediction JSON
inference time
number of detections
class labels
confidence scores
bounding boxes
```

---

## 10. Weights & Biases Tracking

Training can optionally be tracked with Weights & Biases.

Example:

```bash
python scripts/train/train_classifier.py \
  --data-config configs/data/taco_7class.yaml \
  --model-config configs/models/hybrid_yolov8n_effb0.yaml \
  --experiment-config configs/experiments/run1_baseline.yaml \
  --use-wandb \
  --wandb-mode online
```

Tracked information includes:

```text
training loss
validation loss
accuracy
macro F1
learning rate
model checkpoint artifacts
training configuration
```

For detector models, Ultralytics training results are logged through the generated `results.csv`.

---

## 11. Expected Output Artifacts

After running experiments, the repository saves results into:

```text
outputs/checkpoints/
```

Model checkpoints:

```text
YOLO / RT-DETR:
outputs/checkpoints/{model_name}/{run_name}/weights/best.pt
outputs/checkpoints/{model_name}/{run_name}/weights/last.pt

EfficientNet-B0:
outputs/checkpoints/efficientnet_b0/{run_name}/best.pth
```

```text
outputs/metrics/
```

Evaluation metrics:

```text
detector metrics
classifier metrics
hybrid metrics
error analysis
wrong predictions
experiment registry
```

```text
outputs/figures/
```

Figures for reporting:

```text
training curves
confusion matrices
wrong prediction grids
detector PR/F1 curves
hybrid prediction visualizations
```

```text
outputs/predictions/
```

Inference results:

```text
predicted images
prediction JSON files
```

---

## 12. Recommended Experiment Matrix

### Hybrid Model

```text
H1 = detector run1 + classifier run1
H2 = detector run2 + classifier run2
H3 = detector run3 + classifier run3
H4 = detector run2 + classifier run4
```

Run 4 is paired with detector run 2 because run 4 mainly changes the classifier sampling strategy.

### YOLOv8s Detector-only

```text
YOLOv8s run1
YOLOv8s run2
YOLOv8s run3
```

### RT-DETR-L Detector-only

```text
RT-DETR-L run1
RT-DETR-L run2 if GPU memory allows
RT-DETR-L run3 if GPU memory allows
```

---

## 13. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install -r requirements-optional.txt
```

For local development:

```bash
pip install -r requirements-dev.txt
```

Set Python path:

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

---

## 14. Notes

* The hybrid model is not trained end-to-end as a single neural network.
* YOLOv8n and EfficientNet-B0 are trained separately.
* End-to-end behavior happens during inference and evaluation.
* TACO and Roboflow data should be standardized into the same 7-class label space before training.
* Offline augmentation must be applied only to the training split to avoid data leakage.
* Smoke tests should be run before full training to catch runtime, path, and configuration errors.

---

## 15. Authors

This repository was developed for the project:

```text
Waste Detection with Transfer Learning and Fine-tuning
```

The implementation follows a modular object-oriented design with separated components for data preprocessing, model training, evaluation, inference, visualization, and experiment tracking.
