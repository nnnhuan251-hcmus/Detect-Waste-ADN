# Waste Detection with Transfer Learning and Fine-tuning

## 1. Project Overview

This project solves an image-based waste detection problem using Deep Learning.

- Input: raw image.
- Output: image with bounding boxes, waste class labels, and confidence scores.
- Dataset: TACO dataset in COCO annotation format.
- Main task: object detection and waste classification.
- Environment: Kaggle / Google Colab.
- Experiment tracking: Weights & Biases.

## 2. Models

This project compares three model settings:

1. Hybrid YOLOv8n + EfficientNet-B0
   - YOLOv8n detects waste bounding boxes.
   - EfficientNet-B0 classifies cropped waste objects into 7 classes.

2. YOLOv8s
   - Detector-only model.
   - Predicts bounding boxes and 7 waste classes directly.

3. RT-DETR-R18
   - Detector-only model.
   - Predicts bounding boxes and 7 waste classes directly.

## 3. Target Classes

The original TACO categories are mapped into 7 target classes:

1. plastic
2. paper
3. metal
4. glass
5. organic
6. cigarette
7. other

Annotations mapped to `ignore` are excluded from training.

## 4. Ablation Study

Each model is trained with 3 experimental settings:

### Run 1: Baseline

- Resize and normalize only.
- Standard loss.
- Pretrained initialization.
- Constant learning rate.
- End-to-end training from the beginning.

### Run 2: Strong Augmentation

- Orthogonal rotation: 90 / 180 degrees.
- Mosaic augmentation.
- Color jitter.
- Focal loss for classifier.
- Constant learning rate.

### Run 3: Freeze / Unfreeze + Cosine Warmup

- Same data configuration as Run 2.
- Freeze / unfreeze fine-tuning strategy.
- Cosine annealing learning rate scheduler.
- Warm-up.
- Gradient clipping.

## 5. Repository Structure

```text
configs/      Configuration files for data, models, and experiments.
data/         Raw, interim, and processed datasets.
notebooks/    Kaggle or Colab runners only.
scripts/      CLI entry points.
src/          Main OOP implementation.
outputs/      Checkpoints, metrics, predictions, figures, and W&B logs.
tests/        Unit tests.
```

## 6. Dataset Placement

Place the TACO dataset as follows:

```text
data/raw/TACO/
├── batch_1/
├── batch_2/
├── ...
├── batch_15/
├── annotations.json
├── annotations_unofficial.json
└── all_image_urls.csv
```

Only `annotations.json` is used in this project.

## 7. Planned Pipeline

```text
Raw TACO COCO annotations
    ↓
Check COCO structure
    ↓
Map original labels to 7 target classes
    ↓
Split train / val / test
    ↓
Convert COCO to YOLO format
    ↓
Create crop dataset for EfficientNet-B0
    ↓
Train detector / classifier / hybrid models
    ↓
Evaluate and analyze errors
```

## 8. Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## 9. Reproducibility

The project uses fixed random seeds in data splitting, training, and evaluation.

Default seed:

```text
42
```

## 10. Notes

- Kaggle notebooks should only import and run scripts.
- Main logic must be implemented inside `src/waste_detection/`.
- Do not use `annotations_unofficial.json` in the main pipeline.
