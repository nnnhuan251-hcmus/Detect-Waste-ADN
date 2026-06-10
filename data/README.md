# Data Directory

This directory stores raw, interim, and processed data.

## 1. Raw Data

Place the original TACO dataset here:

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

Only `annotations.json` is used in the main pipeline.

## 2. Interim Data

```text
data/interim/
```

Temporary intermediate files can be stored here during preprocessing.

## 3. Processed Data

```text
data/processed/
├── coco_7class/
├── yolo_7class/
├── yolo_binary_waste/
└── crops_7class/
```

### `coco_7class/`

Stores COCO annotations after mapping original TACO categories into 7 target classes.

### `yolo_7class/`

Stores YOLO-format detection dataset for YOLOv8s and RT-DETR detector-only experiments.

### `yolo_binary_waste/`

Stores YOLO-format binary detection dataset for the detector stage of the hybrid YOLOv8n + EfficientNet-B0 pipeline.

### `crops_7class/`

Stores cropped object images for EfficientNet-B0 classification.
