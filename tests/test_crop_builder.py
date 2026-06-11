from __future__ import annotations

from pathlib import Path

from PIL import Image

from waste_detection.data.crop_builder import CropDatasetBuilder


def _make_image(path: Path, size=(100, 80), color=(255, 255, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size=size, color=color)
    image.save(path)


def _build_sample_coco(image_path: Path):
    return {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": str(image_path),
                "width": 100,
                "height": 80,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 10, 30, 20],
                "area": 600,
                "iscrowd": 0,
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "plastic",
                "supercategory": "waste",
            }
        ],
    }


def test_crop_dataset_builder_creates_class_folder_and_crop(tmp_path):
    source_image = tmp_path / "raw" / "sample.jpg"
    output_root = tmp_path / "crops"

    _make_image(source_image)

    dataset = _build_sample_coco(source_image)

    class_names = [
        "plastic",
        "paper",
        "metal",
        "glass",
        "organic",
        "cigarette",
        "other",
    ]

    builder = CropDatasetBuilder(
        output_root=output_root,
        class_names=class_names,
        margin_ratio=0.10,
        min_crop_size=8,
        save_format="jpg",
        jpeg_quality=95,
    )

    report = builder.build_from_coco(
        dataset=dataset,
        image_root=tmp_path,
        split_name="train",
    )

    crop_dir = output_root / "train" / "plastic"

    assert crop_dir.exists()

    crop_files = list(crop_dir.glob("*.jpg"))

    assert len(crop_files) == 1
    assert report.num_saved_crops == 1
    assert report.num_invalid_bboxes == 0


def test_crop_dataset_builder_skips_invalid_bbox(tmp_path):
    source_image = tmp_path / "raw" / "sample.jpg"
    output_root = tmp_path / "crops"

    _make_image(source_image)

    dataset = _build_sample_coco(source_image)
    dataset["annotations"][0]["bbox"] = [10, 10, 0, 20]

    class_names = [
        "plastic",
        "paper",
        "metal",
        "glass",
        "organic",
        "cigarette",
        "other",
    ]

    builder = CropDatasetBuilder(
        output_root=output_root,
        class_names=class_names,
        margin_ratio=0.10,
        min_crop_size=8,
        save_format="jpg",
        jpeg_quality=95,
    )

    report = builder.build_from_coco(
        dataset=dataset,
        image_root=tmp_path,
        split_name="train",
    )

    crop_dir = output_root / "train" / "plastic"
    crop_files = list(crop_dir.glob("*.jpg")) if crop_dir.exists() else []

    assert len(crop_files) == 0
    assert report.num_saved_crops == 0
    assert report.num_invalid_bboxes == 1
