from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from waste_detection.data.image_resolver import ImageResolver
from waste_detection.data.orthogonal_augmenter import OrthogonalAugmenter


def _make_test_image(path: Path, width: int = 100, height: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (255, 255, 255)
    cv2.imwrite(str(path), image)


def _build_sample_coco(file_name: str):
    return {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": file_name,
                "width": 100,
                "height": 80,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
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


def test_orthogonal_augmenter_adds_rotated_images_and_annotations(tmp_path):
    image_root = tmp_path / "images"
    source_image = image_root / "batch_1" / "sample.jpg"

    _make_test_image(source_image)

    dataset = _build_sample_coco("batch_1/sample.jpg")

    resolver = ImageResolver(
        image_root=image_root,
        batch_dirs=["batch_1"],
        allowed_extensions=[".jpg", ".jpeg", ".png"],
    )

    output_image_dir = tmp_path / "augmented_images"

    augmenter = OrthogonalAugmenter(
        output_image_dir=output_image_dir,
        angles=[90, 180],
        use_absolute_file_names=True,
        jpeg_quality=95,
    )

    augmented_dataset, report = augmenter.augment_dataset(
        dataset=dataset,
        image_resolver=resolver,
    )

    assert report.num_original_images == 1
    assert report.num_original_annotations == 1
    assert report.num_augmented_images == 2
    assert report.num_augmented_annotations == 2
    assert report.num_missing_images == 0
    assert report.num_invalid_bboxes == 0

    assert len(augmented_dataset["images"]) == 3
    assert len(augmented_dataset["annotations"]) == 3

    augmented_files = list(output_image_dir.glob("*.jpg"))
    assert len(augmented_files) == 2


def test_orthogonal_augmenter_rejects_non_orthogonal_angle(tmp_path):
    try:
        OrthogonalAugmenter(
            output_image_dir=tmp_path,
            angles=[45],
        )
    except ValueError as error:
        assert "Chỉ hỗ trợ" in str(error)
    else:
        raise AssertionError("OrthogonalAugmenter phải reject angle 45.")
