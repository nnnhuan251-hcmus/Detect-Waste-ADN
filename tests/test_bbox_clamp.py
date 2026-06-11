from __future__ import annotations

from waste_detection.evaluation.hybrid_metrics import HybridMetrics
from waste_detection.data.orthogonal_augmenter import OrthogonalAugmenter


def test_coco_bbox_to_xyxy_conversion():
    bbox = [10, 20, 30, 40]

    xyxy = HybridMetrics.coco_bbox_to_xyxy(bbox)

    assert xyxy == [10.0, 20.0, 40.0, 60.0]


def test_iou_identical_boxes_is_one():
    box = [10, 20, 40, 60]

    iou = HybridMetrics.compute_iou_xyxy(box, box)

    assert iou == 1.0


def test_iou_non_overlapping_boxes_is_zero():
    box_a = [0, 0, 10, 10]
    box_b = [20, 20, 30, 30]

    iou = HybridMetrics.compute_iou_xyxy(box_a, box_b)

    assert iou == 0.0


def test_rotate_coco_bbox_180_degrees():
    image_width = 100
    image_height = 80

    bbox = [10, 20, 30, 40]

    rotated = OrthogonalAugmenter._rotate_coco_bbox(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        angle=180,
    )

    assert rotated is not None

    x, y, width, height = rotated

    assert x == 60.0
    assert y == 20.0
    assert width == 30.0
    assert height == 40.0


def test_rotate_coco_bbox_90_degrees_clockwise():
    image_width = 100
    image_height = 80

    bbox = [10, 20, 30, 40]

    rotated = OrthogonalAugmenter._rotate_coco_bbox(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        angle=90,
    )

    assert rotated is not None

    x, y, width, height = rotated

    assert x == 20.0
    assert y == 10.0
    assert width == 40.0
    assert height == 30.0
