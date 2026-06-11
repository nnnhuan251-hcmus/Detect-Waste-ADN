from __future__ import annotations

import pytest

from waste_detection.data.label_mapper import TacoLabelMapper


def _build_sample_coco():
    return {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "batch_1/000001.jpg",
                "width": 640,
                "height": 480,
            }
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 100,
                "bbox": [10, 20, 50, 60],
                "area": 3000,
                "iscrowd": 0,
            },
            {
                "id": 11,
                "image_id": 1,
                "category_id": 101,
                "bbox": [100, 120, 30, 40],
                "area": 1200,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {
                "id": 100,
                "name": "Clear plastic bottle",
                "supercategory": "trash",
            },
            {
                "id": 101,
                "name": "Normal paper",
                "supercategory": "trash",
            },
        ],
    }


def test_label_mapper_maps_original_categories_to_target_ids():
    dataset = _build_sample_coco()

    class_names = [
        "plastic",
        "paper",
        "metal",
        "glass",
        "organic",
        "cigarette",
        "other",
    ]

    name_to_id = {class_name: index for index, class_name in enumerate(class_names)}

    mapping_dict = {
        "plastic": ["Clear plastic bottle"],
        "paper": ["Normal paper"],
        "metal": [],
        "glass": [],
        "organic": [],
        "cigarette": [],
        "other": [],
        "ignore": [],
    }

    mapper = TacoLabelMapper(
        target_class_names=class_names,
        target_name_to_id=name_to_id,
        ignore_key="ignore",
        fail_on_unmapped=True,
    )

    mapped_dataset, report = mapper.transform(
        dataset=dataset,
        mapping_dict=mapping_dict,
    )

    assert len(mapped_dataset["categories"]) == 7
    assert mapped_dataset["categories"][0]["name"] == "plastic"
    assert mapped_dataset["categories"][1]["name"] == "paper"

    category_ids = [ann["category_id"] for ann in mapped_dataset["annotations"]]

    assert category_ids == [0, 1]
    assert report.num_unmapped_categories == 0
    assert report.num_ignored_annotations == 0


def test_label_mapper_raises_on_unmapped_category_when_strict():
    dataset = _build_sample_coco()

    class_names = [
        "plastic",
        "paper",
        "metal",
        "glass",
        "organic",
        "cigarette",
        "other",
    ]

    name_to_id = {class_name: index for index, class_name in enumerate(class_names)}

    mapping_dict = {
        "plastic": ["Clear plastic bottle"],
        "paper": [],
        "metal": [],
        "glass": [],
        "organic": [],
        "cigarette": [],
        "other": [],
        "ignore": [],
    }

    mapper = TacoLabelMapper(
        target_class_names=class_names,
        target_name_to_id=name_to_id,
        ignore_key="ignore",
        fail_on_unmapped=True,
    )

    with pytest.raises(KeyError):
        mapper.transform(
            dataset=dataset,
            mapping_dict=mapping_dict,
        )
