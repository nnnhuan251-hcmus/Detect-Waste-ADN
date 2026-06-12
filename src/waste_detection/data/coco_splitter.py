import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from waste_detection.data.dataset_utils import DatasetUtils


logger = logging.getLogger("CocoSplitter")


@dataclass
class SplitReport:
    strategy: str
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    num_images: Dict[str, int] = field(default_factory=dict)
    num_annotations: Dict[str, int] = field(default_factory=dict)
    class_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    used_iterative_stratification: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "num_images": self.num_images,
            "num_annotations": self.num_annotations,
            "class_distribution": self.class_distribution,
            "used_iterative_stratification": self.used_iterative_stratification,
        }


class CocoSplitter:
    """
    Chia COCO dataset theo image-level thành train/val/test.

    Ưu tiên:
    - Dùng MultilabelStratifiedShuffleSplit nếu có thư viện iterative-stratification.
    - Nếu không có, fallback sang random split có seed.

    Lý do:
    - Object detection là bài toán multi-label ở cấp ảnh.
    - Một ảnh có thể chứa nhiều class.
    - Split theo annotation riêng lẻ sẽ gây leakage vì cùng ảnh có thể rơi vào nhiều tập.
    """

    def __init__(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
        shuffle: bool = True,
        strategy: str = "multilabel_stratified",
    ) -> None:
        total = train_ratio + val_ratio + test_ratio

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio phải bằng 1. "
                f"Hiện tại = {total}"
            )

        if strategy not in {"multilabel_stratified", "image_level", "random"}:
            raise ValueError(
                "strategy không hợp lệ. "
                "Chỉ hỗ trợ: multilabel_stratified, image_level, random."
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.strategy = strategy

    def split(self, dataset: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], SplitReport]:
        images = list(dataset.get("images", []))

        if not images:
            raise ValueError("Dataset không có images để split.")

        used_iterative = False

        if self.strategy == "multilabel_stratified":
            feature_matrix = self._build_image_class_matrix(dataset)

            try:
                train_indices, val_indices, test_indices = self._iterative_split(
                    images=images,
                    feature_matrix=feature_matrix,
                )
                used_iterative = True
                logger.info("Đã split bằng MultilabelStratifiedShuffleSplit.")

            except Exception as error:
                logger.warning(
                    "Không dùng được MultilabelStratifiedShuffleSplit. "
                    "Fallback sang random image-level split. Lý do: %s",
                    error,
                )
                train_indices, val_indices, test_indices = self._random_split(
                    len(images)
                )

        else:
            logger.info(
                "Split strategy=%s. Dùng random image-level split.",
                self.strategy,
            )
            train_indices, val_indices, test_indices = self._random_split(len(images))

        train_images = [images[index] for index in train_indices]
        val_images = [images[index] for index in val_indices]
        test_images = [images[index] for index in test_indices]

        splits = {
            "train": DatasetUtils.create_coco_subset(dataset, train_images),
            "val": DatasetUtils.create_coco_subset(dataset, val_images),
            "test": DatasetUtils.create_coco_subset(dataset, test_images),
        }

        report = self._build_report(
            splits=splits,
            dataset=dataset,
            used_iterative_stratification=used_iterative,
        )

        logger.info(
            "Split xong: strategy=%s | train=%d images, val=%d images, test=%d images.",
            self.strategy,
            len(train_images),
            len(val_images),
            len(test_images),
        )

        return splits, report

    def _build_image_class_matrix(self, dataset: Dict[str, Any]) -> np.ndarray:
        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])
        categories = dataset.get("categories", [])

        category_ids = [category["id"] for category in categories]
        category_id_to_col = {
            category_id: col_index for col_index, category_id in enumerate(category_ids)
        }

        image_id_to_row = {
            image["id"]: row_index for row_index, image in enumerate(images)
        }

        matrix = np.zeros((len(images), len(categories)), dtype=np.int32)

        for annotation in annotations:
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")

            if image_id not in image_id_to_row:
                continue

            if category_id not in category_id_to_col:
                continue

            row = image_id_to_row[image_id]
            col = category_id_to_col[category_id]
            matrix[row, col] = 1

        return matrix

    def _iterative_split(
        self,
        images: List[Dict[str, Any]],
        feature_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        indices = np.arange(len(images))
        dummy_x = np.zeros((len(images), 1))

        temp_ratio = self.val_ratio + self.test_ratio

        first_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=temp_ratio,
            random_state=self.seed,
        )

        train_indices, temp_indices = next(
            first_splitter.split(dummy_x, feature_matrix)
        )

        temp_feature_matrix = feature_matrix[temp_indices]
        temp_dummy_x = np.zeros((len(temp_indices), 1))

        relative_test_ratio = self.test_ratio / temp_ratio

        second_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=relative_test_ratio,
            random_state=self.seed,
        )

        val_relative_indices, test_relative_indices = next(
            second_splitter.split(temp_dummy_x, temp_feature_matrix)
        )

        val_indices = temp_indices[val_relative_indices]
        test_indices = temp_indices[test_relative_indices]

        return train_indices, val_indices, test_indices

    def _random_split(self, num_images: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.arange(num_images)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        train_end = int(round(num_images * self.train_ratio))
        val_end = train_end + int(round(num_images * self.val_ratio))

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        return train_indices, val_indices, test_indices

    def _build_report(
        self,
        splits: Dict[str, Dict[str, Any]],
        dataset: Dict[str, Any],
        used_iterative_stratification: bool,
    ) -> SplitReport:
        report = SplitReport(
            strategy=self.strategy,
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            used_iterative_stratification=used_iterative_stratification,
        )

        for split_name, split_dataset in splits.items():
            report.num_images[split_name] = len(split_dataset.get("images", []))
            report.num_annotations[split_name] = len(
                split_dataset.get("annotations", [])
            )
            report.class_distribution[split_name] = self._count_annotations_by_class(
                split_dataset
            )

        report.class_distribution["full_dataset"] = self._count_annotations_by_class(
            dataset
        )

        return report

    @staticmethod
    def _count_annotations_by_class(dataset: Dict[str, Any]) -> Dict[str, int]:
        categories = dataset.get("categories", [])
        annotations = dataset.get("annotations", [])

        category_id_to_name = {
            category["id"]: category["name"] for category in categories
        }

        counter = Counter()

        for annotation in annotations:
            category_id = annotation.get("category_id")
            category_name = category_id_to_name.get(category_id, f"unknown_{category_id}")
            counter[category_name] += 1

        return dict(sorted(counter.items()))
