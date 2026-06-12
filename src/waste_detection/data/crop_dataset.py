from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


logger = logging.getLogger("CropClassificationDataset")


class CropClassificationDataset(Dataset):
    """
    Dataset cho EfficientNet-B0 crop classifier.

    Cấu trúc thư mục mong đợi:

    crops_7class/
    ├── train/
    │   ├── plastic/
    │   ├── paper/
    │   ├── metal/
    │   ├── glass/
    │   ├── organic/
    │   ├── cigarette/
    │   └── other/
    ├── val/
    └── test/
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root_dir: str | Path,
        class_names: List[str],
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_names = list(class_names)
        self.transform = transform
        self.return_path = return_path

        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }
        self.idx_to_class = {
            index: class_name for class_name, index in self.class_to_idx.items()
        }

        self.samples: List[Tuple[Path, int]] = []
        self._scan_dataset()

        if not self.samples:
            raise ValueError(f"Không tìm thấy ảnh crop hợp lệ trong: {self.root_dir}")

        logger.info(
            "Loaded crop dataset: %s | %d samples | classes=%s",
            self.root_dir,
            len(self.samples),
            self.class_names,
        )

    def _scan_dataset(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Crop dataset folder không tồn tại: {self.root_dir}")

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                logger.warning("Không tìm thấy folder class: %s", class_dir)
                continue

            if not class_dir.is_dir():
                logger.warning("Đường dẫn class không phải folder: %s", class_dir)
                continue

            label = self.class_to_idx[class_name]

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, str(image_path)

        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        label_counter = Counter(label for _, label in self.samples)

        return {
            class_name: label_counter.get(index, 0)
            for index, class_name in enumerate(self.class_names)
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Tính class weights cho CrossEntropy/FocalLoss.

        Công thức:
            weight_c = total_samples / (num_classes * count_c)

        Nếu một class không có sample, weight = 0 để tránh chia cho 0.
        """
        counts = self.get_class_counts()
        total_samples = len(self.samples)
        num_classes = len(self.class_names)

        weights = []

        for class_name in self.class_names:
            count = counts.get(class_name, 0)

            if count == 0:
                weights.append(0.0)
            else:
                weights.append(total_samples / (num_classes * count))

        return torch.tensor(weights, dtype=torch.float32)
