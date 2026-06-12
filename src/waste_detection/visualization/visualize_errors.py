from __future__ import annotations

from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ErrorVisualizer:
    """
    Visualization utilities for classification evaluation.

    Supported dataset types:
    - CropClassificationDataset / ImageFolder-like dataset with `.samples`
    - torch.utils.data.Subset wrapping a dataset with `.samples`
    """

    @staticmethod
    def _get_image_path(dataset: Any, index: int) -> Path:
        """
        Safely extract image path from dataset.

        This supports:
        - dataset.samples[index] = (image_path, label)
        - torch.utils.data.Subset(dataset, indices)
        - nested Subset objects
        """
        current_dataset = dataset
        current_index = index

        while hasattr(current_dataset, "dataset") and hasattr(current_dataset, "indices"):
            current_index = int(current_dataset.indices[current_index])
            current_dataset = current_dataset.dataset

        if hasattr(current_dataset, "samples"):
            image_path = current_dataset.samples[current_index][0]
            return Path(image_path)

        raise AttributeError(
            "Dataset không hỗ trợ trích xuất đường dẫn ảnh. "
            "Cần có thuộc tính `.samples` dạng List[(image_path, label)]."
        )

    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: List[List[int]],
        class_names: List[str],
        save_path: str | Path,
    ) -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        matrix = np.asarray(confusion_matrix)

        if matrix.ndim != 2:
            raise ValueError(
                f"confusion_matrix phải là ma trận 2 chiều. Shape hiện tại={matrix.shape}"
            )

        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(matrix, interpolation="nearest", cmap="viridis")

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Confusion Matrix")

        threshold = matrix.max() / 2.0 if matrix.size > 0 else 0.0

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                text_color = "white" if value > threshold else "black"

                ax.text(
                    col_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color=text_color,
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        return save_path

    @staticmethod
    def plot_wrong_predictions(
        dataset: Any,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        save_path: str | Path,
        max_samples: int = 9,
    ) -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"len(y_true) phải bằng len(y_pred). "
                f"Got len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}"
            )

        max_samples = max(1, int(max_samples))

        wrong_indices = [
            index
            for index, (true_label, pred_label) in enumerate(zip(y_true, y_pred))
            if int(true_label) != int(pred_label)
        ]

        wrong_indices = wrong_indices[:max_samples]

        if not wrong_indices:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(
                0.5,
                0.5,
                "No wrong predictions",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            return save_path

        num_images = len(wrong_indices)
        num_cols = min(3, num_images)
        num_rows = int(np.ceil(num_images / num_cols))

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(4 * num_cols, 4 * num_rows),
        )

        axes = np.asarray(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for plot_index, sample_index in enumerate(wrong_indices):
            ax = axes[plot_index]

            image_path = ErrorVisualizer._get_image_path(dataset, sample_index)

            with Image.open(image_path) as image:
                image_rgb = image.convert("RGB")

            true_label = int(y_true[sample_index])
            pred_label = int(y_pred[sample_index])

            ax.imshow(image_rgb)
            ax.set_title(
                f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
                fontsize=10,
                color="red",
            )
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        return save_path
