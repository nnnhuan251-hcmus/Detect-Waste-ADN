from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


class ErrorVisualizer:
    """
    Trực quan hóa lỗi classifier.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: List[List[int]],
        class_names: List[str],
        save_path: str | Path,
    ) -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        matrix = np.array(confusion_matrix)

        plt.figure(figsize=(9, 7))
        plt.imshow(matrix, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)

        threshold = matrix.max() / 2 if matrix.size > 0 else 0

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text_color = "white" if value > threshold else "black"
                plt.text(
                    col,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color=text_color,
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        return save_path

    @staticmethod
    def plot_wrong_predictions(
        dataset,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        save_path: str | Path,
        max_samples: int = 9,
    ) -> Path | None:
        wrong_indices = [
            index for index, (true_id, pred_id) in enumerate(zip(y_true, y_pred))
            if true_id != pred_id
        ]

        if not wrong_indices:
            return None

        selected_indices = wrong_indices[:max_samples]

        cols = 3
        rows = int(np.ceil(len(selected_indices) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))

        if rows == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for plot_index, dataset_index in enumerate(selected_indices):
            sample = dataset[dataset_index]
            image_tensor = sample[0]

            image_np = ErrorVisualizer._tensor_to_image(image_tensor)

            true_label = class_names[y_true[dataset_index]]
            pred_label = class_names[y_pred[dataset_index]]

            axes[plot_index].imshow(image_np)
            axes[plot_index].set_title(
                f"True: {true_label}\nPred: {pred_label}",
                color="red",
                fontsize=10,
            )
            axes[plot_index].axis("off")

        for empty_index in range(len(selected_indices), len(axes)):
            axes[empty_index].axis("off")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        return save_path

    @staticmethod
    def _tensor_to_image(image_tensor: torch.Tensor):
        image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        image_np = ErrorVisualizer.IMAGENET_STD * image_np + ErrorVisualizer.IMAGENET_MEAN
        image_np = np.clip(image_np, 0.0, 1.0)
        return image_np
