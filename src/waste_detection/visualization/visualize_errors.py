from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ErrorVisualizer:
    """
    Visualization utilities for classification evaluation.
    """

    @staticmethod
    def _get_image_path(dataset, index: int) -> Path | str:
        """
        Trích xuất đường dẫn ảnh an toàn, hỗ trợ cả torchvision ImageFolder 
        và torch.utils.data.Subset (thường gặp khi split train/val).
        """
        if hasattr(dataset, "samples"):
            return dataset.samples[index][0]
        elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            # Xử lý trường hợp dataset bị bọc bởi Subset
            real_index = dataset.indices[index]
            return dataset.dataset.samples[real_index][0]
        else:
            raise AttributeError(
                "Dataset không hỗ trợ trích xuất đường dẫn ảnh. "
                "Cần có thuộc tính '.samples' (như ImageFolder)."
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

        fig, ax = plt.subplots(figsize=(9, 8))
        # Thêm cmap='viridis' hoặc 'Blues' để màu sắc hiện đại hơn
        im = ax.imshow(matrix, interpolation="nearest", cmap="viridis")

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Confusion Matrix")

        # Đổi màu chữ để dễ đọc
        threshold = matrix.max() / 2 if matrix.size > 0 else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text_color = "white" if val > threshold else "black"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color=text_color,
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        return save_path

    @staticmethod
    def plot_wrong_predictions(
        dataset,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        save_path: str | Path,
        max_samples: int = 9,
    ) -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        wrong_indices = [
            index
            for index, (true_label, pred_label) in enumerate(zip(y_true, y_pred))
            if int(true_label) != int(pred_label)
        ]

        wrong_indices = wrong_indices[:max_samples]

        # Xử lý an toàn khi mô hình dự đoán đúng 100%
        if not wrong_indices:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(
                0.5,
                0.5,
                "No wrong predictions 🎉",
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

        # Chuyển đổi axes thành mảng 1D để dễ lặp qua (kể cả khi chỉ có 1 ảnh)
        axes = np.asarray(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for plot_index, sample_index in enumerate(wrong_indices):
            ax = axes[plot_index]

            # Sử dụng hàm helper an toàn thay vì truy cập trực tiếp
            image_path = ErrorVisualizer._get_image_path(dataset, sample_index)
            image = Image.open(image_path).convert("RGB")

            true_label = int(y_true[sample_index])
            pred_label = int(y_pred[sample_index])

            ax.imshow(image)
            
            # Sử dụng màu đỏ cho tiêu đề để nhấn mạnh đây là lỗi
            ax.set_title(
                f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
                fontsize=10,
                color="red",
            )

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        return save_path
