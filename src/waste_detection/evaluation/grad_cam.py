from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


class GradCAMExplainer:
    """
    Grad-CAM wrapper cho EfficientNet-B0 classifier.

    File này phụ thuộc package pytorch_grad_cam.
    Nếu chưa cài, hãy thêm vào requirements:
        grad-cam
    """

    def __init__(self, model: torch.nn.Module, target_layer) -> None:
        try:
            from pytorch_grad_cam import GradCAM
        except ImportError as error:
            raise ImportError(
                "Thiếu package grad-cam. Hãy cài bằng: pip install grad-cam"
            ) from error

        self.model = model
        self.model.eval()
        self.target_layers = [target_layer]
        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers,
        )

    def explain(
        self,
        input_tensor: torch.Tensor,
        original_image_rgb: np.ndarray,
        target_class: int | None,
        save_path: str | Path,
    ) -> Path:
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        targets = (
            [ClassifierOutputTarget(target_class)]
            if target_class is not None
            else None
        )

        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
        )

        grayscale_cam = grayscale_cam[0, :]

        image_float = np.float32(original_image_rgb) / 255.0

        visualization = show_cam_on_image(
            image_float,
            grayscale_cam,
            use_rgb=True,
        )

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(original_image_rgb)
        axes[0].set_title("Original Crop")
        axes[0].axis("off")

        axes[1].imshow(visualization)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        return save_path

    @staticmethod
    def get_efficientnet_last_conv_layer(model: torch.nn.Module):
        """
        Lấy block conv cuối cho EfficientNet-B0 wrapper.
        """
        if hasattr(model, "model") and hasattr(model.model, "features"):
            return model.model.features[-1]

        if hasattr(model, "features"):
            return model.features[-1]

        raise ValueError("Không tìm thấy EfficientNet features để dùng Grad-CAM.")
