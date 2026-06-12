from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torchvision import models

from waste_detection.utils.io import IOUtils

logger = logging.getLogger("EfficientNetB0Classifier")


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 classifier cho crop dataset 7 class.

    Project scope:
    - Không đổi sang model khác.
    - Dùng Transfer Learning với ImageNet pretrained weights.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes phải > 0. Hiện tại={num_classes}")

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout phải nằm trong [0, 1). Hiện tại={dropout}")

        self.num_classes = num_classes
        self.dropout = dropout
        self.pretrained = pretrained

        try:
            weights = (
                models.EfficientNet_B0_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            self.model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            # Fallback cho torchvision cũ.
            self.model = models.efficientnet_b0(pretrained=pretrained)

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def freeze_backbone(self) -> None:
        """
        Freeze toàn bộ backbone, chỉ train classifier head.
        """
        for parameter in self.model.features.parameters():
            parameter.requires_grad = False

        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

        logger.info("Đã freeze EfficientNet-B0 backbone.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze toàn bộ model.
        """
        for parameter in self.model.parameters():
            parameter.requires_grad = True

        logger.info("Đã unfreeze toàn bộ EfficientNet-B0.")

    def unfreeze_last_blocks(self, num_blocks: int = 2) -> None:
        """
        Unfreeze vài block cuối của EfficientNet-B0.
        Dùng cho gradual fine-tuning.
        """
        self.freeze_backbone()

        feature_blocks = list(self.model.features.children())

        for block in feature_blocks[-num_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

        logger.info("Đã unfreeze %d feature blocks cuối.", num_blocks)

    def count_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def count_total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save_checkpoint(
        self,
        save_path: str | Path,
        class_names: List[str],
        extra: dict | None = None,
    ) -> None:
        save_path = Path(save_path)
        IOUtils.ensure_dir(save_path.parent)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "class_names": class_names,
            "dropout": self.dropout,
            "pretrained": self.pretrained,
        }

        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, save_path)

        logger.info("Đã lưu classifier checkpoint: %s", save_path)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple["EfficientNetB0Classifier", List[str]]:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model = cls(
            num_classes=checkpoint["num_classes"],
            dropout=checkpoint.get("dropout", 0.2),
            pretrained=False,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        class_names = checkpoint.get("class_names", [])

        if not class_names:
            logger.warning(
                "Checkpoint không chứa class_names. Inference vẫn chạy nhưng label name có thể bị thiếu."
            )

        if class_names and len(class_names) != checkpoint["num_classes"]:
            raise ValueError(
                "Checkpoint class_names không khớp num_classes: "
                f"len(class_names)={len(class_names)}, "
                f"num_classes={checkpoint['num_classes']}"
            )

        logger.info("Đã load classifier checkpoint: %s", checkpoint_path)

        return model, class_names
