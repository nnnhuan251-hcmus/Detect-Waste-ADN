from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Dùng cho Run 2/3 để giảm ảnh hưởng class dễ và tập trung vào class khó/ít mẫu.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction phải là: mean, sum hoặc none")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()

        if self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss


class LossFactory:
    """
    Tạo loss function từ experiment config.
    """

    @staticmethod
    def build_classification_loss(
        loss_config: dict,
        class_weights: torch.Tensor | None,
        device: torch.device,
    ) -> nn.Module:
        classifier_loss = str(
            loss_config.get("classifier_loss", "cross_entropy")
        ).lower().strip()

        use_class_weighting = bool(loss_config.get("class_weighting", False))

        weight_tensor = None

        if use_class_weighting and class_weights is not None:
            weight_tensor = class_weights.to(device)

        if classifier_loss == "cross_entropy":
            return nn.CrossEntropyLoss(weight=weight_tensor)

        if classifier_loss == "focal_loss":
            focal_config = loss_config.get("focal_loss", {})
            gamma = float(focal_config.get("gamma", 2.0))
            reduction = str(focal_config.get("reduction", "mean")).lower().strip()

            return FocalLoss(
                alpha=weight_tensor,
                gamma=gamma,
                reduction=reduction,
            )

        raise ValueError(f"Unsupported classifier loss: {classifier_loss}")
