from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class SchedulerFactory:
    """
    Tạo LR scheduler cho classifier training.
    """

    @staticmethod
    def build_scheduler(
        optimizer: Optimizer,
        scheduler_config: dict,
        total_epochs: int,
    ):
        enabled = bool(scheduler_config.get("enabled", False))
        scheduler_name = scheduler_config.get("name", "constant")

        if not enabled or scheduler_name == "constant":
            return None

        if scheduler_name == "cosine_annealing":
            warmup_config = scheduler_config.get("warmup", {})
            cosine_config = scheduler_config.get("cosine", {})

            warmup_enabled = bool(warmup_config.get("enabled", False))
            warmup_epochs = int(warmup_config.get("warmup_epochs", 0))
            min_lr = float(cosine_config.get("min_lr", 1e-6))

            base_lr = optimizer.param_groups[0]["lr"]
            min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

            def lr_lambda(epoch: int) -> float:
                if warmup_enabled and warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)

                cosine_epoch = epoch - warmup_epochs
                cosine_total = max(1, total_epochs - warmup_epochs)

                cosine_value = 0.5 * (
                    1.0 + math.cos(math.pi * cosine_epoch / cosine_total)
                )

                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value

            return LambdaLR(optimizer, lr_lambda=lr_lambda)

        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
