from __future__ import annotations

import logging
from typing import Any, Dict


logger = logging.getLogger("WandbLogger")


class WandbLogger:
    """
    Wrapper nhẹ cho Weights & Biases.

    Nếu wandb chưa cài hoặc chưa login, code vẫn có thể tắt tracking.
    """

    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str,
        config: Dict[str, Any] | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.group = group
        self.tags = tags or []
        self.run = None

    def start(self):
        if not self.enabled:
            logger.info("W&B disabled.")
            return None

        try:
            import wandb
        except ImportError:
            logger.warning("wandb chưa được cài. Bỏ qua experiment tracking.")
            self.enabled = False
            return None

        self.run = wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.config,
            group=self.group,
            tags=self.tags,
        )

        logger.info("W&B run started: %s", self.run_name)

        return self.run

    def log(self, data: Dict[str, Any]) -> None:
        if self.enabled and self.run is not None:
            self.run.log(data)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()
            logger.info("W&B run finished.")
