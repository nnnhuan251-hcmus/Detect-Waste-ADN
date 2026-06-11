from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch


logger = logging.getLogger("CheckpointManager")


class CheckpointManager:
    """
    Quản lý lưu/load checkpoint PyTorch.
    """

    @staticmethod
    def save(
        path: str | Path,
        payload: Dict[str, Any],
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(payload, path)

        logger.info("Đã lưu checkpoint: %s", path)

        return path

    @staticmethod
    def load(
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> Dict[str, Any]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

        checkpoint = torch.load(path, map_location=map_location)

        logger.info("Đã load checkpoint: %s", path)

        return checkpoint
