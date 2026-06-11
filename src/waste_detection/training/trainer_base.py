from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class TrainerBase(ABC):
    """
    Base interface cho trainer.

    Các trainer cụ thể:
    - DetectorTrainer
    - ClassifierTrainer
    """

    @abstractmethod
    def train(self) -> Any:
        raise NotImplementedError

    def validate(self, *args, **kwargs) -> Dict:
        raise NotImplementedError("Trainer này chưa hỗ trợ validate().")
