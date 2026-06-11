from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class PredictorBase(ABC):
    """
    Base interface cho các predictor.

    Các predictor cụ thể:
    - DetectorPredictor
    - ClassifierPredictor
    - HybridPredictor
    """

    @abstractmethod
    def predict(self, source: str | Path, **kwargs) -> Any:
        raise NotImplementedError
