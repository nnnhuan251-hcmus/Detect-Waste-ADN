from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDatasetAdapter(ABC):
    """
    Base class cho các adapter chuyển dataset nguồn về format chung.

    Ví dụ:
    - TACO official COCO
    - Roboflow COCO
    - Roboflow YOLO
    """

    @abstractmethod
    def import_dataset(self) -> Dict[str, Any]:
        """
        Import dataset nguồn và trả về report.
        """
        raise NotImplementedError
