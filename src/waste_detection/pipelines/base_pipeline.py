from __future__ import annotations

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """
    Base class cho mọi pipeline trong project.
    """

    @abstractmethod
    def execute(self):
        raise NotImplementedError
