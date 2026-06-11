from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """
    Early stopping theo metric cần maximize hoặc minimize.
    """

    patience: int
    mode: str = "max"
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode phải là 'max' hoặc 'min'.")

        self.best_score: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return True

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return False
