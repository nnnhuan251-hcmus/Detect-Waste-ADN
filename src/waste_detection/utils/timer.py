from __future__ import annotations

import time
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    """
    Simple timer dùng cho inference time / profiling.
    """

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.elapsed_seconds: float | None = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed_seconds = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()

        if self.start_time is not None:
            self.elapsed_seconds = self.end_time - self.start_time

        return False
