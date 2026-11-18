"""Timing utilities for training and inference benchmarking.

Owner: Member C
Phase 1: Shared harness
"""

import time
from contextlib import contextmanager
from typing import Generator


class Timer:
    """Context manager for measuring elapsed time in seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self.start_time: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self.start_time

    @property
    def seconds(self) -> float:
        return self.elapsed


@contextmanager
def time_block() -> Generator[Timer, None, None]:
    """Convenience context manager yielding a Timer instance."""
    timer = Timer()
    with timer:
        yield timer
