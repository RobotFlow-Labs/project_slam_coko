"""Paper-style compaction schedule for local submap optimization."""

from __future__ import annotations

import numpy as np


class CompactionScheduler:
    def __init__(
        self,
        *,
        start_ratio: float = 0.7,
        end_ratio: float = 0.95,
        prune_ratio: float = 0.05,
        total_iterations: int = 1000,
    ) -> None:
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.prune_ratio = prune_ratio
        self.total_iterations = total_iterations

    def window_bounds(self, total_iterations: int | None = None) -> tuple[int, int]:
        total = total_iterations or self.total_iterations
        start = int(self.start_ratio * total)
        end = int(self.end_ratio * total)
        return start, end

    def is_active(self, iteration: int, total_iterations: int | None = None) -> bool:
        start, end = self.window_bounds(total_iterations)
        return start <= iteration <= end

    def apply(self, iteration: int, opacity: np.ndarray, total_iterations: int | None = None) -> np.ndarray:
        opacity = np.asarray(opacity, dtype=np.float32).reshape(-1, 1)
        if opacity.size == 0 or not self.is_active(iteration, total_iterations):
            return opacity

        keep_count = max(1, int(round((1.0 - self.prune_ratio) * opacity.shape[0])))
        if keep_count >= opacity.shape[0]:
            return opacity

        flat = opacity[:, 0]
        keep_indices = np.argpartition(flat, -keep_count)[-keep_count:]
        updated = np.zeros_like(flat)
        updated[keep_indices] = flat[keep_indices]
        return updated.reshape(-1, 1)
