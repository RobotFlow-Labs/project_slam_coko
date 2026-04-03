"""Local keyframe mapping, seeding, and scheduled compaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .compaction import CompactionScheduler
from .gaussian_state import GaussianState


@dataclass(slots=True)
class MappingStep:
    added_points: int
    gaussian_count: int
    optimization_iterations: int


class Mapper:
    def __init__(
        self,
        intrinsics: np.ndarray,
        *,
        iterations: int = 50,
        new_submap_iterations: int = 1000,
        new_submap_points_num: int = 60000,
        prune_ratio: float = 0.05,
    ) -> None:
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self.iterations = iterations
        self.new_submap_iterations = new_submap_iterations
        self.new_submap_points_num = new_submap_points_num
        self.compaction = CompactionScheduler(prune_ratio=prune_ratio)

    def _seed_mask(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120) > 0
        return edges & (depth > 0)

    def _backproject(self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = self._seed_mask(rgb, depth)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        if ys.size > self.new_submap_points_num:
            sample_ids = np.linspace(0, ys.size - 1, self.new_submap_points_num, dtype=int)
            ys = ys[sample_ids]
            xs = xs[sample_ids]

        z = depth[ys, xs]
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]

        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        camera_points = np.stack([x, y, z], axis=1).astype(np.float32)

        pose = np.asarray(pose, dtype=np.float32)
        camera_points_h = np.concatenate(
            [camera_points, np.ones((camera_points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        world_points = (pose @ camera_points_h.T).T[:, :3]
        colors = rgb[ys, xs].astype(np.float32) / 255.0
        return world_points, colors

    def map_keyframe(
        self,
        frame: dict[str, Any],
        pose: np.ndarray,
        state: GaussianState,
        *,
        is_new_submap: bool = False,
    ) -> MappingStep:
        rgb = np.asarray(frame["rgb"], dtype=np.uint8)
        depth = np.asarray(frame["depth"], dtype=np.float32)
        xyz, colors = self._backproject(rgb, depth, pose)
        state.add_points(xyz, colors=colors)

        total_iterations = self.new_submap_iterations if is_new_submap else self.iterations
        opacity = state.opacity.copy()
        for iteration in range(total_iterations):
            opacity = self.compaction.apply(iteration, opacity, total_iterations)
        state.opacity = opacity
        state.prune_zero_opacity()

        return MappingStep(
            added_points=int(xyz.shape[0]),
            gaussian_count=state.size,
            optimization_iterations=total_iterations,
        )
