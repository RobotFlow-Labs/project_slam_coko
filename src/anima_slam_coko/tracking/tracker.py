"""Tracker initialization from odometry plus optional Gaussian-map pose refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .visual_odometer import VisualOdometer


class GaussianPoseRefiner(Protocol):
    def refine_pose(
        self,
        initial_pose: np.ndarray,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        w_color_loss: float,
    ) -> np.ndarray: ...


@dataclass(slots=True)
class TrackerConfig:
    odometry_type: str = "odometer"
    odometer_method: str = "point_to_plane"
    w_color_loss: float = 0.95
    debug_fallback: bool = False


class Tracker:
    def __init__(self, intrinsics: np.ndarray, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.odometer = VisualOdometer(intrinsics, self.config.odometer_method)

    def _constant_speed_pose(self, prev_c2ws: np.ndarray) -> np.ndarray:
        if prev_c2ws.shape[0] >= 2:
            return prev_c2ws[-1] @ np.linalg.inv(prev_c2ws[-2]) @ prev_c2ws[-1]
        return prev_c2ws[-1].copy()

    def _normalize_pose(self, pose: np.ndarray) -> np.ndarray:
        normalized = np.asarray(pose, dtype=np.float32).copy()
        normalized[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return normalized

    def _initial_pose(self, frame: dict[str, Any], prev_c2ws: np.ndarray) -> np.ndarray:
        if self.config.odometry_type == "const_speed":
            return self._constant_speed_pose(prev_c2ws)

        if self.config.odometry_type == "odometer":
            rel_pose = self.odometer.estimate_rel_pose(frame["rgb"], frame["depth"])
            return prev_c2ws[-1] @ rel_pose

        if self.config.odometry_type == "gt" and "gt_c2w" in frame:
            return np.asarray(frame["gt_c2w"], dtype=np.float32)

        return prev_c2ws[-1].copy()

    def track(
        self,
        frame: dict[str, Any],
        gaussian_state: GaussianPoseRefiner | None,
        prev_c2ws: np.ndarray,
    ) -> np.ndarray:
        prev_c2ws = np.asarray(prev_c2ws, dtype=np.float32)
        try:
            initial_pose = self._initial_pose(frame, prev_c2ws)
            if gaussian_state is None:
                return self._normalize_pose(initial_pose)

            refined_pose = gaussian_state.refine_pose(
                initial_pose,
                rgb=frame["rgb"],
                depth=frame["depth"],
                w_color_loss=self.config.w_color_loss,
            )
            return self._normalize_pose(refined_pose)
        except Exception:
            if not self.config.debug_fallback:
                raise
            return self._normalize_pose(self._constant_speed_pose(prev_c2ws))
