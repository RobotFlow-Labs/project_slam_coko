"""Lightweight RGB-D odometry with OpenCV feature tracking."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray


class VisualOdometer:
    def __init__(self, intrinsics: np.ndarray, method_name: str = "point_to_plane") -> None:
        if intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3)")
        if method_name not in {"point_to_plane", "hybrid"}:
            raise ValueError("method_name must be 'point_to_plane' or 'hybrid'")

        self.intrinsics = intrinsics.astype(np.float32, copy=False)
        self.method_name = method_name
        self._last_frame: RGBDFrame | None = None

    def update_last_rgbd(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        self._last_frame = RGBDFrame(rgb=rgb, depth=depth.astype(np.float32, copy=False))

    def estimate_rel_pose(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        init_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        if init_transform is None:
            init_transform = np.eye(4, dtype=np.float32)
        else:
            init_transform = np.asarray(init_transform, dtype=np.float32).copy()

        if self._last_frame is None:
            self.update_last_rgbd(rgb, depth)
            return init_transform

        prev_gray = cv2.cvtColor(self._last_frame.rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=256,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        if points is None or len(points) < 8:
            self.update_last_rgbd(rgb, depth)
            return init_transform

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None)
        valid_prev = points[status[:, 0] == 1]
        valid_curr = next_points[status[:, 0] == 1]

        if len(valid_prev) < 8:
            self.update_last_rgbd(rgb, depth)
            return init_transform

        pixel_shift = np.median(valid_curr - valid_prev, axis=0).reshape(2)
        fx = float(self.intrinsics[0, 0])
        fy = float(self.intrinsics[1, 1])

        prev_depth = self._last_frame.depth[self._last_frame.depth > 0]
        curr_depth = depth[depth > 0]
        prev_depth_median = float(np.median(prev_depth)) if prev_depth.size else 1.0
        curr_depth_median = float(np.median(curr_depth)) if curr_depth.size else prev_depth_median

        rel_transform = init_transform.copy()
        rel_transform[0, 3] = -pixel_shift[0] * prev_depth_median / max(fx, 1e-6)
        rel_transform[1, 3] = -pixel_shift[1] * prev_depth_median / max(fy, 1e-6)
        rel_transform[2, 3] = curr_depth_median - prev_depth_median
        rel_transform[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        self.update_last_rgbd(rgb, depth)
        return rel_transform
