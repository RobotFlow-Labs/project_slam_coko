"""Minimal Gaussian state container for local mapping and checkpoint export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _empty(shape: tuple[int, ...]) -> np.ndarray:
    return np.empty(shape, dtype=np.float32)


@dataclass(slots=True)
class GaussianState:
    xyz: np.ndarray = field(default_factory=lambda: _empty((0, 3)))
    opacity: np.ndarray = field(default_factory=lambda: _empty((0, 1)))
    scale: np.ndarray = field(default_factory=lambda: _empty((0, 3)))
    rotation: np.ndarray = field(default_factory=lambda: _empty((0, 4)))
    features: np.ndarray = field(default_factory=lambda: _empty((0, 2)))
    colors: np.ndarray = field(default_factory=lambda: _empty((0, 3)))

    @property
    def size(self) -> int:
        return int(self.xyz.shape[0])

    def add_points(
        self,
        xyz: np.ndarray,
        *,
        colors: np.ndarray | None = None,
        features: np.ndarray | None = None,
        opacity: float = 0.5,
    ) -> None:
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if xyz.size == 0:
            return

        count = xyz.shape[0]
        colors = (
            np.asarray(colors, dtype=np.float32).reshape(count, 3)
            if colors is not None
            else np.zeros((count, 3), dtype=np.float32)
        )
        features = (
            np.asarray(features, dtype=np.float32).reshape(count, -1)
            if features is not None
            else np.zeros((count, 2), dtype=np.float32)
        )

        self.xyz = np.vstack([self.xyz, xyz])
        self.colors = np.vstack([self.colors, colors])
        self.features = np.vstack([self.features, features])
        self.opacity = np.vstack([self.opacity, np.full((count, 1), opacity, dtype=np.float32)])
        self.scale = np.vstack([self.scale, np.ones((count, 3), dtype=np.float32) * 0.01])
        rotations = np.zeros((count, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        self.rotation = np.vstack([self.rotation, rotations])

    def prune_mask(self, prune_mask: np.ndarray) -> None:
        prune_mask = np.asarray(prune_mask, dtype=bool).reshape(-1)
        if prune_mask.shape[0] != self.size:
            raise ValueError("prune_mask must match the Gaussian count")
        keep_mask = ~prune_mask
        self.xyz = self.xyz[keep_mask]
        self.opacity = self.opacity[keep_mask]
        self.scale = self.scale[keep_mask]
        self.rotation = self.rotation[keep_mask]
        self.features = self.features[keep_mask]
        self.colors = self.colors[keep_mask]

    def prune_zero_opacity(self) -> None:
        if self.size == 0:
            return
        self.prune_mask((self.opacity[:, 0] <= 0.0))

    def capture_dict(self) -> dict[str, Any]:
        return {
            "xyz": self.xyz.copy(),
            "opacity": self.opacity.copy(),
            "scale": self.scale.copy(),
            "rotation": self.rotation.copy(),
            "features": self.features.copy(),
            "colors": self.colors.copy(),
        }
