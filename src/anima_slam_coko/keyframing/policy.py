"""Feature-distance policies for keyframing and submap rotation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_feature(feature: np.ndarray | Sequence[float]) -> np.ndarray:
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Feature vectors must have non-zero norm")
    return vector / norm


class KeyframePolicy:
    def __init__(
        self,
        *,
        alpha: float = 0.02,
        submapping_threshold: float = 0.05,
        min_keyframes_per_submap: int = 10,
    ) -> None:
        self.alpha = alpha
        self.submapping_threshold = submapping_threshold
        self.min_keyframes_per_submap = min_keyframes_per_submap

    def cosine_distance(
        self, feature_a: np.ndarray | Sequence[float], feature_b: np.ndarray | Sequence[float]
    ) -> float:
        normalized_a = _as_feature(feature_a)
        normalized_b = _as_feature(feature_b)
        return float(1.0 - np.dot(normalized_a, normalized_b))

    def min_feature_distance(
        self,
        feature: np.ndarray | Sequence[float],
        prior_features: Sequence[np.ndarray | Sequence[float]],
    ) -> float:
        if not prior_features:
            return float("inf")
        distances = [self.cosine_distance(feature, prior) for prior in prior_features]
        return float(min(distances))

    def should_promote(
        self,
        feature: np.ndarray | Sequence[float],
        prior_features: Sequence[np.ndarray | Sequence[float]],
    ) -> bool:
        return self.min_feature_distance(feature, prior_features) >= self.alpha

    def should_rotate_submap(
        self,
        feature: np.ndarray | Sequence[float],
        submap_anchor_feature: np.ndarray | Sequence[float],
        keyframe_features: Sequence[np.ndarray | Sequence[float]],
    ) -> bool:
        if len(keyframe_features) < self.min_keyframes_per_submap:
            return False
        distance = self.cosine_distance(feature, submap_anchor_feature)
        return distance >= self.submapping_threshold
