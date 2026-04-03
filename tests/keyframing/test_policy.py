from __future__ import annotations

import numpy as np

from anima_slam_coko.keyframing.policy import KeyframePolicy


def test_keyframe_promotes_on_feature_distance() -> None:
    policy = KeyframePolicy(alpha=0.02)

    prior_features = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.99, 0.01, 0.0], dtype=np.float32),
    ]
    feature = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    assert policy.should_promote(feature, prior_features) is True


def test_keyframe_rejects_when_too_similar() -> None:
    policy = KeyframePolicy(alpha=0.02)

    prior_features = [np.asarray([1.0, 0.0, 0.0], dtype=np.float32)]
    feature = np.asarray([0.9999, 0.01, 0.0], dtype=np.float32)

    assert policy.should_promote(feature, prior_features) is False


def test_submap_rotation_waits_for_minimum_keyframes() -> None:
    policy = KeyframePolicy(alpha=0.02, submapping_threshold=0.05, min_keyframes_per_submap=10)

    submap_anchor = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    divergent_feature = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    not_enough_keyframes = [submap_anchor for _ in range(9)]
    enough_keyframes = [submap_anchor for _ in range(10)]

    assert policy.should_rotate_submap(divergent_feature, submap_anchor, not_enough_keyframes) is False
    assert policy.should_rotate_submap(divergent_feature, submap_anchor, enough_keyframes) is True
