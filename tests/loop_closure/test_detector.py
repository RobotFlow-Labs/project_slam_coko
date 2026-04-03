"""Tests for loop closure detector — inter/intra agent loop detection."""

from __future__ import annotations

import numpy as np

from anima_slam_coko.loop_closure.detector import LoopCandidate, LoopDetector


def _make_submap(
    agent_id: int,
    submap_id: int,
    frame_id: int,
    feature_vector: np.ndarray,
    pose_offset: float = 0.0,
) -> dict:
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = pose_offset
    return {
        "agent_id": agent_id,
        "submap_id": submap_id,
        "keyframe_ids": np.array([frame_id], dtype=np.int64),
        "submap_c2ws": np.stack([c2w]),
        "submap_features": feature_vector.astype(np.float32),
        "gaussian_xyz": np.zeros((5, 3), dtype=np.float32),
        "gaussian_opacity": np.full((5, 1), 0.5, dtype=np.float32),
        "gaussian_scale": np.ones((5, 3), dtype=np.float32) * 0.01,
        "gaussian_rotation": np.tile([1, 0, 0, 0], (5, 1)).astype(np.float32),
        "gaussian_features": np.zeros((5, 2), dtype=np.float32),
        "submap_start_frame_id": frame_id,
    }


def test_inter_loop_detection_finds_similar_features() -> None:
    """Two agents with identical features should produce an inter-agent loop."""
    feat = np.array([1.0, 0.0, 0.0] * 128, dtype=np.float32)  # 384-d
    agents_submaps = {
        0: [_make_submap(0, 0, 0, feat, pose_offset=0.0)],
        1: [_make_submap(1, 0, 100, feat, pose_offset=1.0)],
    }
    detector = LoopDetector(embed_size=384, feature_dist_threshold=0.5)
    _, inter = detector.detect(agents_submaps)
    assert len(inter) >= 1
    assert inter[0].is_inter_agent


def test_intra_loop_detection_with_revisit() -> None:
    """Agent 0 visits similar place twice — should detect an intra-agent loop."""
    feat_a = np.random.RandomState(42).randn(384).astype(np.float32)
    feat_a /= np.linalg.norm(feat_a)
    feat_b = feat_a + np.random.RandomState(43).randn(384).astype(np.float32) * 0.01
    feat_b /= np.linalg.norm(feat_b)

    agents_submaps = {
        0: [
            _make_submap(0, 0, 0, feat_a, 0.0),
            _make_submap(0, 1, 10, feat_b, 1.0),
        ],
    }
    detector = LoopDetector(
        embed_size=384,
        feature_dist_threshold=0.5,
        time_threshold=0,
    )
    intra, _ = detector.detect(agents_submaps)
    assert len(intra) >= 1
    assert not intra[0].is_inter_agent


def test_no_loops_when_features_are_dissimilar() -> None:
    """Orthogonal features should produce no loops."""
    feat_a = np.zeros(384, dtype=np.float32)
    feat_a[0] = 1.0
    feat_b = np.zeros(384, dtype=np.float32)
    feat_b[1] = 1.0

    agents_submaps = {
        0: [_make_submap(0, 0, 0, feat_a)],
        1: [_make_submap(1, 0, 100, feat_b)],
    }
    detector = LoopDetector(embed_size=384, feature_dist_threshold=0.05)
    intra, inter = detector.detect(agents_submaps)
    assert len(intra) == 0
    assert len(inter) == 0


def test_filter_removes_low_fitness_loops() -> None:
    detector = LoopDetector(fitness_threshold=0.5, inlier_rmse_threshold=0.1)
    good = LoopCandidate(0, 0, 1, 100, fitness=0.8, inlier_rmse=0.05)
    bad_fitness = LoopCandidate(0, 0, 1, 100, fitness=0.2, inlier_rmse=0.05)
    bad_rmse = LoopCandidate(0, 0, 1, 100, fitness=0.8, inlier_rmse=0.5)
    filtered = detector.filter_loops([good, bad_fitness, bad_rmse])
    assert len(filtered) == 1
    assert filtered[0] is good
