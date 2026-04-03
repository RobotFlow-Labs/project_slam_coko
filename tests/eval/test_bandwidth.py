"""Tests for bandwidth auditing."""

from __future__ import annotations

import numpy as np

from anima_slam_coko.eval.bandwidth import (
    BandwidthReport,
    compute_bandwidth,
    compare_budget_to_paper,
    estimate_submap_bytes,
)


def _make_submap(agent_id: int, n_gauss: int = 100, n_kf: int = 5) -> dict:
    return {
        "agent_id": agent_id,
        "keyframe_ids": np.arange(n_kf, dtype=np.int64),
        "submap_c2ws": np.tile(np.eye(4, dtype=np.float32), (n_kf, 1, 1)),
        "gaussian_xyz": np.zeros((n_gauss, 3), dtype=np.float32),
        "gaussian_opacity": np.full((n_gauss, 1), 0.5, dtype=np.float32),
        "gaussian_scale": np.ones((n_gauss, 3), dtype=np.float32) * 0.01,
        "gaussian_rotation": np.tile([1, 0, 0, 0], (n_gauss, 1)).astype(np.float32),
        "gaussian_features": np.zeros((n_gauss, 2), dtype=np.float32),
        "descriptor_vector": np.zeros(384, dtype=np.float32),
    }


def test_estimate_submap_bytes_positive() -> None:
    sub = _make_submap(0, n_gauss=100, n_kf=5)
    nbytes = estimate_submap_bytes(sub)
    assert nbytes > 0
    # 100 gaussians * (3+1+3+4+2)*4 = 100*52 = 5200 minimum
    assert nbytes >= 5200


def test_estimate_submap_bytes_with_depth() -> None:
    sub = _make_submap(0)
    sub["camera_depth"] = np.ones((480, 640), dtype=np.float32)
    without_depth = estimate_submap_bytes(_make_submap(0))
    with_depth = estimate_submap_bytes(sub)
    assert with_depth > without_depth
    assert with_depth - without_depth == 480 * 640 * 4


def test_compute_bandwidth_two_agents() -> None:
    agents = {
        0: [_make_submap(0, n_gauss=100), _make_submap(0, n_gauss=200)],
        1: [_make_submap(1, n_gauss=150)],
    }
    report = compute_bandwidth(agents)
    assert isinstance(report, BandwidthReport)
    assert len(report.agents) == 2
    assert report.agents[0].submap_count == 2
    assert report.agents[1].submap_count == 1
    assert report.agents[0].gaussian_count == 300
    assert report.agents[1].gaussian_count == 150
    assert report.total_transmitted_mb > 0


def test_compare_budget_to_paper() -> None:
    agents = {
        0: [_make_submap(0, n_gauss=10)],
        1: [_make_submap(1, n_gauss=10)],
        2: [_make_submap(2, n_gauss=10)],
    }
    report = compute_bandwidth(agents)
    gaps = compare_budget_to_paper("aria_room0", report)
    # Tiny submaps should be well under budget (negative gap)
    for agent_id, gap in gaps.items():
        assert gap < 0
