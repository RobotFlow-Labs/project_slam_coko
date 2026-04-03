"""Tests for production release validation."""

from __future__ import annotations

from anima_slam_coko.ops.release_checks import (
    check_bandwidth,
    check_depth_l1,
    check_gaussian_count,
    check_loop_detection,
    check_submap_count,
    release_verdict,
    validate_release,
)


def test_depth_l1_pass() -> None:
    r = check_depth_l1(0.48)
    assert r.passed


def test_depth_l1_fail() -> None:
    r = check_depth_l1(2.0)
    assert not r.passed


def test_bandwidth_pass() -> None:
    r = check_bandwidth(100.0)
    assert r.passed


def test_bandwidth_fail() -> None:
    r = check_bandwidth(200.0)
    assert not r.passed


def test_loop_detection_pass() -> None:
    r = check_loop_detection(2, min_loops=1)
    assert r.passed


def test_gaussian_count_pass() -> None:
    r = check_gaussian_count(500000)
    assert r.passed


def test_submap_count_pass() -> None:
    r = check_submap_count(42)
    assert r.passed


def test_validate_release_all_pass() -> None:
    results = {
        "fusion": {
            "intra_loops": 0,
            "inter_loops": 1,
            "post_refine_gaussians": 1850000,
            "total_submaps": 42,
            "bandwidth_per_agent_mb": 100.0,
        },
        "evaluation": {"depth_l1_mean": 0.48},
    }
    checks = validate_release(results)
    passed, msg = release_verdict(checks)
    assert passed
    assert "APPROVED" in msg


def test_validate_release_fails_on_bad_depth() -> None:
    results = {
        "fusion": {
            "intra_loops": 0,
            "inter_loops": 1,
            "post_refine_gaussians": 1850000,
            "total_submaps": 42,
            "bandwidth_per_agent_mb": 100.0,
        },
        "evaluation": {"depth_l1_mean": 5.0},
    }
    checks = validate_release(results)
    passed, msg = release_verdict(checks)
    assert not passed
    assert "BLOCKED" in msg
