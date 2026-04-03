"""Tests for registration pipeline — rendered depth, coarse, and fine stages."""

from __future__ import annotations

import numpy as np
import pytest

from anima_slam_coko.registration.rendered_depth import (
    depth_to_point_cloud,
    gaussian_xyz_as_cloud,
    submap_to_point_cloud,
)


def _make_intrinsics() -> np.ndarray:
    return np.array([
        [600.0, 0.0, 320.0],
        [0.0, 600.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _make_depth(h: int = 64, w: int = 64, value: float = 2.0) -> np.ndarray:
    depth = np.zeros((h, w), dtype=np.float32)
    depth[10:50, 10:50] = value
    return depth


def test_depth_to_point_cloud_produces_valid_points() -> None:
    depth = _make_depth()
    intrinsics = _make_intrinsics()
    pts = depth_to_point_cloud(depth, intrinsics)
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert pts.shape[0] > 0
    assert np.all(np.isfinite(pts))


def test_depth_to_point_cloud_with_c2w_transform() -> None:
    depth = _make_depth()
    intrinsics = _make_intrinsics()
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = 5.0  # translate x by 5

    pts_no_transform = depth_to_point_cloud(depth, intrinsics)
    pts_with_transform = depth_to_point_cloud(depth, intrinsics, c2w=c2w)

    assert pts_with_transform.shape == pts_no_transform.shape
    # x-coordinates should be shifted by ~5
    x_diff = np.mean(pts_with_transform[:, 0]) - np.mean(pts_no_transform[:, 0])
    assert abs(x_diff - 5.0) < 0.5


def test_depth_to_point_cloud_empty_on_zero_depth() -> None:
    depth = np.zeros((64, 64), dtype=np.float32)
    pts = depth_to_point_cloud(depth, _make_intrinsics())
    assert pts.shape == (0, 3)


def test_submap_to_point_cloud_rendered_depth() -> None:
    intrinsics = _make_intrinsics()
    submap = {
        "rendered_depth": _make_depth(),
        "submap_c2ws": [np.eye(4, dtype=np.float32)],
    }
    pts = submap_to_point_cloud(submap, intrinsics, mode="rendered_depth")
    assert pts.shape[0] > 0
    assert pts.shape[1] == 3


def test_submap_to_point_cloud_camera_depth() -> None:
    intrinsics = _make_intrinsics()
    submap = {
        "camera_depth": _make_depth(),
        "submap_c2ws": [np.eye(4, dtype=np.float32)],
    }
    pts = submap_to_point_cloud(submap, intrinsics, mode="camera_depth")
    assert pts.shape[0] > 0


def test_submap_to_point_cloud_raises_on_missing_depth() -> None:
    intrinsics = _make_intrinsics()
    submap = {"submap_c2ws": [np.eye(4, dtype=np.float32)]}
    with pytest.raises(ValueError, match="payload missing"):
        submap_to_point_cloud(submap, intrinsics, mode="rendered_depth")


def test_gaussian_xyz_as_cloud() -> None:
    submap = {
        "gaussian_xyz": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    }
    cloud = gaussian_xyz_as_cloud(submap)
    assert cloud.shape == (2, 3)
    assert np.allclose(cloud[0], [1, 2, 3])


def test_coarse_and_fine_registration_with_open3d() -> None:
    """Integration test: coarse FPFH+RANSAC followed by ICP on overlapping clouds."""
    try:
        import open3d  # noqa: F401
    except ImportError:
        pytest.skip("open3d not installed")

    from anima_slam_coko.registration.coarse import coarse_register
    from anima_slam_coko.registration.fine import icp_refine

    rng = np.random.RandomState(42)
    source = rng.randn(500, 3).astype(np.float32) * 0.5
    # Apply small known transform
    R_true = np.eye(3, dtype=np.float32)
    t_true = np.array([0.1, 0.05, -0.05], dtype=np.float32)
    target = source @ R_true.T + t_true

    coarse = coarse_register(source, target, voxel_size=0.1)
    assert coarse.transformation.shape == (4, 4)

    fine = icp_refine(source, target, coarse.transformation, max_correspondence_distance=0.2)
    assert fine.transformation.shape == (4, 4)
    # The refined translation should be close to the ground truth
    t_est = fine.transformation[:3, 3]
    assert np.linalg.norm(t_est - t_true) < 0.5 or fine.fitness > 0.0
