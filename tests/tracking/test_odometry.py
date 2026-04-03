from __future__ import annotations

import numpy as np

from anima_slam_coko.tracking.tracker import Tracker, TrackerConfig
from anima_slam_coko.tracking.visual_odometer import VisualOdometer


def _make_intrinsics() -> np.ndarray:
    return np.asarray(
        [
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _make_frame(shift: int = 0) -> dict[str, np.ndarray]:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[20:30, 20 + shift : 30 + shift] = 255
    depth = np.ones((64, 64), dtype=np.float32)
    return {"rgb": rgb, "depth": depth}


def test_odometry_returns_homogeneous_transform() -> None:
    odometer = VisualOdometer(_make_intrinsics())
    first = _make_frame(shift=0)
    second = _make_frame(shift=2)

    odometer.update_last_rgbd(first["rgb"], first["depth"])
    transform = odometer.estimate_rel_pose(second["rgb"], second["depth"])

    assert transform.shape == (4, 4)
    assert np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_tracker_returns_homogeneous_pose() -> None:
    tracker = Tracker(_make_intrinsics(), TrackerConfig())
    frame0 = _make_frame(shift=0)
    frame1 = _make_frame(shift=2)

    tracker.odometer.update_last_rgbd(frame0["rgb"], frame0["depth"])
    prev_c2ws = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
    pose = tracker.track(frame1, gaussian_state=None, prev_c2ws=prev_c2ws)

    assert pose.shape == (4, 4)
    assert np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def test_tracker_falls_back_cleanly_in_debug_mode() -> None:
    tracker = Tracker(
        _make_intrinsics(),
        TrackerConfig(odometry_type="odometer", debug_fallback=True),
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("forced failure")

    tracker.odometer.estimate_rel_pose = _raise  # type: ignore[method-assign]

    prev_pose0 = np.eye(4, dtype=np.float32)
    prev_pose1 = np.eye(4, dtype=np.float32)
    prev_pose1[0, 3] = 1.0
    prev_c2ws = np.stack([prev_pose0, prev_pose1])
    pose = tracker.track(_make_frame(shift=4), gaussian_state=None, prev_c2ws=prev_c2ws)

    assert pose.shape == (4, 4)
    assert np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
