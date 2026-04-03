from __future__ import annotations

from pathlib import Path

import numpy as np

from anima_slam_coko.io.submap_store import (
    read_submap,
    write_camera_depth_submap,
    write_rendered_depth_submap,
)
from anima_slam_coko.mapping.compaction import CompactionScheduler
from anima_slam_coko.mapping.gaussian_state import GaussianState
from anima_slam_coko.schemas.submap import SubmapRecord


def _make_record(runtime_mode: str) -> SubmapRecord:
    payload = {
        "agent_id": 0,
        "submap_id": 0,
        "runtime_mode": runtime_mode,
        "keyframe_ids": np.asarray([0, 1], dtype=np.int64),
        "submap_c2ws": np.asarray([np.eye(4), np.eye(4)], dtype=np.float32),
        "gaussian_xyz": np.asarray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        "gaussian_opacity": np.asarray([[0.5], [0.8]], dtype=np.float32),
        "gaussian_scale": np.asarray([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]], dtype=np.float32),
        "gaussian_rotation": np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "gaussian_features": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "descriptor_vector": np.asarray([0.2, 0.4, 0.6], dtype=np.float32),
    }
    if runtime_mode == "rendered_depth":
        payload["rendered_depth"] = np.ones((2, 2), dtype=np.float32)
    else:
        payload["camera_depth"] = np.ones((2, 2), dtype=np.float32)
    return SubmapRecord(**payload)


def test_compaction_window_matches_paper() -> None:
    scheduler = CompactionScheduler(total_iterations=1000)

    assert scheduler.window_bounds() == (700, 950)
    assert scheduler.is_active(699) is False
    assert scheduler.is_active(700) is True
    assert scheduler.is_active(950) is True
    assert scheduler.is_active(951) is False


def test_zero_opacity_pruning_is_covered() -> None:
    state = GaussianState()
    state.add_points(np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32))
    state.opacity = np.asarray([[0.0], [0.7]], dtype=np.float32)

    state.prune_zero_opacity()

    assert state.size == 1
    assert np.allclose(state.opacity, np.asarray([[0.7]], dtype=np.float32))


def test_rendered_and_camera_depth_writers_roundtrip(tmp_path: Path) -> None:
    rendered_record = _make_record("rendered_depth")
    rendered_path = tmp_path / "rendered_depth.json"
    write_rendered_depth_submap(rendered_record, rendered_path)

    camera_record = _make_record("camera_depth")
    camera_path = tmp_path / "camera_depth.json"
    write_camera_depth_submap(camera_record, camera_path)

    rendered_restored = read_submap(rendered_path)
    camera_restored = read_submap(camera_path)

    assert rendered_restored.runtime_mode == "rendered_depth"
    assert camera_restored.runtime_mode == "camera_depth"
