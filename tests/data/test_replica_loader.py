"""Tests for Replica SLAM data loader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anima_slam_coko.data.replica_loader import ReplicaSLAMDataset, load_scene


@pytest.fixture
def prepared_scene(tmp_path: Path) -> Path:
    """Create a minimal prepared scene for testing."""
    agent_dir = tmp_path / "test_scene" / "agent_0"
    results_dir = agent_dir / "results"
    results_dir.mkdir(parents=True)

    # Create 5 frames
    for i in range(5):
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(results_dir / f"frame{i:06d}.jpg"), rgb)

        depth = np.ones((480, 640), dtype=np.uint16) * 3000
        cv2.imwrite(str(results_dir / f"depth{i:06d}.png"), depth)

    # Create trajectory (5 identity poses as 4x4 matrices)
    lines = []
    for i in range(5):
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = float(i) * 0.1
        lines.append(" ".join(f"{v:.8f}" for v in c2w.flatten()))
    (agent_dir / "traj.txt").write_text("\n".join(lines) + "\n")

    return tmp_path / "test_scene"


def test_loader_loads_frames(prepared_scene: Path) -> None:
    ds = ReplicaSLAMDataset(prepared_scene / "agent_0")
    assert len(ds) == 5


def test_loader_returns_correct_types(prepared_scene: Path) -> None:
    ds = ReplicaSLAMDataset(prepared_scene / "agent_0")
    fid, rgb, depth, c2w = ds[0]
    assert isinstance(fid, int)
    assert rgb.shape == (480, 640, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (480, 640)
    assert depth.dtype == np.float32
    assert c2w.shape == (4, 4)


def test_loader_depth_scale(prepared_scene: Path) -> None:
    ds = ReplicaSLAMDataset(prepared_scene / "agent_0", depth_scale=6553.5)
    _, _, depth, _ = ds[0]
    # We wrote 3000 uint16, so depth_m = 3000 / 6553.5 ≈ 0.458
    assert 0.4 < depth.mean() < 0.5


def test_loader_pose_translation(prepared_scene: Path) -> None:
    ds = ReplicaSLAMDataset(prepared_scene / "agent_0")
    _, _, _, c2w0 = ds[0]
    _, _, _, c2w1 = ds[1]
    assert abs(c2w0[0, 3] - 0.0) < 1e-6
    assert abs(c2w1[0, 3] - 0.1) < 1e-6


def test_loader_frame_limit(prepared_scene: Path) -> None:
    ds = ReplicaSLAMDataset(prepared_scene / "agent_0", frame_limit=3)
    assert len(ds) == 3


def test_loader_get_frame_dict(prepared_scene: Path) -> None:
    ds = ReplicaSLAMDataset(prepared_scene / "agent_0")
    fd = ds.get_frame_dict(0)
    assert "rgb" in fd
    assert "depth" in fd
    assert "c2w" in fd
    assert "intrinsics" in fd
    assert fd["intrinsics"].shape == (3, 3)


def test_load_scene(prepared_scene: Path) -> None:
    agents = load_scene("test_scene", slam_data_root=prepared_scene.parent)
    assert 0 in agents
    assert len(agents[0]) == 5


def test_load_scene_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_scene("nonexistent", slam_data_root=tmp_path)
