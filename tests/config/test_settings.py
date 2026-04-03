from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from anima_slam_coko.config import load_settings
from anima_slam_coko.data.catalog import resolve_scene


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "coko" / "base.toml"
REPLICA_PRESET = REPO_ROOT / "configs" / "coko" / "replica_office0.toml"
ARIA_PRESET = REPO_ROOT / "configs" / "coko" / "aria_room0.toml"


def test_replica_settings_load() -> None:
    settings = load_settings(BASE_CONFIG, REPLICA_PRESET)

    assert settings.project.codename == "SLAM-COKO"
    assert settings.project.paper_arxiv == "2604.00804"
    assert settings.runtime.mode == "rendered_depth"
    assert settings.data.dataset == "replica"
    assert settings.data.scene == "office_0"
    assert settings.data.agent_ids == (0, 1)
    assert settings.loop_detection.embed_size == 384
    assert settings.submapping.keyframe_num == 10


def test_aria_settings_load() -> None:
    settings = load_settings(BASE_CONFIG, ARIA_PRESET)

    assert settings.data.dataset == "aria"
    assert settings.data.scene == "room0"
    assert settings.data.agent_ids == (0, 1, 2)
    assert settings.camera.height == 512
    assert settings.tracking.iterations == 200


def test_runtime_mode_validation() -> None:
    with pytest.raises(ValidationError):
        load_settings(overrides={"runtime": {"mode": "invalid"}})


def test_scene_resolution_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "ReplicaMultiagent" / "office_0").mkdir(parents=True)
    (tmp_path / "AriaMultiagent" / "room0").mkdir(parents=True)

    replica_scene = resolve_scene("replica", "office_0", tmp_path)
    aria_scene = resolve_scene("aria", "room0", tmp_path)

    assert replica_scene == tmp_path / "ReplicaMultiagent" / "office_0"
    assert aria_scene == tmp_path / "AriaMultiagent" / "room0"
