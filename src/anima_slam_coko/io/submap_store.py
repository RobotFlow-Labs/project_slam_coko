"""Checkpoint readers and writers for submap payloads."""

from __future__ import annotations

import json
from pathlib import Path

from anima_slam_coko.schemas.submap import SubmapRecord


def _write_submap(record: SubmapRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_serializable(), indent=2))


def write_rendered_depth_submap(record: SubmapRecord, path: Path) -> None:
    if record.runtime_mode != "rendered_depth":
        raise ValueError("record.runtime_mode must be 'rendered_depth'")
    _write_submap(record, path)


def write_camera_depth_submap(record: SubmapRecord, path: Path) -> None:
    if record.runtime_mode != "camera_depth":
        raise ValueError("record.runtime_mode must be 'camera_depth'")
    _write_submap(record, path)


def read_submap(path: Path) -> SubmapRecord:
    payload = json.loads(path.read_text())
    return SubmapRecord.from_serializable(payload)
