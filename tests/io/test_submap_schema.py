from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from anima_slam_coko.schemas.submap import SubmapRecord


def _make_record(runtime_mode: str) -> SubmapRecord:
    base_payload = {
        "agent_id": 0,
        "submap_id": 1,
        "runtime_mode": runtime_mode,
        "keyframe_ids": np.asarray([3, 5], dtype=np.int64),
        "submap_c2ws": np.asarray([np.eye(4), np.eye(4)], dtype=np.float32),
        "gaussian_xyz": np.asarray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        "gaussian_opacity": np.asarray([[0.5], [0.7]], dtype=np.float32),
        "gaussian_scale": np.asarray([[1.0, 1.0, 1.0], [0.9, 0.8, 0.7]], dtype=np.float32),
        "gaussian_rotation": np.asarray([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.1, 0.1]], dtype=np.float32),
        "gaussian_features": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "descriptor_vector": np.asarray([0.2, 0.4, 0.6], dtype=np.float32),
    }
    if runtime_mode == "rendered_depth":
        base_payload["rendered_depth"] = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    else:
        base_payload["camera_depth"] = np.asarray([[4.0, 3.0], [2.0, 1.0]], dtype=np.float32)
    return SubmapRecord(**base_payload)


def test_rendered_depth_submap_roundtrip() -> None:
    record = _make_record("rendered_depth")

    payload = record.to_serializable()
    restored = SubmapRecord.from_serializable(payload)

    assert restored.runtime_mode == "rendered_depth"
    assert restored.rendered_depth is not None
    assert restored.camera_depth is None
    assert restored.gaussian_count == 2
    assert np.array_equal(restored.keyframe_ids, record.keyframe_ids)
    assert np.allclose(restored.gaussian_xyz, record.gaussian_xyz)


def test_camera_depth_submap_roundtrip() -> None:
    record = _make_record("camera_depth")

    payload = record.to_serializable()
    restored = SubmapRecord.from_serializable(payload)

    assert restored.runtime_mode == "camera_depth"
    assert restored.camera_depth is not None
    assert restored.rendered_depth is None
    assert np.allclose(restored.camera_depth, record.camera_depth)


def test_malformed_payloads_are_rejected() -> None:
    payload = _make_record("rendered_depth").to_serializable()
    payload["gaussian_xyz"]["data"] = [[0.0, 0.1], [0.2, 0.3]]

    with pytest.raises(ValidationError):
        SubmapRecord.from_serializable(payload)

    camera_payload = _make_record("camera_depth").to_serializable()
    camera_payload.pop("camera_depth")

    with pytest.raises(ValidationError):
        SubmapRecord.from_serializable(camera_payload)
