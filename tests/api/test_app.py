"""Tests for SLAM-COKO FastAPI application."""

from __future__ import annotations

from anima_slam_coko.api.models import (
    HealthResponse,
    JobStatus,
    RunRequest,
    RunResponse,
    RuntimeMode,
)


def test_run_request_defaults() -> None:
    req = RunRequest(scene="room0")
    assert req.scene == "room0"
    assert req.runtime_mode == RuntimeMode.rendered_depth
    assert req.num_agents == 2
    assert req.config_overrides == {}


def test_run_response_creation() -> None:
    resp = RunResponse(job_id="abc123", status=JobStatus.queued, scene="room0")
    assert resp.job_id == "abc123"
    assert resp.status == JobStatus.queued


def test_health_response() -> None:
    h = HealthResponse(status="ok", gpu_available=True, datasets_mounted=True, weights_available=True)
    assert h.module == "slam-coko"
    assert h.version == "0.1.0"


def test_run_request_validation() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RunRequest(scene="room0", num_agents=0)

    with pytest.raises(ValidationError):
        RunRequest(scene="room0", num_agents=100)
