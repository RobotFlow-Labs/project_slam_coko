"""Request / response schemas for the SLAM-COKO API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RuntimeMode(str, Enum):
    rendered_depth = "rendered_depth"
    camera_depth = "camera_depth"


class RunRequest(BaseModel):
    scene: str = Field(description="Scene name, e.g. 'room0'")
    runtime_mode: RuntimeMode = RuntimeMode.rendered_depth
    num_agents: int = Field(default=2, ge=1, le=8)
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class RunResponse(BaseModel):
    job_id: str
    status: JobStatus
    scene: str
    message: str = ""


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    scene: str
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    module: str = "slam-coko"
    version: str = "0.1.0"
    gpu_available: bool = False
    datasets_mounted: bool = False
    weights_available: bool = False
