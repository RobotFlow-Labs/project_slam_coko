"""FastAPI application for the SLAM-COKO pipeline."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from anima_slam_coko.api.jobs import JobManager
from anima_slam_coko.api.models import (
    HealthResponse,
    JobInfo,
    RunRequest,
    RunResponse,
    JobStatus,
)

app = FastAPI(
    title="SLAM-COKO",
    description="Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM",
    version="0.1.0",
)

_jobs = JobManager()

SLAM_DATA = Path("/mnt/forge-data/datasets/replica_slam")
DINO_WEIGHTS = Path("/mnt/forge-data/models/facebook--dinov2-small")


@app.get("/health", response_model=HealthResponse)
def health():
    import torch

    return HealthResponse(
        status="ok",
        gpu_available=torch.cuda.is_available(),
        datasets_mounted=SLAM_DATA.exists(),
        weights_available=DINO_WEIGHTS.exists(),
    )


@app.get("/ready")
def ready():
    if not SLAM_DATA.exists():
        raise HTTPException(503, "Datasets not mounted")
    if not DINO_WEIGHTS.exists():
        raise HTTPException(503, "DINOv2 weights not found")
    return {"ready": True, "module": "slam-coko"}


@app.get("/info")
def info():
    scenes = sorted(d.name for d in SLAM_DATA.iterdir() if d.is_dir()) if SLAM_DATA.exists() else []
    return {
        "module": "slam-coko",
        "version": "0.1.0",
        "paper": "arXiv:2604.00804",
        "available_scenes": scenes,
    }


@app.post("/runs", response_model=RunResponse)
def create_run(req: RunRequest):
    scene_dir = SLAM_DATA / req.scene
    if not scene_dir.exists():
        raise HTTPException(404, f"Scene '{req.scene}' not found at {scene_dir}")
    job_id = _jobs.enqueue(req)
    return RunResponse(
        job_id=job_id,
        status=JobStatus.queued,
        scene=req.scene,
        message=f"SLAM run queued for scene '{req.scene}'",
    )


@app.get("/runs/{job_id}", response_model=JobInfo)
def get_run(job_id: str):
    info = _jobs.get(job_id)
    if info is None:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return info


@app.get("/runs", response_model=list[JobInfo])
def list_runs():
    return _jobs.list_jobs()
