# PRD-05: API & Docker

> Module: SLAM-COKO | Priority: P1  
> Depends on: PRD-01, PRD-02, PRD-03, PRD-04  
> Status: ✅ Complete

## Objective

Expose the paper-faithful pipeline as a reproducible service with FastAPI and CUDA Docker packaging, without changing core algorithm behavior.

## Context (from paper)

The paper implementation is script-driven. ANIMA needs a stable service boundary for launching runs, exporting artifacts, and integrating downstream modules.

**Paper reference:** operationalization of Fig. 2

## Acceptance Criteria

- [ ] FastAPI exposes run, status, artifact, and evaluation endpoints
- [ ] Docker image includes CUDA, Open3D, FAISS, and GTSAM dependencies
- [ ] Rendered-depth and camera-depth modes are selectable via API request
- [ ] Health checks validate dataset mounts and weight presence
- [ ] Test: `uv run pytest tests/api/test_app.py -v` passes
- [ ] Test: `docker compose config` succeeds

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_coko/api/app.py` | FastAPI app | Fig. 2 | ~180 |
| `src/anima_slam_coko/api/models.py` | request / response schemas | — | ~100 |
| `src/anima_slam_coko/api/jobs.py` | background job orchestration | — | ~160 |
| `Dockerfile` | CUDA runtime image | repo README | ~80 |
| `docker-compose.yml` | local service stack | — | ~60 |
| `.env.example` | runtime config sample | — | ~40 |
| `tests/api/test_app.py` | API tests | — | ~120 |

## Architecture Detail (from paper)

### Inputs
- `RunRequest.dataset`
- `RunRequest.scene`
- `RunRequest.runtime_mode`

### Outputs
- run ID, artifact locations, metrics summary

### Algorithm
```python
@app.post("/runs")
def create_run(req: RunRequest) -> RunResponse:
    job_id = jobs.enqueue(req)
    return RunResponse(job_id=job_id, status="queued")
```

## Dependencies

```toml
fastapi = ">=0.115"
uvicorn = ">=0.30"
python-multipart = ">=0.0.9"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| Datasets volume mount | large | `/Volumes/AIFlowDev/RobotFlowLabs/datasets` | local mount |
| Repo cache / weights | medium | `/workspace/cache` | local or container cache |

## Test Plan

```bash
uv run pytest tests/api/test_app.py -v
docker compose config
```

## References

- Reference impl: `repositories/coko-slam/README.md`
- Depends on: PRD-03, PRD-04
- Feeds into: PRD-06, PRD-07
