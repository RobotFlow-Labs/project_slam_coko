# PRD-01: Foundation & Config

> Module: SLAM-COKO | Priority: P0  
> Depends on: None  
> Status: ✅ Complete

## Objective

Normalize the scaffold into a buildable SLAM-COKO package with canonical configs, schemas, and dataset manifests so the paper pipeline can be implemented without ambiguity.

## Context (from paper)

The paper is a centralized multi-agent RGB-D Gaussian Splatting SLAM system with explicit per-agent runtime, submap artifacts, and evaluation datasets. A clean implementation needs stable package boundaries before algorithm work starts.

**Paper reference:** §1, §4, Fig. 2

## Acceptance Criteria

- [x] Canonical package path is `src/anima_slam_coko/`
- [x] Project metadata uses paper ID `2604.00804`
- [x] Replica and Aria dataset manifests load from shared-volume paths
- [x] Submap checkpoint schemas support rendered-depth and camera-depth modes
- [x] Test: `uv run pytest tests/config/test_settings.py -v` passes
- [x] Test: `uv run pytest tests/io/test_submap_schema.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_coko/config.py` | Pydantic settings and config loading | §1, §4 | ~180 |
| `src/anima_slam_coko/schemas/submap.py` | Typed submap/checkpoint contracts | Fig. 2, §3.3 | ~180 |
| `src/anima_slam_coko/data/catalog.py` | Dataset manifest and shared-volume resolution | §4 | ~140 |
| `configs/coko/base.toml` | Canonical default config | repo configs | ~120 |
| `configs/coko/replica_office0.toml` | Reproduction preset | Table 1 | ~80 |
| `configs/coko/aria_room0.toml` | Reproduction preset | Table 2 | ~80 |
| `tests/config/test_settings.py` | Config validation tests | — | ~80 |
| `tests/io/test_submap_schema.py` | Serialization tests | — | ~120 |

## Architecture Detail (from paper)

### Inputs
- `project_config`: `dict[str, Any]`
- `dataset_manifest`: `dict[str, str]`

### Outputs
- `settings`: `BaseSettings`
- `submap_record`: schema with Gaussian tensors, feature vectors, poses, and optional depth payload

### Algorithm
```python
class SlamCokoSettings(BaseSettings):
    paper_arxiv: str = "2604.00804"
    runtime_mode: Literal["rendered_depth", "camera_depth"]
    dataset_root: Path


class SubmapRecord(BaseModel):
    agent_id: int
    submap_id: int
    keyframe_ids: NDArray[np.int64]
    submap_c2ws: NDArray[np.float32]
    gaussian_xyz: NDArray[np.float32]
    gaussian_opacity: NDArray[np.float32]
```

## Dependencies

```toml
pydantic = ">=2.7"
numpy = ">=1.26"
tomli = ">=2.0"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| ReplicaMultiagent | large | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ReplicaMultiagent` | git-lfs clone |
| AriaMultiagent | medium | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/AriaMultiagent` | preprocess from ADT |

## Test Plan

```bash
uv run pytest tests/config/test_settings.py -v
uv run pytest tests/io/test_submap_schema.py -v
```

## References

- Paper: §1, §4
- Reference impl: `repositories/coko-slam/run_slam.py`
- Feeds into: PRD-02, PRD-03
