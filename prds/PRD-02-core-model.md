# PRD-02: Core Local Agent Model

> Module: SLAM-COKO | Priority: P0  
> Depends on: PRD-01  
> Status: ✅ Complete

## Objective

Implement the paper-faithful single-agent runtime: DINOv2 feature extraction, feature-based keyframing/submapping, RGB-D odometry tracking, Gaussian mapping, and GaussianSPA-style compaction.

## Context (from paper)

The paper’s local agent produces compact Gaussian submaps and keyframe feature summaries. Keyframe selection uses feature-space distance, local mapping keeps 10 keyframes per submap, and compaction removes redundant Gaussians during optimization.

**Paper reference:** §3.1, §3.2, Fig. 2, Algorithm 1

## Acceptance Criteria

- [x] DINOv2-Small embeddings are computed as `Float32[1,384]`
- [x] Keyframe selection uses minimum feature distance threshold `alpha`
- [x] Submapping rotates after 10 keyframes and feature divergence threshold
- [x] Mapping runs 1000 iterations for new submaps with compaction window `[700, 950]`
- [x] Rendered-depth and camera-depth checkpoint writers both work
- [x] Test: `uv run pytest tests/keyframing/test_policy.py -v` passes
- [x] Test: `uv run pytest tests/mapping/test_compaction.py -v` passes
- [x] Test: `uv run pytest tests/tracking/test_odometry.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_coko/features/dino.py` | DINOv2 feature wrapper | §3.1 | ~120 |
| `src/anima_slam_coko/keyframing/policy.py` | Keyframe and submap decision logic | §3.1 | ~140 |
| `src/anima_slam_coko/tracking/visual_odometer.py` | RGB-D odometry | §2.1 recap | ~140 |
| `src/anima_slam_coko/tracking/tracker.py` | Pose refinement against Gaussian model | §2.1 recap | ~220 |
| `src/anima_slam_coko/mapping/gaussian_state.py` | Gaussian parameter container | §2.1 | ~220 |
| `src/anima_slam_coko/mapping/mapper.py` | Seeding, growth, optimization | §2.1, §3.2 | ~260 |
| `src/anima_slam_coko/mapping/compaction.py` | Optimization-sparsification loop | §3.2 | ~180 |
| `src/anima_slam_coko/runtime/agent_runtime.py` | Agent orchestration and submap saves | Fig. 2 | ~240 |
| `tests/keyframing/test_policy.py` | Keyframing tests | — | ~90 |
| `tests/mapping/test_compaction.py` | Compaction tests | — | ~120 |
| `tests/tracking/test_odometry.py` | Tracking tests | — | ~90 |

## Architecture Detail (from paper)

### Inputs
- `rgb`: `UInt8[H,W,3]`
- `depth`: `Float32[H,W]`
- `feature`: `Float32[1,384]`
- `gaussian_xyz`: `Float32[N,3]`
- `gaussian_opacity`: `Float32[N,1]`

### Outputs
- `estimated_c2w`: `Float32[4,4]`
- `submap_record`: schema from PRD-01

### Algorithm
```python
class KeyframePolicy:
    def should_promote(self, feature, submap_index) -> bool:
        min_dist = self.min_feature_distance(feature, submap_index)
        return min_dist >= self.alpha


class CompactionScheduler:
    def step(self, iteration: int, opacity: Tensor) -> Tensor:
        if 700 <= iteration <= 950:
            return self.apply_optimizing_spa(opacity)
        return opacity
```

## Dependencies

```toml
torch = ">=2.5"
torchvision = ">=0.20"
open3d = ">=0.18"
faiss-cpu = ">=1.8"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| DINOv2-Small weights | ~100MB | `./dinov2-small` or cache dir | upstream model download |
| Replica sample scene | medium | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ReplicaMultiagent/office_0` | HF clone |

## Test Plan

```bash
uv run pytest tests/keyframing/test_policy.py -v
uv run pytest tests/mapping/test_compaction.py -v
uv run pytest tests/tracking/test_odometry.py -v
```

## References

- Paper: §3.1, §3.2
- Reference impl: `repositories/coko-slam/src/entities/agent.py`
- Reference impl: `repositories/coko-slam/src/entities/mapper.py`
- Feeds into: PRD-03, PRD-04
