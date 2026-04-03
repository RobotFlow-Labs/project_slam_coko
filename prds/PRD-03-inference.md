# PRD-03: Inference, Loop Closure & Fusion

> Module: SLAM-COKO | Priority: P0  
> Depends on: PRD-01, PRD-02  
> Status: ✅ Complete

## Objective

Run the full multi-agent paper pipeline end-to-end: detect loops from keyframe features, register overlapping submaps without known initial relative poses, optimize poses with GTSAM, and merge/refine the global Gaussian map.

## Context (from paper)

The core paper novelty is the ability to merge submaps from different agents without initial relative poses, using only pure 3D Gaussian submaps in rendered-depth mode or lightweight depth payloads in camera-depth mode.

**Paper reference:** §3.3, Eq. (3), Fig. 1, Fig. 2

## Acceptance Criteria

- [x] Loop detector finds inter- and intra-agent candidates from submap feature vectors
- [x] Rendered-depth mode converts first-keyframe renders into point clouds
- [x] Camera-depth mode registers stored depth maps and supports co-visible Gaussian removal
- [x] Registration uses FPFH + RANSAC before ICP refinement
- [x] GTSAM PGO solves odometry + loop edges and returns optimized submap poses
- [x] Global map merge and refinement produce a merged Gaussian map
- [x] Test: `uv run pytest tests/loop_closure/test_detector.py -v` passes
- [x] Test: `uv run pytest tests/registration/test_pipeline.py -v` passes
- [x] Test: `uv run pytest tests/pgo/test_gtsam_solver.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_coko/loop_closure/detector.py` | loop candidate search from feature vectors | §3.3 | ~180 |
| `src/anima_slam_coko/registration/rendered_depth.py` | render depth -> point cloud path | §3.3 | ~140 |
| `src/anima_slam_coko/registration/coarse.py` | FPFH + RANSAC coarse alignment | §3.3 | ~140 |
| `src/anima_slam_coko/registration/fine.py` | ICP refinement | §3.3 | ~120 |
| `src/anima_slam_coko/pgo/gtsam_solver.py` | pose graph optimization | Eq. (3) | ~220 |
| `src/anima_slam_coko/fusion/merge.py` | submap concatenation and pruning | §3.3 | ~180 |
| `src/anima_slam_coko/fusion/refine.py` | post-merge refinement | §3.3 | ~160 |
| `src/anima_slam_coko/cli/run_slam.py` | top-level CLI | Fig. 2 | ~100 |
| `tests/loop_closure/test_detector.py` | loop tests | — | ~90 |
| `tests/registration/test_pipeline.py` | registration tests | — | ~120 |
| `tests/pgo/test_gtsam_solver.py` | solver tests | — | ~100 |

## Architecture Detail (from paper)

### Inputs
- `submap_features`: `Float32[M,384]`
- `submap_start_depth`: `Float32[H,W]` or rendered depth
- `submap_c2w`: `Float32[4,4]`

### Outputs
- `loop_edges`: list of rigid transforms and fitness statistics
- `optimized_submap_poses`: `Float32[S,4,4]`
- `merged_map`: Gaussian state

### Algorithm
```python
class LoopClosurePipeline:
    def run(self, submaps: list[SubmapRecord]) -> FusionResult:
        loops = self.detector.detect(submaps)
        registrations = self.registration.register_all(loops, submaps)
        optimized = self.gtsam_solver.optimize(submaps, registrations)
        merged = self.merger.merge(submaps, optimized)
        return self.refiner.refine(merged)
```

## Dependencies

```toml
gtsam = ">=4.2"
open3d = ">=0.18"
numpy = ">=1.26"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| Replica submap pair fixtures | small | `tests/fixtures/replica_submaps/` | generated from sample run |
| Aria submap pair fixtures | small | `tests/fixtures/aria_submaps/` | generated from sample run |

## Test Plan

```bash
uv run pytest tests/loop_closure/test_detector.py -v
uv run pytest tests/registration/test_pipeline.py -v
uv run pytest tests/pgo/test_gtsam_solver.py -v
```

## References

- Paper: §3.3, Eq. (3)
- Reference impl: `repositories/coko-slam/src/entities/gtsam_pose_graph.py`
- Feeds into: PRD-04, PRD-05, PRD-06
