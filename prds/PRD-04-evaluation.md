# PRD-04: Evaluation & Paper Reproduction

> Module: SLAM-COKO | Priority: P1  
> Depends on: PRD-01, PRD-02, PRD-03  
> Status: ⬜ Not started

## Objective

Reproduce the paper’s rendering and communication tables on Replica and Aria, and emit machine-readable ANIMA reports that highlight any metric gaps.

## Context (from paper)

The paper evaluates training-view and novel-view synthesis, keyframe counts, and total transmitted data. Reproduction work must preserve both fidelity and communication efficiency.

**Paper reference:** §4, Tables 1-5

## Acceptance Criteria

- [ ] Replica training-view PSNR / SSIM / LPIPS / depth-L1 are computed scene-by-scene
- [ ] Aria novel-view evaluation is computed room-by-room
- [ ] Keyframe counts and total transmitted data are reported per agent
- [ ] A gap report compares local metrics against the paper tables
- [ ] Test: `uv run pytest tests/eval/test_replica_metrics.py -v` passes
- [ ] Test: `uv run pytest tests/eval/test_bandwidth.py -v` passes
- [ ] Benchmark: `artifacts/reports/paper_gap_report.md` generated

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_coko/eval/metrics.py` | PSNR/SSIM/LPIPS/depth-L1 helpers | §4 | ~140 |
| `src/anima_slam_coko/eval/replica.py` | Replica evaluation runner | Table 1 | ~180 |
| `src/anima_slam_coko/eval/aria.py` | Aria evaluation runner | Table 2 | ~180 |
| `src/anima_slam_coko/eval/bandwidth.py` | keyframe and transmitted-data audits | Tables 3-5 | ~160 |
| `scripts/reproduce_replica.py` | reproduction script | §4 | ~120 |
| `scripts/reproduce_aria.py` | reproduction script | §4 | ~120 |
| `tests/eval/test_replica_metrics.py` | metrics tests | — | ~100 |
| `tests/eval/test_bandwidth.py` | bandwidth tests | — | ~100 |

## Architecture Detail (from paper)

### Inputs
- `merged_map`
- `dataset`
- `keyframe_ids`: `Int64[K]`
- `optimized_kf_c2ws`: `Float32[K,4,4]`

### Outputs
- `metrics.json`
- `paper_gap_report.md`

### Algorithm
```python
def compare_to_paper(scene_metrics: dict, paper_targets: dict) -> dict:
    return {
        metric: scene_metrics[metric] - paper_targets[metric]
        for metric in paper_targets
    }
```

## Dependencies

```toml
torchmetrics = ">=1.4"
pytorch-msssim = ">=1.0"
matplotlib = ">=3.9"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| Replica runs | large | `runs/replica/` | generated |
| Aria runs | large | `runs/aria/` | generated |

## Test Plan

```bash
uv run pytest tests/eval/test_replica_metrics.py -v
uv run pytest tests/eval/test_bandwidth.py -v
```

## References

- Paper: §4, Tables 1-5
- Reference impl: `repositories/coko-slam/src/utils/mapping_eval.py`
- Feeds into: PRD-05, PRD-07
