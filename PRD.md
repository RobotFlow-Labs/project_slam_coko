# SLAM-COKO: Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM — Implementation PRD
## ANIMA Wave-7 Module

**Status:** In progress  
**Version:** 0.2  
**Date:** 2026-04-03  
**Paper:** Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM  
**Paper Link:** https://arxiv.org/abs/2604.00804  
**Repo:** https://github.com/lemonci/coko-slam  
**Functional Name:** SLAM-COKO  
**Stack:** ATLAS

## Build Plan — Executable PRDs

> Total PRDs: 7 | Tasks: 21 | Status: 12/21 complete

| # | PRD | Title | Priority | Tasks | Status |
|---|---|---|---|---|---|
| 1 | [PRD-01](prds/PRD-01-foundation.md) | Foundation & Config | P0 | 3 | ✅ |
| 2 | [PRD-02](prds/PRD-02-core-model.md) | Core Local Agent Model | P0 | 3 | ✅ |
| 3 | [PRD-03](prds/PRD-03-inference.md) | Inference, Loop Closure & Fusion | P0 | 3 | ✅ |
| 4 | [PRD-04](prds/PRD-04-evaluation.md) | Evaluation & Paper Reproduction | P1 | 3 | ✅ |
| 5 | [PRD-05](prds/PRD-05-api-docker.md) | API & Docker | P1 | 3 | ⬜ |
| 6 | [PRD-06](prds/PRD-06-ros2.md) | ROS2 Integration | P1 | 3 | ⬜ |
| 7 | [PRD-07](prds/PRD-07-production.md) | Production Hardening | P2 | 3 | ⬜ |

## 1. Executive Summary

SLAM-COKO reproduces the paper’s centralized multi-agent RGB-D Gaussian Splatting SLAM pipeline with the same core design: DINOv2-based keyframing, per-agent compacted Gaussian submaps, server-side loop detection, FPFH + RANSAC coarse registration, ICP refinement, and GTSAM pose graph optimization. The implementation target is paper-faithful CUDA execution first, followed by ANIMA service and ROS2 integration.

## 2. Paper Verification Status

- [x] Title verified against live arXiv entry
- [x] Correct arXiv ID verified: `2604.00804`
- [x] Reference repo confirmed and inspected
- [x] Paper read for architecture, datasets, metrics, and algorithm details
- [x] Local project metadata normalized away from placeholder `AMATERASU`
- [ ] Datasets mounted on shared volume and checked
- [ ] End-to-end paper-style reproduction run completed
- [ ] Metrics matched within tolerance
- **Verdict:** READY TO BUILD

## 3. What We Take From The Paper

- Multi-agent RGB-D Gaussian Splatting SLAM with centralized server fusion.
- DINOv2-Small feature embeddings for both keyframe selection and loop detection.
- Keyframe rule based on minimum feature-space distance threshold `alpha`.
- Fixed 10-keyframe target per submap.
- GaussianSPA-style optimization-sparsification to prune redundant Gaussians during local mapping.
- Two loop-closure operating modes:
  - rendered-depth mode using only Gaussian submaps
  - camera-depth mode carrying lightweight depth images for stronger registration
- FPFH + RANSAC coarse registration, followed by ICP fine alignment.
- GTSAM-based pose graph optimization with loop and odometry edges.
- Replica and Aria evaluation protocol, including rendering and communication metrics.

## 4. What We Skip

- Re-implementing unsupported baselines inside this repo.
- MLX-first or CPU-first re-architecture in the first pass.
- Decentralized SLAM research extensions beyond the centralized paper design.
- Non-RGB-D sensor fusion in the reproduction phase.

## 5. What We Adapt

- We split the monolithic reference repo into ANIMA-style subsystems under `src/anima_slam_coko/`.
- We formalize configs with TOML and Pydantic settings instead of YAML-only runtime state.
- We wrap the pipeline in API and ROS2 entrypoints required by ANIMA.
- We preserve the paper’s two registration modes while making them explicit runtime options.
- We add regression tests and artifact reporting around the paper tables.

## 6. Architecture

### Inputs
- `rgb`: `UInt8[H,W,3]`
- `depth`: `Float32[H,W]`
- `intrinsics`: `Float32[3,3]`
- `agent_id`: `int`

### Outputs
- `submap.ckpt`: compacted Gaussian state + keyframe features + keyframe poses
- `loop_edges.json`: validated inter/intra-agent loop registrations
- `optimized_submap_poses.ckpt`: `Float32[S,4,4]`
- `merged_map.ply`: fused global Gaussian map
- `metrics.json`: rendering, keyframe count, and transmitted-data statistics

### Runtime Graph
- Agent side: `feature extractor -> keyframing -> odometry -> mapper -> compaction -> submap store`
- Server side: `loop detection -> registration -> GTSAM PGO -> map merge -> refinement -> evaluation`

## 7. Implementation Phases

### Phase 1 — Foundation + Schema
- Normalize project metadata to SLAM-COKO.
- Create canonical package and config layout.
- Define dataset, submap, and runtime schemas.

### Phase 2 — Paper-Faithful Local Agent
- Implement feature-based keyframing and submapping.
- Implement tracker, mapper, and compaction.
- Serialize rendered-depth and camera-depth submaps.

### Phase 3 — Server Fusion + Reproduction
- Implement loop detection, registration, and GTSAM PGO.
- Merge and refine submaps.
- Reproduce Table 1 / Table 2 / Table 4 style outputs.

### Phase 4 — ANIMA Interfaces
- Add FastAPI, Docker, and ROS2 integration.
- Add experiment artifact packaging and validation gates.

## 8. Datasets

| Dataset | Split / Scenes | URL | Phase Needed |
|---|---|---|---|
| ReplicaMultiagent | `office_0`, `apart_0`, `apart_1`, `apart_2` | https://huggingface.co/datasets/voviktyl/ReplicaMultiagent | Phase 2 |
| AriaDigitalTwin | raw source sequences | https://www.projectaria.com/datasets/adt/ | Phase 2 |
| AriaMultiagent | `room0`, `room1` after preprocessing | generated by repo script | Phase 3 |

## 9. Dependencies on Other Wave Projects

| Needs output from | What it provides |
|---|---|
| None | This module is paper-self-contained |

## 10. Success Criteria

- Replica training-view PSNR matches paper within roughly 1 dB on all four scenes in camera-depth mode.
- Aria novel-view PSNR is within roughly 1.5 dB of paper values on both rooms.
- Communication reduction remains at or above 85% relative to the large-map baselines cited by the paper.
- Unknown initial relative poses between agents no longer prevent map merging.
- Both rendered-depth and camera-depth registration paths run end-to-end.

## 11. Risk Assessment

- The current local PDF is incorrect, so future work must keep `2604.00804` as the canonical paper ID.
- The reference implementation is CUDA- and Open3D-heavy; portability is limited.
- Rendered-depth registration degrades at lower resolutions, especially on Aria.
- GTSAM packaging and multi-GPU orchestration may create environment fragility.
- Current project scaffold uses placeholder package names and must be normalized before implementation.

## 12. Shenzhen / Demo Readiness

- Minimum demo target: two-agent Replica reproduction with successful unknown-pose map merge.
- Preferred demo target: real-room replay using Aria-style or ZED 2i RGB-D streams through the ROS2 bridge.
- Demo artifact set: merged map visualization, keyframe/submap telemetry, paper-metric comparison, transmitted-data chart.
