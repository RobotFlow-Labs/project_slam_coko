# SLAM-COKO — Execution Ledger

Resume rule: Read this file COMPLETELY before writing any code.
This project covers exactly ONE paper: Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM.

## 1. Working Rules
- Work only inside `project_slam_coko/`
- This wave has 17 parallel projects, 17 papers, 17 agents
- Prefix every commit with `[SLAM-COKO]`
- Stage only `project_slam_coko/` files
- VERIFY THE PAPER BEFORE BUILDING ANYTHING

## 2. The Paper
- **Title**: Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM
- **ArXiv**: 2604.00804
- **Link**: https://arxiv.org/abs/2604.00804
- **Repo**: https://github.com/lemonci/coko-slam
- **Compute**: GPU-NEED
- **Verification status**: ArXiv ID ✅ | Repo ✅ | Paper read ✅
- **Important note**: `papers/2503.15868_CokO-SLAM.pdf` is a mismatched local PDF and must not be used as source of truth.

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: PRD-03 COMPLETE, ready for PRD-04
- **MVP Readiness**: 45%
- **Accomplished**:
  1. Paper-grounded PRD suite and task breakdown created
  2. Canonical `anima_slam_coko` package scaffolded
  3. Python 3.11 + UV project metadata normalized
  4. Dataset catalog, base presets, and submap schemas implemented
  5. Linux CUDA bootstrap script added for post-prebuild deployment
  6. DINO wrapper and feature-distance keyframing policy implemented
  7. RGB-D odometry and tracker fallback path implemented
  8. Gaussian state container, compaction scheduler, and submap checkpoint writers implemented
  9. Loop closure detector with FAISS-style numpy search implemented
  10. Rendered-depth and camera-depth point cloud extraction implemented
  11. FPFH + RANSAC coarse registration implemented
  12. ICP fine registration implemented
  13. GTSAM PGO solver with naive fallback implemented
  14. Submap fusion merge + duplicate pruning implemented
  15. Post-merge refinement with compaction implemented
  16. CLI orchestrator chaining full pipeline implemented
  17. Data catalog module created (was missing, broke config imports)
  18. Code review: 6 bugs found and fixed
  19. All 32 tests passing (including open3d integration test)
  20. All training deps installed (torch 2.5.1+cu121, open3d, scipy, transformers)
- **TODO**:
  1. Implement PRD-04 evaluation & paper reproduction
  2. Implement PRD-05 API & Docker serving
  3. Implement PRD-06 ROS2 integration
  4. Implement PRD-07 production hardening
  5. Confirm dataset availability on shared volume (Replica at /mnt/forge-data/datasets/replica/)
  6. Pull DINOv2 weights for first runtime smoke test
- **Blockers**: None

## 4. Datasets
### Required for this paper
| Dataset | Size | URL | Format | Phase Needed |
|---------|------|-----|--------|-------------|
| ReplicaMultiagent | Large | Internal shared volume / reference repo prep | RGB-D multi-agent scenes | Reproduction |
| AriaMultiagent | Medium | Internal shared volume / ADT preprocessing | RGB-D multi-agent scenes | Reproduction |

### Available on server
- Replica: `/mnt/forge-data/datasets/replica/`
- TUM RGB-D: `/mnt/forge-data/datasets/tum/`
- DINOv2-Small: `/mnt/forge-data/models/facebook--dinov2-small`

### Training paths
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Checkpoints: `/mnt/artifacts-datai/checkpoints/project_slam_coko/`
- Logs: `/mnt/artifacts-datai/logs/project_slam_coko/`

## 5. Hardware
- ZED 2i stereo camera: Available
- Unitree L2 3D LiDAR: Available
- xArm 6 cobot: Pending purchase
- Mac Studio M-series: MLX dev
- 8x RTX 6000 Pro Blackwell: GCloud

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Project scaffolded |
| 2026-04-03 | Codex | Created paper-grounded PRDs/tasks and synced reference repo |
| 2026-04-03 | Codex | Started `anima-autopilot` build; normalized project metadata, added Python 3.11/UV foundation, dataset catalog, submap schemas, and CUDA bootstrap script |
| 2026-04-03 | Codex | Completed PRD-02 task slice with DINO wrapper, keyframing policy, odometry/tracker path, mapping/compaction scaffold, and passing local tests |
| 2026-04-03 | Opus 4.6 | Completed PRD-03: loop closure, registration (FPFH+RANSAC+ICP), GTSAM PGO, fusion merge+refine, CLI orchestrator. Code review: fixed 6 bugs. All 32 tests pass. |
