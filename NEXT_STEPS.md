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

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: ALL 6 SCENES TRAINED, ready for HF export
- **MVP Readiness**: 75%
- **Accomplished**:
  1. PRD-01 Foundation ✅
  2. PRD-02 Core Local Agent ✅
  3. PRD-03 Server Fusion ✅ (loop closure, registration, PGO, merge, refine)
  4. PRD-04 Evaluation ✅ (metrics, bandwidth, gap reports)
  5. Code review: 12 bugs fixed across 2 rounds
  6. Shared Replica RGB-D rendering: 6 scenes × 500 frames at /mnt/forge-data/datasets/replica_rgbd/
  7. SLAM data prep: 6 scenes × 2 agents × 250 frames at /mnt/forge-data/datasets/replica_slam/
  8. Training pipeline end-to-end tested
  9. room0 training COMPLETE: 42 submaps, 1.85M gaussians, depth-L1=0.4795, 221.5s
  10. All 53 tests passing, ruff clean
- **TODO**:
  1. Run training on remaining 5 scenes (room1, room2, office0, office1, apartment_0)
  2. Run /anima-hf-strategy with TRT fp16+fp32 MANDATORY
  3. PRD-05 API & Docker
  4. PRD-06 ROS2 Integration
  5. PRD-07 Production Hardening
- **Blockers**: None

## 4. Shared Data Infrastructure
### Rendered RGB-D (for ALL SLAM modules)
- `/mnt/forge-data/datasets/replica_rgbd/` — 6 scenes × 500 frames (mesh→RGB-D)
- `/mnt/forge-data/datasets/replica_slam/` — 6 scenes × 2 agents × 250 frames (SLAM format)

### Training paths
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Checkpoints: `/mnt/artifacts-datai/checkpoints/project_slam_coko/`
- Logs: `/mnt/artifacts-datai/logs/project_slam_coko/`
- Reports: `/mnt/artifacts-datai/reports/project_slam_coko/`

## 5. Training Results
| Scene | Submaps | Gaussians | Loops | Depth-L1 | Bandwidth | Time | Status |
|-------|---------|-----------|-------|----------|-----------|------|--------|
| room0 | 42 | 1,850,652 | 1 | 0.4795 | 297.9 MB | 222s | ✅ |
| room1 | 39 | 1,044,551 | 1 | 0.5755 | 189.5 MB | 216s | ✅ |
| room2 | 46 | 1,445,703 | 2 | 0.4967 | 271.9 MB | 484s | ✅ |
| office0 | 49 | 756,968 | 1 | 0.5700 | 133.4 MB | 48s | ✅ |
| office1 | 47 | 593,876 | 1 | 0.5874 | 146.3 MB | 48s | ✅ |
| apartment_0 | 47 | 2,252,584 | 1 | 1.0126 | 277.7 MB | 69s | ✅ |

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Project scaffolded |
| 2026-04-03 | Codex | Created paper-grounded PRDs/tasks and synced reference repo |
| 2026-04-03 | Codex | PRD-01 + PRD-02 implementation |
| 2026-04-03 | Opus 4.6 | PRD-03 server fusion + PRD-04 evaluation |
| 2026-04-03 | Opus 4.6 | Code review (12 bugs fixed), Replica rendering (6 scenes), data prep, room0 training COMPLETE |
