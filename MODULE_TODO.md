# SLAM-COKO — Design & Implementation Checklist

## Paper: Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM
## ArXiv: 2604.00804
## Repo: https://github.com/lemonci/coko-slam

---

## Phase 1: Scaffold + Verification
- [x] Project structure created
- [x] Correct paper identified and planning docs grounded to it
- [x] Reference repo cloned into `repositories/coko-slam/`
- [x] Canonical `anima_slam_coko` package created
- [x] Python 3.11 + UV foundation scaffolded
- [x] Pydantic settings, dataset catalog, and submap schemas added
- [ ] Reference demo runs successfully
- [ ] Shared datasets accessibility confirmed

## Phase 2: Reproduce
- [ ] Core model implemented in `src/anima_slam_coko/`
- [ ] Training pipeline (scripts/train.py)
- [ ] Evaluation pipeline (scripts/eval.py)
- [ ] Metrics match paper (within ±5%)
- [ ] Dual-compute verified (MLX + CUDA)

## Phase 3: Adapt to Hardware
- [ ] ZED 2i data pipeline (if applicable)
- [ ] Unitree L2 LiDAR pipeline (if applicable)
- [ ] xArm 6 integration (if manipulation module)
- [ ] Real sensor inference test
- [ ] MLX inference port validated

## Phase 4: ANIMA Integration
- [ ] ROS2 bridge node
- [ ] Docker container builds and runs
- [ ] API endpoints defined
- [ ] Integration test with stack: ATLAS

## Shenzhen Demo Readiness
- [ ] Demo script works end-to-end
- [ ] Demo data prepared
- [ ] Demo runs in < 30 seconds
- [ ] Demo visuals are compelling
