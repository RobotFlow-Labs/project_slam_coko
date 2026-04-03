# SLAM-COKO — ANIMA Module

> **Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM**
> Paper: [arXiv:2604.00804](https://arxiv.org/abs/2604.00804)

Part of the [ANIMA Intelligence Compiler Suite](https://github.com/RobotFlow-Labs) by AIFLOW LABS LIMITED.

## Domain
SLAM

## Status
- [x] Paper-grounded PRDs and tasks created
- [x] Python 3.11 + UV foundation scaffolded
- [x] Dataset catalog and submap schemas added
- [ ] Core tracker / mapper runtime
- [ ] Loop closure and GTSAM pose graph
- [ ] Replica and Aria evaluation
- [ ] ROS2 bridge, API, and Docker runtime

## Quick Start
```bash
cd project_slam_coko
uv venv .venv --python python3.11
uv sync --extra dev
uv run pytest tests/config/test_settings.py tests/io/test_submap_schema.py -v
```

## CUDA Bootstrap
Use the local Mac scaffold for prebuild work, then bring up the Linux GPU environment with:

```bash
bash scripts/bootstrap_cuda_uv.sh
```

This keeps the repository UV-native while deferring CUDA wheel resolution to the Linux host.

## License
MIT — AIFLOW LABS LIMITED
