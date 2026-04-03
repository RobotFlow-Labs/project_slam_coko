# SLAM-COKO — Asset Manifest

## Paper
- Title: Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM
- Short name: Coko-SLAM
- ArXiv: 2604.00804
- URL: https://arxiv.org/abs/2604.00804
- Authors: Monica M.Q. Li, Pierre-Yves Lajoie, Jialing Liu, Giovanni Beltrame
- Reference code: https://github.com/lemonci/coko-slam
- Note: `papers/2503.15868_CokO-SLAM.pdf` is not the Coko-SLAM paper. The planning documents below use arXiv `2604.00804` as the source of truth.

## Status: ALMOST

The project scaffold exists, the paper is verified, and the reference implementation is available under `repositories/coko-slam/`. The execution plan still needs implementation.

## Reference Code / External Assets

| Asset | Purpose | Source | Local Path | Status |
|---|---|---|---|---|
| Coko-SLAM reference repo | Canonical implementation and configs | https://github.com/lemonci/coko-slam | `repositories/coko-slam/` | DONE |
| DINOv2-Small weights | Keyframe and loop feature extraction | https://github.com/facebookresearch/dinov2 | `./dinov2-small` in reference config | REQUIRED |
| GTSAM | Pose graph optimization | https://github.com/borglab/gtsam | system / wheel / source build | REQUIRED |
| Open3D | RGB-D odometry, FPFH, RANSAC, ICP | https://www.open3d.org/ | Python dependency | REQUIRED |
| GaussianSPA paper / method | Real-time compaction reference | https://arxiv.org/abs/2411.06019 | external reference | REQUIRED |

## Datasets

| Dataset | Type | Split / Scenes | Source | Suggested Path | Status |
|---|---|---|---|---|---|
| ReplicaMultiagent | Synthetic RGB-D multi-agent | `office_0`, `apart_0`, `apart_1`, `apart_2` | https://huggingface.co/datasets/voviktyl/ReplicaMultiagent | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/ReplicaMultiagent` | REQUIRED |
| AriaDigitalTwin raw sequences | Real RGB-D source dataset | sequences listed in `repositories/coko-slam/scripts/prepare_aria_room_data.py` | https://www.projectaria.com/datasets/adt/ | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/AriaDigitalTwin` | REQUIRED |
| AriaMultiagent processed | Multi-agent evaluation set derived from ADT | `room0`, `room1` | produced by repo preprocessing script | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/AriaMultiagent` | REQUIRED |

## Hardware / Runtime Assumptions

| Item | Paper / Repo Evidence | Planning Assumption |
|---|---|---|
| GPU | Paper Section 4: NVIDIA H100 80GB for experiments; repo README mentions RTX 4070 and H100 | Primary implementation target is CUDA |
| Multi-agent GPU layout | Repo README: ideally `n + 1` GPUs for `n` agents | First build supports single-GPU debug mode and multi-GPU production mode |
| Camera modality | RGB-D everywhere in paper and repo | Core build remains RGB-D; ROS2 bridge adapts ZED 2i streams |

## Hyperparameters and Control Knobs

### Paper-grounded

| Parameter | Value | Source |
|---|---|---|
| Feature extractor | DINOv2-Small | Paper §3.1 |
| Keyframe criterion | add frame if `min ||phi(E) - phi(K)|| >= alpha` | Paper §3.1 |
| Keyframes per submap | 10 | Paper §3.1 |
| Compaction feasibility tolerance `epsilon` | 0.005 | Paper §3.2 |
| Compaction terminal iteration `T` | 950 | Paper §3.2 |
| Total submap optimization iterations | 1000 | Paper §3.2 |
| Compaction start iteration | 700 | Paper §3.2 |
| Loop registration | FPFH + RANSAC coarse, ICP fine | Paper §3.3 |
| PGO backend | GTSAM with robust line-process style objective | Paper Eq. (3), §3.3 |

### Reference-config values to reproduce paper runs

| Parameter | Replica | Aria | Source |
|---|---|---|---|
| `mapping.new_submap_every` | 50 | 20 | `repositories/coko-slam/configs/*/*_multiagent.yaml` |
| `mapping.iterations` | 50 | 100 | repo configs |
| `mapping.new_submap_iterations` | 1000 | 1000 | repo configs |
| `mapping.prune_ratio` | 0.05 | 0.05 | repo configs |
| `mapping.init_rho` | 0.0005 | 0.0005 | repo configs |
| `mapping.compaction_start` | 0.7 | 0.7 | repo configs |
| `mapping.compaction_end` | 0.95 | 0.95 | repo configs |
| `tracking.iterations` | 60 | 200 | repo configs |
| `tracking.w_color_loss` | 0.95 | 0.6 | repo configs |
| `keyframing_threshold` | 0.02 | 0.02 | repo configs |
| `submapping_threshold` | 0.05 | 0.05 | repo configs |
| `loop.feature_dist_threshold` | 0.1 | 0.1 | repo configs |
| `loop.max_loops_per_frame` | 1 | 1 | repo configs |

## Expected Metrics (Paper Targets)

### Rendering quality

| Benchmark | Mode | Metric | Paper Value | Our Target |
|---|---|---|---|---|
| Replica `office_0` | training, camera depth | PSNR | 39.287 | >= 38.5 |
| Replica `apart_0` | training, camera depth | PSNR | 36.634 | >= 35.5 |
| Replica `apart_1` | training, camera depth | PSNR | 29.189 | >= 28.0 |
| Replica `apart_2` | training, camera depth | PSNR | 31.072 | >= 30.0 |
| Aria `room0` | novel view, camera depth | PSNR | 19.080 | >= 18.0 |
| Aria `room1` | novel view, camera depth | PSNR | 24.176 | >= 23.0 |

### Communication budget

| Benchmark | Mode | Paper Value | Our Target |
|---|---|---|---|
| Replica transmitted data reduction | rendered / camera depth vs prior work | 85-95% | >= 85% |
| Room0 total data per agent | camera depth | 55 / 79 / 65 MB | <= 90 MB per agent |
| Room1 total data per agent | camera depth | 66 / 71 / 50 MB | <= 90 MB per agent |

## Planned Local Paths

| Path | Purpose |
|---|---|
| `src/anima_slam_coko/` | New canonical package for this module |
| `configs/coko/` | Reproduction and deployment configs |
| `runs/replica/` | Offline experiment artifacts |
| `runs/aria/` | Real-world experiment artifacts |
| `artifacts/maps/` | Exported submaps and merged maps |
| `artifacts/reports/` | Evaluation tables, plots, and build reports |
