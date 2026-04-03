# SLAM-COKO — Pipeline Map

## Paper Pipeline -> Implementation Mapping

| Paper Component | Paper Ref | Reference Repo | Planned File(s) | PRD |
|---|---|---|---|---|
| RGB-D agent input and per-agent runtime | §1, Fig. 2 | `run_slam.py`, `src/entities/agent.py` | `src/anima_slam_coko/runtime/agent_runtime.py` | PRD-02 |
| Visual odometry pose initialization | §2.1 recap of MAGiC-SLAM | `src/entities/visual_odometer.py`, `src/entities/tracker.py` | `src/anima_slam_coko/tracking/visual_odometer.py`, `src/anima_slam_coko/tracking/tracker.py` | PRD-02 |
| DINOv2-Small feature extraction | §3.1 | `src/entities/loop_detection/feature_extractors.py` | `src/anima_slam_coko/features/dino.py` | PRD-02 |
| Feature-distance keyframing | §3.1 | `src/entities/agent.py` | `src/anima_slam_coko/keyframing/policy.py` | PRD-02 |
| Feature-distance submapping | §3.1 | `src/entities/agent.py` | `src/anima_slam_coko/submapping/policy.py` | PRD-02 |
| Gaussian seeding and local mapping | §2.1 recap, §3.2 | `src/entities/mapper.py`, `src/entities/gaussian_model.py` | `src/anima_slam_coko/mapping/mapper.py`, `src/anima_slam_coko/mapping/gaussian_state.py` | PRD-02 |
| Optimization-sparsification compaction | §3.2, Algorithm 1 | `src/entities/mapper.py`, `src/utils/optimizing_spa.py` | `src/anima_slam_coko/mapping/compaction.py` | PRD-02 |
| Submap serialization for rendered-depth mode | §1, §3.3 | `save_current_submap_render` in `agent.py` | `src/anima_slam_coko/io/submap_store.py` | PRD-02 |
| Loop detection from submap feature vectors | §3.3 | `src/entities/loop_detection/loop_detector.py` | `src/anima_slam_coko/loop_closure/detector.py` | PRD-03 |
| Rendered depth -> point cloud | §3.3 | `run_visualization.py`, `utils/*` | `src/anima_slam_coko/loop_closure/rendered_depth.py` | PRD-03 |
| FPFH + RANSAC coarse registration | §3.3 | `src/utils/magic_slam_utils.py` | `src/anima_slam_coko/registration/coarse.py` | PRD-03 |
| ICP fine registration | §3.3 | `src/utils/magic_slam_utils.py` | `src/anima_slam_coko/registration/fine.py` | PRD-03 |
| Pose graph optimization with GTSAM | Eq. (3), §3.3 | `src/entities/gtsam_pose_graph.py` | `src/anima_slam_coko/pgo/gtsam_solver.py` | PRD-03 |
| Merge and refine global Gaussian map | §3.3 | `src/utils/magic_slam_utils.py` | `src/anima_slam_coko/fusion/merge.py`, `src/anima_slam_coko/fusion/refine.py` | PRD-03 |
| Replica training-view evaluation | §4, Table 1 | `src/utils/mapping_eval.py` | `src/anima_slam_coko/eval/replica.py` | PRD-04 |
| Aria novel-view evaluation | §4, Table 2 | `src/utils/mapping_eval.py` | `src/anima_slam_coko/eval/aria.py` | PRD-04 |
| Communication and keyframe audits | §4, Tables 3-5 | ad hoc in repo outputs | `src/anima_slam_coko/eval/bandwidth.py` | PRD-04 |
| Service API for runs and reports | ANIMA serving requirement | none in paper repo | `src/anima_slam_coko/api/app.py` | PRD-05 |
| CUDA Docker runtime | deployment requirement | none in paper repo | `Dockerfile`, `docker-compose.yml` | PRD-05 |
| ROS2 online bridge | ANIMA stack requirement | none in paper repo | `src/anima_slam_coko/ros2/bridge_node.py` | PRD-06 |
| Production orchestration and safety gates | ANIMA production requirement | none in paper repo | `scripts/validate_release.sh`, `artifacts/reports/production_checklist.md` | PRD-07 |

## Data Flow

1. `RGB-D frame -> DINOv2 feature`
2. `DINOv2 feature -> keyframe decision`
3. `keyframe -> local odometry + Gaussian mapper`
4. `mapper -> compacted submap checkpoint + keyframe feature summary`
5. `server -> inter/intra loop detection`
6. `loop candidates -> rendered/camera depth point clouds -> FPFH + RANSAC -> ICP`
7. `validated loop edges + odometry edges -> GTSAM PGO`
8. `optimized submap poses -> merged global Gaussian map -> refinement`
9. `merged map -> rendering metrics + communication metrics + exports`

## Tensor / Data Contracts

| Artifact | Shape / Type | Notes |
|---|---|---|
| RGB frame | `UInt8[H,W,3]` | Replica `1200x680`, Aria `512x512` in reference configs |
| Depth frame | `Float32[H,W]` | meters after `depth_scale` conversion |
| DINO feature | `Float32[1,384]` | DINOv2-Small embedding |
| Camera pose | `Float32[4,4]` | `c2w` homogeneous transform |
| Gaussian xyz | `Float32[N,3]` | per-submap |
| Gaussian rotation | `Float32[N,4]` | unit quaternion |
| Gaussian scaling | `Float32[N,3]` | isotropic regularization applied in loss |
| Gaussian opacity | `Float32[N,1]` | compaction target |
| Submap keyframe ids | `Int64[K]` | `K <= 10` target |
| Loop registration transform | `Float64[4,4]` | RANSAC / ICP output |

## Build Notes

- The current scaffold still uses the placeholder package name `anima_amaterasu`; PRD-01 treats package normalization to `anima_slam_coko` as the first foundation step.
- The paper pipeline remains CUDA-first. MLX or CPU-only fallbacks are out of scope for matching the paper and move to later ANIMA adaptation work.
