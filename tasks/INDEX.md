# SLAM-COKO Task Index

## Build Order

1. [x] [PRD-0101](PRD-0101.md) Normalize metadata and package boundary
2. [x] [PRD-0102](PRD-0102.md) Add settings and dataset catalog
3. [x] [PRD-0103](PRD-0103.md) Define submap schemas and schema tests
4. [x] [PRD-0201](PRD-0201.md) Implement DINO feature extraction and keyframing policy
5. [x] [PRD-0202](PRD-0202.md) Implement RGB-D odometry and tracker
6. [x] [PRD-0203](PRD-0203.md) Implement mapper, compaction, and checkpoint writers
7. [PRD-0301](PRD-0301.md) Implement loop candidate generation
8. [PRD-0302](PRD-0302.md) Implement rendered-depth and camera-depth registration
9. [PRD-0303](PRD-0303.md) Implement GTSAM PGO, fusion, and CLI orchestration
10. [PRD-0401](PRD-0401.md) Build Replica evaluation runner
11. [PRD-0402](PRD-0402.md) Build Aria evaluation runner
12. [PRD-0403](PRD-0403.md) Add bandwidth audit and paper gap report
13. [PRD-0501](PRD-0501.md) Expose FastAPI run models and create-run endpoint
14. [PRD-0502](PRD-0502.md) Add job execution and artifact endpoints
15. [PRD-0503](PRD-0503.md) Package CUDA Docker runtime and smoke tests
16. [PRD-0601](PRD-0601.md) Implement ROS2 message adapters
17. [PRD-0602](PRD-0602.md) Integrate ROS2 bridge node with runtime
18. [PRD-0603](PRD-0603.md) Add ROS2 launch and replay validation
19. [PRD-0701](PRD-0701.md) Add runtime telemetry and bandwidth counters
20. [PRD-0702](PRD-0702.md) Implement release gates
21. [PRD-0703](PRD-0703.md) Finalize production checklist and validator script

## Dependency Notes

- PRD-01 tasks unlock the rest of the repo.
- PRD-02 tasks must land before PRD-03.
- PRD-04 depends on a working PRD-03 pipeline.
- PRD-05 and PRD-06 wrap the stable pipeline, not vice versa.
- PRD-07 only starts once evaluation and service surfaces exist.
