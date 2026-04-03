# SLAM-COKO PRD Suite

This folder contains the execution PRDs for building a paper-faithful ANIMA implementation of Coko-SLAM from arXiv `2604.00804` and the inspected reference repository under `repositories/coko-slam/`.

## Build Order

1. `PRD-01-foundation.md`
2. `PRD-02-core-model.md`
3. `PRD-03-inference.md`
4. `PRD-04-evaluation.md`
5. `PRD-05-api-docker.md`
6. `PRD-06-ros2.md`
7. `PRD-07-production.md`

## Execution Rules

- Keep the first implementation pass very close to the paper.
- Treat CUDA execution as the mainline.
- Use the reference repo only as implementation guidance, not as a drop-in copy.
- Reproduce the paper’s communication and rendering tables before broad adaptation work.
