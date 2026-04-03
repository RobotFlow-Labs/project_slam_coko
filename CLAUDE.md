# SLAM-COKO

## Paper
**Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM**
arXiv: https://arxiv.org/abs/2604.00804

## Module Identity
- Codename: SLAM-COKO
- Domain: SLAM
- Part of ANIMA Intelligence Compiler Suite

## Structure
```
project_slam_coko/
├── pyproject.toml
├── configs/
├── src/anima_slam_coko/
├── tests/
├── scripts/
├── papers/          # Paper PDF
├── CLAUDE.md        # This file
├── NEXT_STEPS.md
├── ASSETS.md
└── PRD.md
```

## Commands
```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Conventions
- Package manager: uv (never pip)
- Build backend: hatchling
- Python: >=3.11,<3.12
- Config: TOML + Pydantic BaseSettings
- Lint: ruff
- Git commit prefix: [SLAM-COKO]
