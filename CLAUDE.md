# SLAM-COKO

## Paper
**CokO-SLAM: Multi-Agent Collaborative GS SLAM**
arXiv: https://arxiv.org/abs/2503.15868

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
- Python: >=3.10
- Config: TOML + Pydantic BaseSettings
- Lint: ruff
- Git commit prefix: [SLAM-COKO]
