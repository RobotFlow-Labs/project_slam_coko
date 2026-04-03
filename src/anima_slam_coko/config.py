"""Typed settings and TOML loading for SLAM-COKO."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .data.catalog import DATASET_AGENT_IDS, SCENE_CATALOG, DatasetName, build_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "coko" / "base.toml"


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _normalize_paths(paths: tuple[str | Path | list[str | Path], ...]) -> list[Path]:
    if len(paths) == 1 and isinstance(paths[0], list):
        return [Path(path) for path in paths[0]]
    return [Path(path) for path in paths]


class ProjectSettings(BaseModel):
    name: str = "anima-slam-coko"
    codename: str = "SLAM-COKO"
    functional_name: str = "Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM"
    wave: int = 7
    paper_title: str = "Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM"
    paper_arxiv: str = "2604.00804"
    reference_repo: str = "https://github.com/lemonci/coko-slam"


class ComputeSettings(BaseModel):
    backend: Literal["auto", "mlx", "cuda", "cpu"] = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"


class RuntimeSettings(BaseModel):
    mode: Literal["rendered_depth", "camera_depth"] = "rendered_depth"
    multi_gpu: bool = False


class DataSettings(BaseModel):
    shared_volume: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets")
    repos_volume: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/repos/wave7")
    dataset: DatasetName = "replica"
    scene: str = "office_0"
    agent_ids: tuple[int, ...] = (0, 1)

    @model_validator(mode="after")
    def _validate_scene_and_agents(self) -> "DataSettings":
        if self.scene not in SCENE_CATALOG[self.dataset]:
            supported = ", ".join(SCENE_CATALOG[self.dataset])
            raise ValueError(
                f"Scene '{self.scene}' is not valid for dataset '{self.dataset}'. Supported: {supported}"
            )

        expected_agent_ids = DATASET_AGENT_IDS[self.dataset]
        if self.agent_ids != expected_agent_ids:
            raise ValueError(
                f"agent_ids for dataset '{self.dataset}' must be {expected_agent_ids}, got {self.agent_ids}"
            )
        return self


class MappingSettings(BaseModel):
    new_submap_every: int = 50
    map_every: int = 5
    iterations: int = 50
    new_submap_iterations: int = 1000
    new_submap_points_num: int = 60000
    new_submap_gradient_points_num: int = 5000
    alpha_threshold: float = 0.6
    pruning_threshold: float = 0.1
    active_compaction: bool = True
    prune_ratio: float = 0.05
    init_rho: float = 0.0005
    compaction_start: float = 0.7
    compaction_end: float = 0.95


class TrackingSettings(BaseModel):
    w_color_loss: float = 0.95
    iterations: int = 60
    cam_rot_lr: float = 0.0002
    cam_trans_lr: float = 0.002
    odometry_type: Literal["odometer", "gt", "const_speed"] = "odometer"
    odometer_method: Literal["point_to_plane", "hybrid"] = "point_to_plane"
    filter_outlier_depth: bool = True
    alpha_threshold: float = 0.98
    soft_alpha: bool = True
    mask_invalid_depth: bool = False


class LoopDetectionSettings(BaseModel):
    feature_extractor_name: Literal["dino", "netvlad"] = "dino"
    weights_path: str = "./dinov2-small"
    embed_size: int = 384
    feature_dist_threshold: float = 0.1
    device: str = "cpu"
    time_threshold: int = 0
    max_loops_per_frame: int = 1
    fitness_threshold: float = 0.35
    inlier_rmse_threshold: float = 0.1


class CameraSettings(BaseModel):
    height: int = 680
    width: int = 1200
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 599.5
    cy: float = 339.5
    depth_scale: float = 6553.5


class KeyframingSettings(BaseModel):
    active: bool = True
    threshold: float = 0.02


class SubmappingSettings(BaseModel):
    active: bool = True
    threshold: float = 0.05
    keyframe_num: int = 10


class SubmapSettings(BaseModel):
    anchor_data: Literal["render_depth", "camera_depth"] = "render_depth"
    initial_transformation_unknown: bool = True
    pgo_backend: Literal["gtsam"] = "gtsam"


class SlamCokoSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANIMA_SLAM_COKO_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    mapping: MappingSettings = Field(default_factory=MappingSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    loop_detection: LoopDetectionSettings = Field(default_factory=LoopDetectionSettings)
    camera: CameraSettings = Field(default_factory=CameraSettings)
    keyframing: KeyframingSettings = Field(default_factory=KeyframingSettings)
    submapping: SubmappingSettings = Field(default_factory=SubmappingSettings)
    submap: SubmapSettings = Field(default_factory=SubmapSettings)

    def resolve_scene_manifest(self):
        return build_manifest(self.data.dataset, self.data.scene, self.data.shared_volume)


def load_settings(
    *config_paths: str | Path | list[str | Path], overrides: dict[str, Any] | None = None
) -> SlamCokoSettings:
    normalized_paths = _normalize_paths(config_paths) if config_paths else [DEFAULT_CONFIG_PATH]
    merged: dict[str, Any] = {}
    for path in normalized_paths:
        merged = _deep_merge(merged, _read_toml(path))
    if overrides:
        merged = _deep_merge(merged, overrides)
    return SlamCokoSettings.model_validate(merged)
