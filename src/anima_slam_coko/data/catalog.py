"""Dataset catalog for SLAM-COKO: scene names, agent IDs, and path builders."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

DatasetName = Literal["replica", "aria"]

SCENE_CATALOG: dict[DatasetName, tuple[str, ...]] = {
    "replica": ("office_0", "apart_0", "apart_1", "apart_2"),
    "aria": ("room0", "room1"),
}

DATASET_AGENT_IDS: dict[DatasetName, tuple[int, ...]] = {
    "replica": (0, 1),
    "aria": (0, 1, 2),
}

DATASET_DIR_NAMES: dict[DatasetName, str] = {
    "replica": "ReplicaMultiagent",
    "aria": "AriaMultiagent",
}


def build_manifest(
    dataset: DatasetName,
    scene: str,
    shared_volume: Path | str,
) -> dict[str, Path]:
    """Return a dict mapping logical names to filesystem paths for a scene."""
    root = Path(shared_volume) / DATASET_DIR_NAMES[dataset]
    return {
        "scene_root": root / scene,
        "dataset_root": root,
    }
