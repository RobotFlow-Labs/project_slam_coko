"""Replica SLAM dataset loader matching reference repo format.

Loads prepared data from /mnt/forge-data/datasets/replica_slam/{scene}/agent_{id}/
with results/frame*.jpg, results/depth*.png, and traj.txt (4x4 matrices).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DEPTH_SCALE = 6553.5


class ReplicaSLAMDataset:
    """Loads a single agent's prepared Replica RGB-D data."""

    def __init__(
        self,
        agent_path: Path,
        *,
        height: int = 480,
        width: int = 640,
        fx: float = 600.0,
        fy: float = 600.0,
        cx: float = 319.5,
        cy: float = 239.5,
        depth_scale: float = DEPTH_SCALE,
        frame_limit: int = -1,
    ) -> None:
        self.agent_path = Path(agent_path)
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

        self.intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )

        results_dir = self.agent_path / "results"
        self.color_paths = sorted(results_dir.glob("frame*.jpg"))
        self.depth_paths = sorted(results_dir.glob("depth*.png"))

        self.poses: list[np.ndarray] = []
        self._load_poses(self.agent_path / "traj.txt")

        n = min(len(self.color_paths), len(self.depth_paths), len(self.poses))
        if frame_limit > 0:
            n = min(n, frame_limit)
        self.color_paths = self.color_paths[:n]
        self.depth_paths = self.depth_paths[:n]
        self.poses = self.poses[:n]
        self.frame_ids = list(range(n))

    def _load_poses(self, path: Path) -> None:
        for line in path.read_text().strip().split("\n"):
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                c2w = np.array(vals, dtype=np.float32).reshape(4, 4)
                self.poses.append(c2w)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (frame_id, rgb_HW3_uint8, depth_HW_float32, c2w_44)."""
        rgb = cv2.imread(str(self.color_paths[index]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / self.depth_scale

        return self.frame_ids[index], rgb, depth, self.poses[index]

    def get_frame_dict(self, index: int) -> dict:
        """Return frame as a dict for pipeline consumption."""
        fid, rgb, depth, c2w = self[index]
        return {
            "frame_id": fid,
            "rgb": rgb,
            "depth": depth,
            "c2w": c2w,
            "intrinsics": self.intrinsics,
        }


def load_scene(
    scene_name: str,
    *,
    slam_data_root: Path = Path("/mnt/forge-data/datasets/replica_slam"),
    **camera_kwargs,
) -> dict[int, ReplicaSLAMDataset]:
    """Load all agents for a scene.

    Returns:
        ``{agent_id: ReplicaSLAMDataset}``
    """
    scene_dir = slam_data_root / scene_name
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene not found: {scene_dir}")

    agents: dict[int, ReplicaSLAMDataset] = {}
    for agent_dir in sorted(scene_dir.glob("agent_*")):
        agent_id = int(agent_dir.name.split("_")[1])
        agents[agent_id] = ReplicaSLAMDataset(agent_dir, **camera_kwargs)

    if not agents:
        raise FileNotFoundError(f"No agent directories in {scene_dir}")

    return agents
