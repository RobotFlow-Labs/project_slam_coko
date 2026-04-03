#!/usr/bin/env python3
"""Convert rendered Replica RGB-D to reference SLAM format.

Input:  /mnt/forge-data/datasets/replica_rgbd/{scene}/rgb/*.png, depth/*.npy, traj.txt
Output: /mnt/forge-data/datasets/replica_slam/{scene}/agent_{id}/results/frame*.jpg, depth*.png, traj.txt

This is shared infrastructure — all SLAM modules use this output.

Usage:
    python scripts/prepare_slam_data.py                    # all scenes
    python scripts/prepare_slam_data.py --scene room0      # single scene
    python scripts/prepare_slam_data.py --num-agents 3     # 3-agent split
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

INPUT_ROOT = Path("/mnt/forge-data/datasets/replica_rgbd")
OUTPUT_ROOT = Path("/mnt/forge-data/datasets/replica_slam")

SCENES = ["room0", "room1", "room2", "office0", "office1", "apartment_0"]
DEPTH_SCALE = 6553.5  # Reference repo default for Replica


def parse_tum_traj(traj_path: Path) -> list[np.ndarray]:
    """Parse TUM-format trajectory (ts tx ty tz qx qy qz qw) → list of 4x4 c2w."""
    poses = []
    for line in traj_path.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        # TUM format: timestamp tx ty tz qx qy qz qw
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        c2w[:3, 3] = [tx, ty, tz]
        poses.append(c2w)
    return poses


def write_matrix_traj(poses: list[np.ndarray], out_path: Path) -> None:
    """Write poses as 4x4 matrix rows (16 floats per line), matching reference format."""
    lines = []
    for c2w in poses:
        flat = c2w.flatten()
        lines.append(" ".join(f"{v:.8f}" for v in flat))
    out_path.write_text("\n".join(lines) + "\n")


def convert_rgb(src_dir: Path, dst_dir: Path, frame_indices: list[int]) -> int:
    """Convert PNG → JPG with reference naming: frame000000.jpg."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for local_idx, global_idx in enumerate(frame_indices):
        src = src_dir / f"{global_idx:06d}.png"
        dst = dst_dir / f"frame{local_idx:06d}.jpg"
        if not src.exists():
            print(f"    WARN: missing {src}")
            continue
        img = cv2.imread(str(src))
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    return count


def convert_depth(src_dir: Path, dst_dir: Path, frame_indices: list[int]) -> int:
    """Convert NPY float32 meters → 16-bit PNG with depth_scale encoding."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for local_idx, global_idx in enumerate(frame_indices):
        src = src_dir / f"{global_idx:06d}.npy"
        dst = dst_dir / f"depth{local_idx:06d}.png"
        if not src.exists():
            print(f"    WARN: missing {src}")
            continue
        depth_m = np.load(str(src))
        # Clamp and convert to uint16
        depth_m = np.clip(depth_m, 0, 65535.0 / DEPTH_SCALE)
        depth_u16 = (depth_m * DEPTH_SCALE).astype(np.uint16)
        cv2.imwrite(str(dst), depth_u16)
        count += 1
    return count


def prepare_scene(
    scene: str, num_agents: int = 2, overwrite: bool = False
) -> bool:
    """Prepare a single scene for SLAM training."""
    in_dir = INPUT_ROOT / scene
    out_dir = OUTPUT_ROOT / scene

    if not in_dir.exists() or not (in_dir / "done.marker").exists():
        print(f"  SKIP: {scene} — not rendered or incomplete")
        return False

    if (out_dir / "done.marker").exists() and not overwrite:
        print(f"  SKIP: {scene} — already prepared")
        return True

    # Clean previous output
    if out_dir.exists():
        shutil.rmtree(out_dir)

    t0 = time.time()

    # Load poses
    traj_path = in_dir / "traj.txt"
    poses = parse_tum_traj(traj_path)
    total_frames = len(poses)
    print(f"  Loaded {total_frames} poses")

    # Count available frames
    rgb_count = len(list((in_dir / "rgb").glob("*.png")))
    depth_count = len(list((in_dir / "depth").glob("*.npy")))
    total_frames = min(total_frames, rgb_count, depth_count)
    print(f"  Available: {rgb_count} RGB, {depth_count} depth, {total_frames} usable")

    # Split frames across agents
    frames_per_agent = total_frames // num_agents
    for agent_id in range(num_agents):
        start = agent_id * frames_per_agent
        end = start + frames_per_agent if agent_id < num_agents - 1 else total_frames
        frame_indices = list(range(start, end))
        agent_poses = poses[start:end]

        agent_dir = out_dir / f"agent_{agent_id}"
        results_dir = agent_dir / "results"

        print(f"  Agent {agent_id}: frames {start}-{end - 1} ({len(frame_indices)} frames)")

        # Convert RGB
        n_rgb = convert_rgb(in_dir / "rgb", results_dir, frame_indices)

        # Convert depth
        n_depth = convert_depth(in_dir / "depth", results_dir, frame_indices)

        # Write trajectory files (both formats for unknown/known pose modes)
        write_matrix_traj(agent_poses, agent_dir / "traj.txt")
        write_matrix_traj(agent_poses, agent_dir / "traj_converted.txt")

        print(f"    -> {n_rgb} RGB, {n_depth} depth, {len(agent_poses)} poses")

    # Write intrinsics (copy from source)
    intrinsics_src = in_dir / "intrinsics.txt"
    if intrinsics_src.exists():
        shutil.copy2(str(intrinsics_src), str(out_dir / "intrinsics.txt"))

    # Write metadata
    (out_dir / "info.txt").write_text(
        f"source: {in_dir}\n"
        f"total_frames: {total_frames}\n"
        f"num_agents: {num_agents}\n"
        f"frames_per_agent: {frames_per_agent}\n"
        f"depth_scale: {DEPTH_SCALE}\n"
    )

    # Done marker
    (out_dir / "done.marker").write_text("ok\n")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s → {out_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Replica RGB-D for SLAM")
    parser.add_argument("--scene", type=str, default=None, help="Single scene to prepare")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing")
    args = parser.parse_args()

    scenes = [args.scene] if args.scene else SCENES

    print("=" * 60)
    print("Replica SLAM Data Preparation")
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Scenes: {scenes}")
    print(f"Agents: {args.num_agents}")
    print("=" * 60)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = {}
    for scene in scenes:
        print(f"\n[{scene}]")
        results[scene] = prepare_scene(scene, args.num_agents, args.overwrite)

    print("\n" + "=" * 60)
    print("Summary:")
    for scene, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {scene}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
