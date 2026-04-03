#!/usr/bin/env python3
"""Render RGB-D sequences from Replica meshes for all SLAM modules.

Output: /mnt/forge-data/datasets/replica_rgbd/{scene}/
    rgb/       — PNG images (H, W, 3)
    depth/     — .npy depth maps (H, W) in meters
    traj.txt   — TUM-format trajectory (timestamp tx ty tz qx qy qz qw)
    intrinsics.txt — fx fy cx cy width height

Scenes: room_0, room_1, room_2, office0 (mapped from apartment_0),
        office1 (mapped from apartment_1), apartment_0 (mapped from frl_apartment_0)

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/render_replica_rgbd.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Replica mesh dir → output name mapping
SCENE_MAP: dict[str, str] = {
    "room_0": "room0",
    "room_1": "room1",
    "room_2": "room2",
    "apartment_0": "office0",
    "apartment_1": "office1",
    "frl_apartment_0": "apartment_0",
}

REPLICA_ROOT = Path("/mnt/forge-data/datasets/replica")
OUTPUT_ROOT = Path("/mnt/forge-data/datasets/replica_rgbd")

# Camera intrinsics matching the paper's Replica config
WIDTH, HEIGHT = 640, 480
FX, FY = 600.0, 600.0
CX, CY = 319.5, 239.5

# Trajectory: spiral + random walk inside the room
NUM_FRAMES = 500
SEED = 42


def _load_mesh(scene_dir: Path):
    """Load Replica mesh via trimesh (handles polygon PLY) → Open3D raycasting."""
    import open3d as o3d
    import trimesh

    tm = trimesh.load(str(scene_dir / "mesh.ply"), process=False)
    print(f"  Loaded {len(tm.vertices)} verts, {len(tm.faces)} faces")

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
    if tm.visual.vertex_colors is not None:
        colors = np.asarray(tm.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    bounds = np.asarray(o3d_mesh.vertices)
    center = bounds.mean(axis=0)
    extent = bounds.max(axis=0) - bounds.min(axis=0)

    return scene, o3d_mesh, center, extent


def _generate_trajectory(
    center: np.ndarray, extent: np.ndarray, num_frames: int, rng: np.random.RandomState
) -> list[np.ndarray]:
    """Generate a smooth camera trajectory inside the room.

    Spiral path around the room center with slight vertical variation,
    always looking toward the center. Stays within 80% of the room bounds.
    """
    from scipy.spatial.transform import Rotation as R

    poses = []
    radius = min(extent[0], extent[1]) * 0.3  # stay well inside

    for i in range(num_frames):
        t = i / num_frames
        angle = t * 4 * np.pi  # two full rotations

        # Position: spiral around center
        x = center[0] + radius * np.cos(angle) + rng.randn() * 0.02
        y = center[1] + radius * np.sin(angle) + rng.randn() * 0.02
        z = center[2] + 0.2 * np.sin(t * 2 * np.pi) + rng.randn() * 0.01

        pos = np.array([x, y, z])

        # Look toward center with slight offset
        look_target = center + rng.randn(3) * 0.1
        fwd = look_target - pos
        fwd /= np.linalg.norm(fwd) + 1e-8

        # Build rotation: camera looks along -z in OpenCV convention
        up_world = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_world)
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            up_world = np.array([0.0, 1.0, 0.0])
            right = np.cross(fwd, up_world)
            norm = np.linalg.norm(right)
        right /= norm
        up = np.cross(right, fwd)
        up /= np.linalg.norm(up)

        # Camera-to-world: columns are right, -up, fwd (OpenGL convention)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = -up
        c2w[:3, 2] = fwd
        c2w[:3, 3] = pos
        poses.append(c2w)

    return poses


def _render_frame(
    scene, c2w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB + depth from a camera pose using Open3D raycasting."""
    import open3d as o3d

    # Extrinsic = inverse of c2w
    w2c = np.linalg.inv(c2w)

    intrinsic = np.array([
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1],
    ], dtype=np.float64)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=intrinsic,
        extrinsic_matrix=w2c,
        width_px=WIDTH,
        height_px=HEIGHT,
    )

    result = scene.cast_rays(rays)
    depth = result["t_hit"].numpy().astype(np.float32)

    # Get primitive IDs for vertex color interpolation
    prim_ids = result["primitive_ids"].numpy()
    prim_uvs = result["primitive_uvs"].numpy()

    return depth, prim_ids, prim_uvs


def _extract_rgb(
    o3d_mesh, prim_ids: np.ndarray, prim_uvs: np.ndarray, depth: np.ndarray
) -> np.ndarray:
    """Extract RGB from vertex colors using barycentric interpolation."""
    triangles = np.asarray(o3d_mesh.triangles)
    vertex_colors = np.asarray(o3d_mesh.vertex_colors)  # (V, 3) in [0, 1]

    h, w = depth.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    valid = (prim_ids != 0xFFFFFFFF) & np.isfinite(depth) & (depth > 0)

    valid_prim = prim_ids[valid]
    valid_uvs = prim_uvs[valid]

    # Barycentric coords: u, v, w = uv[0], uv[1], 1-uv[0]-uv[1]
    u = valid_uvs[:, 0]
    v = valid_uvs[:, 1]
    w_coord = 1.0 - u - v

    tri_verts = triangles[valid_prim]  # (N, 3) vertex indices
    c0 = vertex_colors[tri_verts[:, 0]]  # (N, 3)
    c1 = vertex_colors[tri_verts[:, 1]]
    c2 = vertex_colors[tri_verts[:, 2]]

    interp_color = (
        w_coord[:, None] * c0 + u[:, None] * c1 + v[:, None] * c2
    )

    rgb[valid] = np.clip(interp_color, 0, 1)
    return (rgb * 255).astype(np.uint8)


def _pose_to_tum_line(timestamp: float, c2w: np.ndarray) -> str:
    """Convert c2w to TUM format: timestamp tx ty tz qx qy qz qw."""
    from scipy.spatial.transform import Rotation as R

    t = c2w[:3, 3]
    quat = R.from_matrix(c2w[:3, :3]).as_quat()  # [x, y, z, w]
    return f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"


def render_scene(mesh_name: str, out_name: str) -> None:
    """Render a single Replica scene to RGB-D."""
    scene_dir = REPLICA_ROOT / mesh_name
    out_dir = OUTPUT_ROOT / out_name

    if not scene_dir.exists():
        print(f"  SKIP: {scene_dir} does not exist")
        return

    # Check if already rendered
    if (out_dir / "done.marker").exists():
        existing = len(list((out_dir / "rgb").glob("*.png")))
        print(f"  SKIP: already rendered ({existing} frames)")
        return

    print(f"  Loading mesh from {scene_dir}...")
    raycasting_scene, o3d_mesh, center, extent = _load_mesh(scene_dir)

    print(f"  Generating {NUM_FRAMES}-frame trajectory...")
    rng = np.random.RandomState(SEED)
    poses = _generate_trajectory(center, extent, NUM_FRAMES, rng)

    # Create output dirs
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    traj_lines = []
    t0 = time.time()

    for i, c2w in enumerate(poses):
        depth, prim_ids, prim_uvs = _render_frame(raycasting_scene, c2w)
        rgb = _extract_rgb(o3d_mesh, prim_ids, prim_uvs, depth)

        # Replace inf/nan depth with 0
        depth[~np.isfinite(depth)] = 0.0

        timestamp = i * 0.033  # ~30 FPS

        # Save
        frame_name = f"{i:06d}"
        cv2.imwrite(str(rgb_dir / f"{frame_name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(str(depth_dir / f"{frame_name}.npy"), depth)

        traj_lines.append(_pose_to_tum_line(timestamp, c2w))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            valid_pct = 100 * np.count_nonzero(depth > 0) / depth.size
            print(f"    Frame {i + 1}/{NUM_FRAMES} — {fps:.1f} fps — {valid_pct:.0f}% valid depth")

    # Write trajectory
    (out_dir / "traj.txt").write_text("\n".join(traj_lines) + "\n")

    # Write intrinsics
    (out_dir / "intrinsics.txt").write_text(
        f"{FX} {FY} {CX} {CY} {WIDTH} {HEIGHT}\n"
    )

    # Write metadata
    (out_dir / "info.txt").write_text(
        f"source: {scene_dir}\n"
        f"frames: {NUM_FRAMES}\n"
        f"resolution: {WIDTH}x{HEIGHT}\n"
        f"fx: {FX}, fy: {FY}, cx: {CX}, cy: {CY}\n"
        f"seed: {SEED}\n"
    )

    # Done marker
    (out_dir / "done.marker").write_text("ok\n")

    elapsed = time.time() - t0
    print(f"  Done: {NUM_FRAMES} frames in {elapsed:.1f}s ({NUM_FRAMES / elapsed:.1f} fps)")
    print(f"  Output: {out_dir}")


def main() -> None:
    print("=" * 60)
    print("Replica RGB-D Renderer — Shared Infrastructure")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Scenes: {list(SCENE_MAP.values())}")
    print("=" * 60)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for mesh_name, out_name in SCENE_MAP.items():
        print(f"\n[{out_name}] Rendering from {mesh_name}...")
        try:
            render_scene(mesh_name, out_name)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All scenes complete. Other SLAM modules can now use:")
    print(f"  {OUTPUT_ROOT}/{{scene}}/rgb/")
    print(f"  {OUTPUT_ROOT}/{{scene}}/depth/")
    print(f"  {OUTPUT_ROOT}/{{scene}}/traj.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
