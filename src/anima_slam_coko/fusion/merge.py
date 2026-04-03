"""Submap concatenation and duplicate-Gaussian pruning (Paper Section 3.3).

After PGO the server transforms each submap's Gaussians into the optimised
global frame, concatenates them, and removes co-visible duplicates in the
overlap regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anima_slam_coko.mapping.gaussian_state import GaussianState


@dataclass(slots=True)
class MergeResult:
    state: GaussianState
    total_submaps: int
    pre_prune_count: int
    post_prune_count: int


def _transform_points(
    xyz: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Apply a (4,4) rigid transform to (N,3) points."""
    xyz = np.asarray(xyz, dtype=np.float32)
    T = np.asarray(transform, dtype=np.float32)
    pts_h = np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1
    )
    return (T @ pts_h.T).T[:, :3]


def _transform_rotations(
    rotations: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Rotate quaternion (w,x,y,z) columns by the rotation part of *transform*."""
    from scipy.spatial.transform import Rotation as R

    rot_matrix = np.asarray(transform[:3, :3], dtype=np.float64)
    delta = R.from_matrix(rot_matrix)

    quats_xyzw = rotations[:, [1, 2, 3, 0]].astype(np.float64)
    original = R.from_quat(quats_xyzw)
    combined = delta * original
    out_xyzw = combined.as_quat().astype(np.float32)
    return out_xyzw[:, [3, 0, 1, 2]]


def _prune_duplicates(
    state: GaussianState,
    voxel_size: float = 0.02,
) -> None:
    """Remove co-located Gaussians, keeping the one with highest opacity.

    Uses structured-array voxel keys (collision-free) and vectorized
    sort-based groupby (O(N log N), no Python loops).
    """
    if state.size == 0:
        return

    # Collision-free voxel keys via structured array
    indices = np.floor(state.xyz / voxel_size).astype(np.int64)
    keys = np.ascontiguousarray(indices).view(
        np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    ).reshape(-1)

    # Sort by voxel key; within each voxel keep the highest-opacity Gaussian
    opacity_flat = state.opacity[:, 0]
    # lexsort: secondary key first → sort by key ascending, then by -opacity ascending
    # so the first element in each group has the highest opacity
    order = np.lexsort((-opacity_flat, keys))
    sorted_keys = keys[order]

    # Mark the first occurrence in each group (= highest opacity per voxel)
    first_mask = np.empty(len(sorted_keys), dtype=bool)
    first_mask[0] = True
    first_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]

    if first_mask.all():
        return  # nothing to prune

    keep_indices = order[first_mask]
    keep = np.zeros(state.size, dtype=bool)
    keep[keep_indices] = True
    state.prune_mask(~keep)


def merge_submaps(
    agents_submaps: dict[int, list[dict]],
    optimized_poses: dict[int, np.ndarray],
    *,
    prune_voxel_size: float = 0.02,
) -> MergeResult:
    """Concatenate all submaps in the global frame and prune duplicates.

    Args:
        agents_submaps: ``{agent_id: [submap_dict, ...]}``.
        optimized_poses: ``{agent_id: (S, 4, 4)}`` from PGO.
        prune_voxel_size: spatial resolution for duplicate removal.

    Returns:
        :class:`MergeResult` with the merged Gaussian state.
    """
    merged = GaussianState()
    total_submaps = 0

    for agent_id in sorted(agents_submaps.keys()):
        submaps = agents_submaps[agent_id]
        poses = optimized_poses[agent_id]  # (S, 4, 4)

        for i, sub in enumerate(submaps):
            T = np.asarray(poses[i], dtype=np.float32)
            xyz = np.asarray(sub["gaussian_xyz"], dtype=np.float32).reshape(-1, 3)
            opacity = np.asarray(sub["gaussian_opacity"], dtype=np.float32).reshape(-1, 1)
            scale = np.asarray(sub["gaussian_scale"], dtype=np.float32).reshape(-1, 3)
            rotation = np.asarray(sub["gaussian_rotation"], dtype=np.float32).reshape(-1, 4)
            features = np.asarray(sub["gaussian_features"], dtype=np.float32)
            if features.ndim == 1:
                feat_dim = merged.features.shape[1] if merged.features.shape[0] > 0 else 2
                features = features.reshape(-1, feat_dim)

            world_xyz = _transform_points(xyz, T)
            world_rot = _transform_rotations(rotation, T)

            count = world_xyz.shape[0]
            merged.xyz = np.vstack([merged.xyz, world_xyz])
            merged.opacity = np.vstack([merged.opacity, opacity])
            merged.scale = np.vstack([merged.scale, scale])
            merged.rotation = np.vstack([merged.rotation, world_rot])
            merged.features = np.vstack([merged.features, features])
            merged.colors = np.vstack(
                [merged.colors, np.zeros((count, 3), dtype=np.float32)]
            )
            total_submaps += 1

    pre_prune = merged.size
    _prune_duplicates(merged, voxel_size=prune_voxel_size)

    return MergeResult(
        state=merged,
        total_submaps=total_submaps,
        pre_prune_count=pre_prune,
        post_prune_count=merged.size,
    )
