"""Convert rendered or camera depth maps to point clouds for registration (Paper Section 3.3).

In rendered-depth mode the server renders a depth image from the first-keyframe
viewpoint of each submap using the Gaussian model, then lifts it to a 3-D point
cloud.  In camera-depth mode the original sensor depth is used directly.
"""

from __future__ import annotations

import numpy as np


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray | None = None,
    depth_scale: float = 1.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> np.ndarray:
    """Lift a depth image to a world-frame point cloud.

    Args:
        depth: ``(H, W)`` depth image.
        intrinsics: ``(3, 3)`` camera intrinsics.
        c2w: optional ``(4, 4)`` camera-to-world transform.
        depth_scale: multiplicative scale applied to raw depth values.
        min_depth: minimum valid depth.
        max_depth: maximum valid depth.

    Returns:
        ``(M, 3)`` world-space points.
    """
    depth = np.asarray(depth, dtype=np.float32) * depth_scale
    mask = (depth > min_depth) & (depth < max_depth)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth[ys, xs]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)

    if c2w is not None:
        c2w = np.asarray(c2w, dtype=np.float32)
        pts_h = np.concatenate(
            [pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1
        )
        pts_cam = (c2w @ pts_h.T).T[:, :3]

    return pts_cam


def submap_to_point_cloud(
    submap: dict,
    intrinsics: np.ndarray,
    *,
    mode: str = "rendered_depth",
) -> np.ndarray:
    """Extract a point cloud from a submap record.

    In *rendered_depth* mode the ``rendered_depth`` payload is used.
    In *camera_depth* mode the ``camera_depth`` payload is used.
    The depth is lifted using the first keyframe pose ``submap_c2ws[0]``.
    """
    if mode == "rendered_depth":
        depth = submap.get("rendered_depth")
        if depth is None:
            raise ValueError("rendered_depth payload missing from submap")
    elif mode == "camera_depth":
        depth = submap.get("camera_depth")
        if depth is None:
            raise ValueError("camera_depth payload missing from submap")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    depth = np.asarray(depth, dtype=np.float32)
    c2w = np.asarray(submap["submap_c2ws"][0], dtype=np.float32)
    return depth_to_point_cloud(depth, intrinsics, c2w=c2w)


def gaussian_xyz_as_cloud(submap: dict) -> np.ndarray:
    """Return Gaussian centres as a fallback point cloud."""
    return np.asarray(submap["gaussian_xyz"], dtype=np.float32).reshape(-1, 3)
