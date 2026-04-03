"""ICP refinement following coarse FPFH+RANSAC alignment (Paper Section 3.3).

Applies point-to-plane ICP to polish the rigid transform produced by the
coarse registration stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None  # type: ignore[assignment]


@dataclass(slots=True)
class ICPResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float


def icp_refine(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray,
    *,
    max_correspondence_distance: float = 0.05,
    max_iterations: int = 50,
    method: str = "point_to_plane",
) -> ICPResult:
    """Refine a coarse alignment with ICP.

    Args:
        source_points: ``(N, 3)`` source cloud.
        target_points: ``(M, 3)`` target cloud.
        init_transform: ``(4, 4)`` initial transform from coarse stage.
        max_correspondence_distance: ICP inlier distance.
        max_iterations: ICP iterations.
        method: ``"point_to_plane"`` or ``"point_to_point"``.

    Returns:
        :class:`ICPResult` with refined transform and fitness.
    """
    if o3d is None:
        raise ImportError("open3d is required for ICP refinement")

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(
        source_points.astype(np.float64)
    )

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(
        target_points.astype(np.float64)
    )

    if method == "point_to_plane":
        radius = max_correspondence_distance * 2.0
        src_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        tgt_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations
    )

    result = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        max_correspondence_distance,
        init_transform.astype(np.float64),
        estimation,
        criteria,
    )

    return ICPResult(
        transformation=np.asarray(result.transformation, dtype=np.float32),
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
    )
