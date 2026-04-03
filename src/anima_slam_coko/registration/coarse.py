"""FPFH + RANSAC coarse registration (Paper Section 3.3).

Computes FPFH descriptors from point-cloud normals and runs RANSAC-based
alignment to produce an initial rigid transform between two submaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None  # type: ignore[assignment]


@dataclass(slots=True)
class CoarseResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondence_count: int


def _numpy_to_o3d_pcd(points: np.ndarray) -> "o3d.geometry.PointCloud":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def _compute_fpfh(
    pcd: "o3d.geometry.PointCloud",
    voxel_size: float,
) -> "o3d.pipelines.registration.Feature":
    radius_normal = voxel_size * 2.0
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return fpfh


def coarse_register(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    voxel_size: float = 0.05,
    distance_threshold: float | None = None,
    max_iterations: int = 100000,
    confidence: float = 0.999,
) -> CoarseResult:
    """Run FPFH + RANSAC coarse registration.

    Args:
        source_points: ``(N, 3)`` source cloud.
        target_points: ``(M, 3)`` target cloud.
        voxel_size: voxel size for downsampling and feature radius.
        distance_threshold: RANSAC distance threshold (default ``1.5 * voxel_size``).
        max_iterations: RANSAC iterations.
        confidence: RANSAC confidence.

    Returns:
        :class:`CoarseResult` with the estimated transform and fitness.
    """
    if o3d is None:
        raise ImportError("open3d is required for coarse registration")

    if distance_threshold is None:
        distance_threshold = voxel_size * 1.5

    src_pcd = _numpy_to_o3d_pcd(source_points)
    tgt_pcd = _numpy_to_o3d_pcd(target_points)

    src_down = src_pcd.voxel_down_sample(voxel_size)
    tgt_down = tgt_pcd.voxel_down_sample(voxel_size)

    if len(src_down.points) < 3 or len(tgt_down.points) < 3:
        return CoarseResult(
            transformation=np.eye(4, dtype=np.float32),
            fitness=0.0,
            inlier_rmse=float("inf"),
            correspondence_count=0,
        )

    src_fpfh = _compute_fpfh(src_down, voxel_size)
    tgt_fpfh = _compute_fpfh(tgt_down, voxel_size)

    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            distance_threshold
        ),
    ]

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iterations, confidence
        ),
    )

    return CoarseResult(
        transformation=np.asarray(result.transformation, dtype=np.float32),
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        correspondence_count=len(result.correspondence_set),
    )
