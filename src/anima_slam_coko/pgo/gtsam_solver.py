"""GTSAM-based Pose Graph Optimization for multi-agent SLAM (Paper Eq. 3, Section 3.3).

Builds a factor graph with odometry and loop-closure between-factors, applies a
prior to the first node of the first agent, and optimises with
Levenberg-Marquardt.  A pure-numpy fallback is provided when ``gtsam`` is not
installed, enabling unit-testing without the C++ dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import gtsam

    _HAS_GTSAM = True
except ImportError:
    _HAS_GTSAM = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PGOEdge:
    source_node: int
    target_node: int
    relative_pose: np.ndarray  # (4,4)
    sigma_rot: float = 0.1
    sigma_trans: float = 0.1


@dataclass(slots=True)
class PGOResult:
    optimized_poses: dict[int, np.ndarray]  # agent_id -> (S, 4, 4)
    initial_error: float = 0.0
    final_error: float = 0.0
    num_factors: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_gtsam_pose3(T: np.ndarray) -> "gtsam.Pose3":
    rotation = gtsam.Rot3(T[:3, :3].astype(np.float64))
    translation = gtsam.Point3(T[:3, 3].astype(np.float64))
    return gtsam.Pose3(rotation, translation)


def _gtsam_pose3_to_numpy(pose: "gtsam.Pose3") -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = pose.rotation().matrix().astype(np.float32)
    T[:3, 3] = pose.translation().astype(np.float32)
    return T


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class GTSAMSolver:
    """Wraps GTSAM NonlinearFactorGraph for multi-agent submap PGO."""

    def __init__(
        self,
        odometry_sigma_rot: float = 0.1,
        odometry_sigma_trans: float = 0.1,
        loop_sigma_rot: float = 0.3,
        loop_sigma_trans: float = 0.2,
        prior_sigma: float = 1e-6,
        max_iterations: int = 100,
    ) -> None:
        if not _HAS_GTSAM:
            raise ImportError(
                "gtsam is required for GTSAMSolver. "
                "Install with: pip install gtsam"
            )
        self.odometry_sigma_rot = odometry_sigma_rot
        self.odometry_sigma_trans = odometry_sigma_trans
        self.loop_sigma_rot = loop_sigma_rot
        self.loop_sigma_trans = loop_sigma_trans
        self.prior_sigma = prior_sigma
        self.max_iterations = max_iterations

    def optimize(
        self,
        agents_submaps: dict[int, list[dict]],
        loops: list,
    ) -> PGOResult:
        """Build and solve the pose graph.

        Args:
            agents_submaps: ``{agent_id: [submap_dict, ...]}`` where each
                submap_dict contains at minimum ``submap_start_frame_id``
                and ``submap_c2ws`` (array of ``(4,4)`` poses).
            loops: list of loop objects with ``source_agent_id``,
                ``source_frame_id``, ``target_agent_id``,
                ``target_frame_id``, ``transformation``, and ``fitness``.

        Returns:
            :class:`PGOResult` with optimised poses per agent.
        """
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        graph_info: dict[tuple[int, int], int] = {}

        # -- Odometry edges --
        odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([
                self.odometry_sigma_rot,
                self.odometry_sigma_rot,
                self.odometry_sigma_rot,
                self.odometry_sigma_trans,
                self.odometry_sigma_trans,
                self.odometry_sigma_trans,
            ])
        )

        first_node_id: int | None = None
        for agent_id in sorted(agents_submaps.keys()):
            submaps = agents_submaps[agent_id]
            for i, sub in enumerate(submaps):
                node_id = len(graph_info)
                start_frame = sub["submap_start_frame_id"]
                graph_info[(agent_id, start_frame)] = node_id

                c2w = np.asarray(sub["submap_c2ws"][0], dtype=np.float64)
                initial.insert(node_id, _numpy_to_gtsam_pose3(c2w))

                if first_node_id is None:
                    first_node_id = node_id

                if i > 0:
                    prev_sub = submaps[i - 1]
                    prev_c2w = np.asarray(
                        prev_sub["submap_c2ws"][0], dtype=np.float64
                    )
                    rel = np.linalg.inv(prev_c2w) @ c2w
                    prev_id = graph_info[
                        (agent_id, prev_sub["submap_start_frame_id"])
                    ]
                    graph.add(
                        gtsam.BetweenFactorPose3(
                            prev_id,
                            node_id,
                            _numpy_to_gtsam_pose3(rel),
                            odometry_noise,
                        )
                    )

        # -- Prior on first node --
        if first_node_id is not None:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.full(6, self.prior_sigma)
            )
            graph.add(
                gtsam.PriorFactorPose3(
                    first_node_id,
                    initial.atPose3(first_node_id),
                    prior_noise,
                )
            )

        # -- Loop closure edges --
        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([
                self.loop_sigma_rot,
                self.loop_sigma_rot,
                self.loop_sigma_rot,
                self.loop_sigma_trans,
                self.loop_sigma_trans,
                self.loop_sigma_trans,
            ])
        )
        for loop in loops:
            src_key = (loop.source_agent_id, loop.source_frame_id)
            tgt_key = (loop.target_agent_id, loop.target_frame_id)
            if src_key not in graph_info or tgt_key not in graph_info:
                continue
            src_id = graph_info[src_key]
            tgt_id = graph_info[tgt_key]
            T_loop = np.asarray(loop.transformation, dtype=np.float64)
            graph.add(
                gtsam.BetweenFactorPose3(
                    src_id,
                    tgt_id,
                    _numpy_to_gtsam_pose3(np.linalg.inv(T_loop)),
                    loop_noise,
                )
            )

        # -- Optimise --
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.max_iterations)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()

        initial_error = float(graph.error(initial))
        final_error = float(graph.error(result))

        # -- Extract poses --
        optimized: dict[int, np.ndarray] = {}
        for agent_id in sorted(agents_submaps.keys()):
            poses = []
            for sub in agents_submaps[agent_id]:
                vid = graph_info[(agent_id, sub["submap_start_frame_id"])]
                poses.append(_gtsam_pose3_to_numpy(result.atPose3(vid)))
            optimized[agent_id] = np.stack(poses, axis=0)

        return PGOResult(
            optimized_poses=optimized,
            initial_error=initial_error,
            final_error=final_error,
            num_factors=graph.size(),
        )


# ---------------------------------------------------------------------------
# Lightweight fallback for environments without gtsam
# ---------------------------------------------------------------------------

class NaivePGOSolver:
    """Trivial pass-through solver that returns initial poses unchanged.

    Useful for unit-testing the pipeline wiring without a GTSAM install.
    """

    def optimize(
        self,
        agents_submaps: dict[int, list[dict]],
        loops: list,
    ) -> PGOResult:
        optimized: dict[int, np.ndarray] = {}
        for agent_id in sorted(agents_submaps.keys()):
            poses = []
            for sub in agents_submaps[agent_id]:
                poses.append(
                    np.asarray(sub["submap_c2ws"][0], dtype=np.float32)
                )
            optimized[agent_id] = np.stack(poses, axis=0)
        return PGOResult(optimized_poses=optimized)


def get_solver(backend: str = "gtsam", **kwargs):
    """Factory that returns a PGO solver.

    Falls back to :class:`NaivePGOSolver` when *gtsam* is unavailable.
    """
    if backend == "gtsam" and _HAS_GTSAM:
        return GTSAMSolver(**kwargs)
    return NaivePGOSolver()
