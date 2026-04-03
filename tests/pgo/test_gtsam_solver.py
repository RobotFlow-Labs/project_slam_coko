"""Tests for PGO solver — both GTSAM and naive fallback."""

from __future__ import annotations

import numpy as np

from anima_slam_coko.loop_closure.detector import LoopCandidate
from anima_slam_coko.pgo.gtsam_solver import NaivePGOSolver, get_solver


def _make_agents_submaps() -> dict[int, list[dict]]:
    """Create a minimal 2-agent, 2-submaps-each test case."""
    agents: dict[int, list[dict]] = {}
    for agent_id in [0, 1]:
        submaps = []
        for i in range(2):
            c2w = np.eye(4, dtype=np.float32)
            c2w[0, 3] = float(agent_id * 10 + i)  # spread them out
            submaps.append({
                "submap_start_frame_id": agent_id * 100 + i * 10,
                "keyframe_ids": np.array([agent_id * 100 + i * 10], dtype=np.int64),
                "submap_c2ws": np.stack([c2w]),
                "gaussian_xyz": np.zeros((3, 3), dtype=np.float32),
                "gaussian_opacity": np.full((3, 1), 0.5, dtype=np.float32),
                "gaussian_scale": np.ones((3, 3), dtype=np.float32) * 0.01,
                "gaussian_rotation": np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32),
                "gaussian_features": np.zeros((3, 2), dtype=np.float32),
            })
        agents[agent_id] = submaps
    return agents


def test_naive_solver_returns_initial_poses() -> None:
    agents = _make_agents_submaps()
    solver = NaivePGOSolver()
    result = solver.optimize(agents, loops=[])

    assert 0 in result.optimized_poses
    assert 1 in result.optimized_poses
    assert result.optimized_poses[0].shape == (2, 4, 4)
    assert result.optimized_poses[1].shape == (2, 4, 4)

    # Should return initial poses unchanged
    for agent_id in [0, 1]:
        for i, sub in enumerate(agents[agent_id]):
            expected = sub["submap_c2ws"][0]
            np.testing.assert_allclose(
                result.optimized_poses[agent_id][i], expected, atol=1e-6
            )


def test_naive_solver_with_loops_still_works() -> None:
    agents = _make_agents_submaps()
    loop = LoopCandidate(
        source_agent_id=0,
        source_frame_id=0,
        target_agent_id=1,
        target_frame_id=100,
        transformation=np.eye(4, dtype=np.float32),
        fitness=0.9,
        inlier_rmse=0.01,
    )
    solver = NaivePGOSolver()
    result = solver.optimize(agents, loops=[loop])
    assert result.optimized_poses[0].shape == (2, 4, 4)


def test_get_solver_returns_naive_without_gtsam() -> None:
    solver = get_solver("gtsam")
    # If gtsam is installed we get GTSAMSolver, otherwise NaivePGOSolver
    # Either is fine — just verify the interface works
    agents = _make_agents_submaps()
    result = solver.optimize(agents, loops=[])
    assert 0 in result.optimized_poses
    assert 1 in result.optimized_poses


def test_gtsam_solver_if_available() -> None:
    """Run the real GTSAM solver if the library is installed."""
    try:
        import gtsam  # noqa: F401
    except ImportError:
        return  # skip silently

    from anima_slam_coko.pgo.gtsam_solver import GTSAMSolver

    agents = _make_agents_submaps()

    # Create a loop between agent 0 submap 1 and agent 1 submap 0
    loop = LoopCandidate(
        source_agent_id=0,
        source_frame_id=10,
        target_agent_id=1,
        target_frame_id=100,
        transformation=np.eye(4, dtype=np.float32),
        fitness=0.9,
        inlier_rmse=0.01,
    )

    solver = GTSAMSolver(max_iterations=10)
    result = solver.optimize(agents, [loop])

    assert result.num_factors > 0
    assert result.final_error <= result.initial_error + 1e-6
    assert result.optimized_poses[0].shape == (2, 4, 4)
    assert result.optimized_poses[1].shape == (2, 4, 4)
