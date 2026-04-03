"""Communication budget auditing (Paper Tables 3-5).

Measures keyframe counts and total transmitted data per agent to verify
the paper's 85-95% communication reduction claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AgentBudget:
    agent_id: int
    submap_count: int
    keyframe_count: int
    gaussian_count: int
    transmitted_bytes: int

    @property
    def transmitted_mb(self) -> float:
        return self.transmitted_bytes / (1024 * 1024)


@dataclass(slots=True)
class BandwidthReport:
    agents: list[AgentBudget]
    total_transmitted_mb: float
    mean_per_agent_mb: float


def estimate_submap_bytes(submap: dict) -> int:
    """Estimate the wire size of a single submap payload.

    Accounts for: Gaussian params (xyz, opacity, scale, rotation, features),
    keyframe poses, descriptor vector, and optional depth payload.
    """
    n_gauss = np.asarray(submap["gaussian_xyz"]).reshape(-1, 3).shape[0]
    n_kf = np.asarray(submap["keyframe_ids"]).reshape(-1).shape[0]
    desc_dim = np.asarray(submap.get("descriptor_vector", np.zeros(1))).reshape(-1).shape[0]

    # Gaussian params: xyz(3) + opacity(1) + scale(3) + rotation(4) + features(F)
    feat_dim = 2
    feats = submap.get("gaussian_features")
    if feats is not None:
        feats = np.asarray(feats)
        if feats.ndim == 2:
            feat_dim = feats.shape[1]
    gauss_bytes = n_gauss * (3 + 1 + 3 + 4 + feat_dim) * 4  # float32

    # Keyframe poses: K * 4*4 * float32
    pose_bytes = n_kf * 16 * 4

    # Descriptor vector
    desc_bytes = desc_dim * 4

    # Depth payload (if camera_depth mode)
    depth_bytes = 0
    for key in ("camera_depth", "rendered_depth"):
        depth = submap.get(key)
        if depth is not None:
            depth = np.asarray(depth)
            depth_bytes = depth.size * 4  # float32

    return gauss_bytes + pose_bytes + desc_bytes + depth_bytes


def compute_bandwidth(
    agents_submaps: dict[int, list[dict]],
) -> BandwidthReport:
    """Compute per-agent communication budget.

    Args:
        agents_submaps: ``{agent_id: [submap_dict, ...]}``.

    Returns:
        :class:`BandwidthReport` with per-agent and total stats.
    """
    budgets: list[AgentBudget] = []

    for agent_id in sorted(agents_submaps.keys()):
        submaps = agents_submaps[agent_id]
        total_kf = 0
        total_gauss = 0
        total_bytes = 0

        for sub in submaps:
            kf_ids = np.asarray(sub["keyframe_ids"]).reshape(-1)
            gauss_xyz = np.asarray(sub["gaussian_xyz"]).reshape(-1, 3)
            total_kf += kf_ids.shape[0]
            total_gauss += gauss_xyz.shape[0]
            total_bytes += estimate_submap_bytes(sub)

        budgets.append(AgentBudget(
            agent_id=agent_id,
            submap_count=len(submaps),
            keyframe_count=total_kf,
            gaussian_count=total_gauss,
            transmitted_bytes=total_bytes,
        ))

    total_mb = sum(b.transmitted_mb for b in budgets)
    mean_mb = total_mb / max(len(budgets), 1)

    return BandwidthReport(
        agents=budgets,
        total_transmitted_mb=total_mb,
        mean_per_agent_mb=mean_mb,
    )


# Paper budget targets (MB per agent, camera-depth mode)
PAPER_BUDGET_TARGETS: dict[str, dict[int, float]] = {
    "aria_room0": {0: 55.0, 1: 79.0, 2: 65.0},
    "aria_room1": {0: 66.0, 1: 71.0, 2: 50.0},
}


def compare_budget_to_paper(
    scene_name: str, report: BandwidthReport
) -> dict[int, float]:
    """Compare per-agent transmitted MB against paper targets.

    Returns ``{agent_id: gap_mb}`` where negative = under budget (good).
    """
    targets = PAPER_BUDGET_TARGETS.get(scene_name, {})
    gaps: dict[int, float] = {}
    for budget in report.agents:
        if budget.agent_id in targets:
            gaps[budget.agent_id] = budget.transmitted_mb - targets[budget.agent_id]
    return gaps
