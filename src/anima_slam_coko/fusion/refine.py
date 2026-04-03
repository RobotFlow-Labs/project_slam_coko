"""Post-merge Gaussian refinement (Paper Section 3.3).

After merging, the server optionally runs a short round of optimization on
the combined map to smooth seams between overlapping submaps.
"""

from __future__ import annotations

from dataclasses import dataclass


from anima_slam_coko.mapping.compaction import CompactionScheduler
from anima_slam_coko.mapping.gaussian_state import GaussianState


@dataclass(slots=True)
class RefineResult:
    pre_count: int
    post_count: int
    iterations: int


def refine_merged_map(
    state: GaussianState,
    *,
    iterations: int = 200,
    prune_ratio: float = 0.05,
    compaction_start: float = 0.6,
    compaction_end: float = 0.9,
) -> RefineResult:
    """Run compaction-style optimization on a merged Gaussian map.

    This mirrors the local-agent compaction schedule but is applied to the
    global map after submap concatenation and duplicate pruning.

    Args:
        state: the merged :class:`GaussianState` (modified in-place).
        iterations: refinement iterations.
        prune_ratio: fraction of Gaussians removed per compaction step.
        compaction_start: fraction of total iterations at which compaction begins.
        compaction_end: fraction of total iterations at which compaction ends.

    Returns:
        :class:`RefineResult` with counts before and after.
    """
    pre_count = state.size
    if pre_count == 0:
        return RefineResult(pre_count=0, post_count=0, iterations=0)

    scheduler = CompactionScheduler(
        start_ratio=compaction_start,
        end_ratio=compaction_end,
        prune_ratio=prune_ratio,
        total_iterations=iterations,
    )

    opacity = state.opacity.copy()
    for it in range(iterations):
        opacity = scheduler.apply(it, opacity, iterations)

    state.opacity = opacity
    state.prune_zero_opacity()

    return RefineResult(
        pre_count=pre_count,
        post_count=state.size,
        iterations=iterations,
    )
