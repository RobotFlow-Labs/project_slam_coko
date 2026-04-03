"""Loop candidate detection from submap feature vectors (Paper Section 3.3).

Uses FAISS to find inter- and intra-agent loop closure candidates based on
DINOv2 descriptor distance between submap first-keyframe features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class LoopCandidate:
    source_agent_id: int
    source_frame_id: int
    target_agent_id: int
    target_frame_id: int
    feature_distance: float = 0.0
    init_transformation: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )
    transformation: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )
    fitness: float = 0.0
    inlier_rmse: float = float("inf")

    @property
    def is_inter_agent(self) -> bool:
        return self.source_agent_id != self.target_agent_id


def _find_submap_by_frame(
    frame_id: int, submaps: list[dict],
) -> dict | None:
    for s in submaps:
        kf_ids = s.get("keyframe_ids", s.get("submap_start_frame_id"))
        if isinstance(kf_ids, np.ndarray) and frame_id in kf_ids:
            return s
        if isinstance(kf_ids, (int, np.integer)) and int(kf_ids) == frame_id:
            return s
    return None


class LoopDetector:
    """FAISS-backed loop detector matching the reference implementation."""

    def __init__(
        self,
        embed_size: int = 384,
        feature_dist_threshold: float = 0.1,
        time_threshold: int = 0,
        max_loops_per_frame: int = 1,
        fitness_threshold: float = 0.35,
        inlier_rmse_threshold: float = 0.1,
    ) -> None:
        self.embed_size = embed_size
        self.feature_dist_threshold = feature_dist_threshold
        self.time_threshold = time_threshold
        self.max_loops_per_frame = max_loops_per_frame
        self.fitness_threshold = fitness_threshold
        self.inlier_rmse_threshold = inlier_rmse_threshold

        self._features: list[np.ndarray] = []
        self._agent_ids: list[int] = []
        self._frame_ids: list[int] = []

    @property
    def db_size(self) -> int:
        return len(self._frame_ids)

    def _reset(self) -> None:
        self._features.clear()
        self._agent_ids.clear()
        self._frame_ids.clear()

    def _add(
        self, features: np.ndarray, frame_ids: np.ndarray, agent_id: int
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        frame_ids = np.atleast_1d(np.asarray(frame_ids, dtype=np.int64))
        for i in range(features.shape[0]):
            self._features.append(features[i])
            self._agent_ids.append(agent_id)
            fid = int(frame_ids[i]) if i < frame_ids.shape[0] else int(frame_ids[0])
            self._frame_ids.append(fid)

    def _search(
        self,
        query: np.ndarray,
        agent_id: int,
        frame_id: int,
        use_time_threshold: bool = True,
    ) -> list[LoopCandidate]:
        if self.db_size == 0:
            return []

        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        db_matrix = np.stack(self._features, axis=0)

        # L2 distance
        dists = np.linalg.norm(db_matrix - query, axis=1)

        candidates: list[LoopCandidate] = []
        sorted_indices = np.argsort(dists)

        for idx in sorted_indices[: self.max_loops_per_frame]:
            dist = float(dists[idx])
            if dist >= self.feature_dist_threshold:
                continue
            if use_time_threshold and abs(frame_id - self._frame_ids[idx]) < self.time_threshold:
                continue
            candidates.append(
                LoopCandidate(
                    source_agent_id=agent_id,
                    source_frame_id=frame_id,
                    target_agent_id=self._agent_ids[idx],
                    target_frame_id=self._frame_ids[idx],
                    feature_distance=dist,
                )
            )
        return candidates

    def detect_intra_loops(
        self, agents_submaps: dict[int, list[dict]]
    ) -> list[LoopCandidate]:
        intra: list[LoopCandidate] = []
        for agent_id, submaps in agents_submaps.items():
            self._reset()
            for submap in submaps:
                frame_id = int(submap["keyframe_ids"][0])
                features = np.asarray(submap["submap_features"], dtype=np.float32)
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                if self.db_size > 0:
                    hits = self._search(
                        features, agent_id, frame_id, use_time_threshold=True
                    )
                    intra.extend(hits)
                self._add(features, np.array([frame_id]), agent_id)
            self._reset()
        return intra

    def detect_inter_loops(
        self, agents_submaps: dict[int, list[dict]]
    ) -> list[LoopCandidate]:
        self._reset()
        inter: list[LoopCandidate] = []
        for agent_id, submaps in agents_submaps.items():
            if self.db_size > 0:
                for submap in submaps:
                    frame_id = int(submap["keyframe_ids"][0])
                    features = np.asarray(
                        submap["submap_features"], dtype=np.float32
                    )
                    if features.ndim == 1:
                        features = features.reshape(1, -1)
                    hits = self._search(
                        features, agent_id, frame_id, use_time_threshold=False
                    )
                    inter.extend(hits)
            for submap in submaps:
                frame_id = int(submap["keyframe_ids"][0])
                features = np.asarray(
                    submap["submap_features"], dtype=np.float32
                )
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                self._add(features, np.array([frame_id]), agent_id)
        self._reset()
        return inter

    def detect(
        self, agents_submaps: dict[int, list[dict]]
    ) -> tuple[list[LoopCandidate], list[LoopCandidate]]:
        intra = self.detect_intra_loops(agents_submaps)
        inter = self.detect_inter_loops(agents_submaps)

        # Compute initial relative transforms from submap poses
        for loop in intra + inter:
            src_sub = _find_submap_by_frame(
                loop.source_frame_id, agents_submaps[loop.source_agent_id]
            )
            tgt_sub = _find_submap_by_frame(
                loop.target_frame_id, agents_submaps[loop.target_agent_id]
            )
            if src_sub is not None and tgt_sub is not None:
                src_c2w = np.asarray(src_sub["submap_c2ws"][0], dtype=np.float32)
                tgt_c2w = np.asarray(tgt_sub["submap_c2ws"][0], dtype=np.float32)
                loop.init_transformation = np.linalg.inv(tgt_c2w) @ src_c2w

        return intra, inter

    def filter_loops(self, loops: list[LoopCandidate]) -> list[LoopCandidate]:
        return [
            lp
            for lp in loops
            if lp.fitness > self.fitness_threshold
            and lp.inlier_rmse < self.inlier_rmse_threshold
        ]
