"""Runtime telemetry counters for SLAM-COKO (Paper Tables 3-5)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AgentTelemetry:
    agent_id: int
    frames_processed: int = 0
    keyframes_promoted: int = 0
    submaps_emitted: int = 0
    total_gaussians: int = 0
    processing_time_s: float = 0.0

    def summary(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "frames": self.frames_processed,
            "keyframes": self.keyframes_promoted,
            "submaps": self.submaps_emitted,
            "gaussians": self.total_gaussians,
            "fps": self.frames_processed / max(self.processing_time_s, 1e-6),
        }


@dataclass
class ServerTelemetry:
    intra_loops: int = 0
    inter_loops: int = 0
    registered_loops: int = 0
    filtered_loops: int = 0
    pgo_initial_error: float = 0.0
    pgo_final_error: float = 0.0
    pre_merge_gaussians: int = 0
    post_merge_gaussians: int = 0
    post_refine_gaussians: int = 0
    merge_time_s: float = 0.0

    def summary(self) -> dict:
        return {
            "loops": {
                "intra": self.intra_loops,
                "inter": self.inter_loops,
                "registered": self.registered_loops,
                "filtered": self.filtered_loops,
            },
            "pgo": {
                "initial_error": self.pgo_initial_error,
                "final_error": self.pgo_final_error,
            },
            "gaussians": {
                "pre_merge": self.pre_merge_gaussians,
                "post_merge": self.post_merge_gaussians,
                "post_refine": self.post_refine_gaussians,
            },
        }


@dataclass
class PipelineTelemetry:
    scene: str = ""
    agents: dict[int, AgentTelemetry] = field(default_factory=dict)
    server: ServerTelemetry = field(default_factory=ServerTelemetry)
    total_time_s: float = 0.0
    start_time: float = field(default_factory=time.time)

    def add_agent(self, agent_id: int) -> AgentTelemetry:
        t = AgentTelemetry(agent_id=agent_id)
        self.agents[agent_id] = t
        return t

    def finalize(self) -> None:
        self.total_time_s = time.time() - self.start_time

    def summary(self) -> dict:
        return {
            "scene": self.scene,
            "total_time_s": self.total_time_s,
            "agents": {str(k): v.summary() for k, v in self.agents.items()},
            "server": self.server.summary(),
        }
