"""Background job orchestration for SLAM runs."""

from __future__ import annotations

import threading
import traceback
import uuid

from anima_slam_coko.api.models import JobInfo, JobStatus, RunRequest


class JobManager:
    """Thread-based job manager for SLAM pipeline runs."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobInfo] = {}
        self._lock = threading.Lock()

    def enqueue(self, request: RunRequest) -> str:
        job_id = str(uuid.uuid4())[:8]
        info = JobInfo(
            job_id=job_id,
            status=JobStatus.queued,
            scene=request.scene,
        )
        with self._lock:
            self._jobs[job_id] = info

        thread = threading.Thread(
            target=self._run_job, args=(job_id, request), daemon=True
        )
        thread.start()
        return job_id

    def get(self, job_id: str) -> JobInfo | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobInfo]:
        with self._lock:
            return list(self._jobs.values())

    def _update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            info = self._jobs[job_id]
            for k, v in kwargs.items():
                setattr(info, k, v)

    def _run_job(self, job_id: str, request: RunRequest) -> None:
        self._update(job_id, status=JobStatus.running, progress=0.0)
        try:
            from anima_slam_coko.data.replica_loader import load_scene
            from anima_slam_coko.config import load_settings
            from anima_slam_coko.train import run_agent, run_server_fusion, _resolve_dino_weights
            from anima_slam_coko.features.dino import DinoFeatureExtractor

            self._update(job_id, progress=0.1)
            settings = load_settings(overrides=request.config_overrides)
            agents = load_scene(request.scene)

            self._update(job_id, progress=0.2)
            dino = DinoFeatureExtractor(
                weights_path=_resolve_dino_weights(), device="cpu"
            )

            agents_submaps: dict[int, list] = {}
            total_agents = len(agents)
            for i, (aid, ds) in enumerate(agents.items()):
                submaps = run_agent(aid, ds, settings, dino)
                agents_submaps[aid] = submaps
                self._update(
                    job_id,
                    progress=0.2 + 0.5 * (i + 1) / total_agents,
                )

            self._update(job_id, progress=0.8)
            result = run_server_fusion(agents_submaps, settings)

            self._update(
                job_id,
                status=JobStatus.completed,
                progress=1.0,
                result=result,
            )
        except Exception as e:
            self._update(
                job_id,
                status=JobStatus.failed,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )
