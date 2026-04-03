"""Top-level CLI orchestrator for the SLAM-COKO pipeline (Paper Fig. 2).

Chains the local agent phase (feature extraction, keyframing, tracking,
mapping, compaction) with the server phase (loop detection, registration,
PGO, merge, refinement).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from anima_slam_coko.config import SlamCokoSettings, load_settings
from anima_slam_coko.fusion.merge import merge_submaps
from anima_slam_coko.fusion.refine import refine_merged_map
from anima_slam_coko.io.submap_store import read_submap
from anima_slam_coko.loop_closure.detector import LoopDetector
from anima_slam_coko.pgo.gtsam_solver import get_solver
from anima_slam_coko.registration.coarse import coarse_register
from anima_slam_coko.registration.fine import icp_refine
from anima_slam_coko.registration.rendered_depth import (
    gaussian_xyz_as_cloud,
    submap_to_point_cloud,
)


def _load_submaps(submap_dir: Path) -> dict[int, list[dict]]:
    """Load all submap JSON files from a directory, grouped by agent_id."""
    agents: dict[int, list[dict]] = {}
    for path in sorted(submap_dir.glob("*.json")):
        record = read_submap(path)
        d: dict = {
            "agent_id": record.agent_id,
            "submap_id": record.submap_id,
            "runtime_mode": record.runtime_mode,
            "keyframe_ids": record.keyframe_ids,
            "submap_c2ws": record.submap_c2ws,
            "gaussian_xyz": record.gaussian_xyz,
            "gaussian_opacity": record.gaussian_opacity,
            "gaussian_scale": record.gaussian_scale,
            "gaussian_rotation": record.gaussian_rotation,
            "gaussian_features": record.gaussian_features,
            "descriptor_vector": record.descriptor_vector,
            "rendered_depth": record.rendered_depth,
            "camera_depth": record.camera_depth,
        }
        d["submap_start_frame_id"] = int(record.keyframe_ids[0])
        d["submap_features"] = record.descriptor_vector
        agents.setdefault(d["agent_id"], []).append(d)
    return agents


def _register_loops(
    loops: list,
    agents_submaps: dict[int, list[dict]],
    settings: SlamCokoSettings,
) -> list:
    """Run coarse + fine registration on each loop candidate."""
    intrinsics = np.array([
        [settings.camera.fx, 0.0, settings.camera.cx],
        [0.0, settings.camera.fy, settings.camera.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    registered = []
    for loop in loops:
        src_sub = None
        tgt_sub = None
        for s in agents_submaps.get(loop.source_agent_id, []):
            if s["submap_start_frame_id"] == loop.source_frame_id:
                src_sub = s
                break
        for s in agents_submaps.get(loop.target_agent_id, []):
            if s["submap_start_frame_id"] == loop.target_frame_id:
                tgt_sub = s
                break
        if src_sub is None or tgt_sub is None:
            continue

        mode = settings.runtime.mode
        try:
            src_cloud = submap_to_point_cloud(src_sub, intrinsics, mode=mode)
            tgt_cloud = submap_to_point_cloud(tgt_sub, intrinsics, mode=mode)
        except ValueError:
            src_cloud = gaussian_xyz_as_cloud(src_sub)
            tgt_cloud = gaussian_xyz_as_cloud(tgt_sub)

        if src_cloud.shape[0] < 10 or tgt_cloud.shape[0] < 10:
            continue

        coarse = coarse_register(src_cloud, tgt_cloud)
        fine = icp_refine(src_cloud, tgt_cloud, coarse.transformation)

        loop.transformation = fine.transformation
        loop.fitness = fine.fitness
        loop.inlier_rmse = fine.inlier_rmse
        registered.append(loop)

    return registered


def run_server_fusion(
    agents_submaps: dict[int, list[dict]],
    settings: SlamCokoSettings,
) -> dict:
    """Execute the full server-side fusion pipeline."""
    # Step 1: Loop detection
    detector = LoopDetector(
        embed_size=settings.loop_detection.embed_size,
        feature_dist_threshold=settings.loop_detection.feature_dist_threshold,
        time_threshold=settings.loop_detection.time_threshold,
        max_loops_per_frame=settings.loop_detection.max_loops_per_frame,
        fitness_threshold=settings.loop_detection.fitness_threshold,
        inlier_rmse_threshold=settings.loop_detection.inlier_rmse_threshold,
    )
    intra_loops, inter_loops = detector.detect(agents_submaps)
    all_loops = intra_loops + inter_loops

    # Step 2: Registration
    registered = _register_loops(all_loops, agents_submaps, settings)
    filtered = detector.filter_loops(registered)

    # Step 3: PGO
    solver = get_solver(settings.submap.pgo_backend)
    pgo_result = solver.optimize(agents_submaps, filtered)

    # Step 4: Merge
    merge_result = merge_submaps(agents_submaps, pgo_result.optimized_poses)

    # Step 5: Refine
    refine_result = refine_merged_map(merge_result.state)

    return {
        "intra_loops": len(intra_loops),
        "inter_loops": len(inter_loops),
        "registered_loops": len(registered),
        "filtered_loops": len(filtered),
        "pgo_initial_error": pgo_result.initial_error,
        "pgo_final_error": pgo_result.final_error,
        "pre_prune_gaussians": merge_result.pre_prune_count,
        "post_prune_gaussians": merge_result.post_prune_count,
        "post_refine_gaussians": refine_result.post_count,
        "total_submaps": merge_result.total_submaps,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SLAM-COKO: multi-agent Gaussian SLAM")
    parser.add_argument("--config", type=Path, help="TOML config path")
    parser.add_argument("--submaps", type=Path, required=True, help="Directory of submap JSONs")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON for results")
    args = parser.parse_args(argv)

    configs = [args.config] if args.config else []
    settings = load_settings(*configs) if configs else load_settings()

    agents_submaps = _load_submaps(args.submaps)
    if not agents_submaps:
        print("No submaps found.", file=sys.stderr)
        sys.exit(1)

    result = run_server_fusion(agents_submaps, settings)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
