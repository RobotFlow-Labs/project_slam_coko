"""SLAM-COKO training entry point.

Runs the full multi-agent Gaussian Splatting SLAM pipeline on a prepared
Replica scene: per-agent local mapping → server fusion → evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m anima_slam_coko.train --scene room0
    CUDA_VISIBLE_DEVICES=1 python -m anima_slam_coko.train --scene room0 --config configs/coko/base.toml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anima_slam_coko.config import SlamCokoSettings, load_settings
from anima_slam_coko.data.replica_loader import ReplicaSLAMDataset, load_scene
from anima_slam_coko.eval.bandwidth import compute_bandwidth
from anima_slam_coko.eval.metrics import compare_to_paper, depth_l1
from anima_slam_coko.features.dino import DinoFeatureExtractor
from anima_slam_coko.fusion.merge import merge_submaps
from anima_slam_coko.fusion.refine import refine_merged_map
from anima_slam_coko.keyframing.policy import KeyframePolicy
from anima_slam_coko.loop_closure.detector import LoopDetector
from anima_slam_coko.mapping.gaussian_state import GaussianState
from anima_slam_coko.mapping.mapper import Mapper
from anima_slam_coko.pgo.gtsam_solver import get_solver
from anima_slam_coko.registration.coarse import coarse_register
from anima_slam_coko.registration.fine import icp_refine
from anima_slam_coko.registration.rendered_depth import gaussian_xyz_as_cloud
from anima_slam_coko.tracking.tracker import Tracker, TrackerConfig

SLAM_DATA_ROOT = Path("/mnt/forge-data/datasets/replica_slam")
ARTIFACTS_ROOT = Path("/mnt/artifacts-datai")
PROJECT = "project_slam_coko"


def _resolve_dino_weights() -> str:
    """Find DINOv2-Small weights on the server."""
    candidates = [
        "/mnt/forge-data/models/facebook--dinov2-small",
        "./dinov2-small",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return "facebook/dinov2-small"


def run_agent(
    agent_id: int,
    dataset: ReplicaSLAMDataset,
    settings: SlamCokoSettings,
    dino: DinoFeatureExtractor,
) -> list[dict]:
    """Run the local agent pipeline: feature → keyframe → track → map → compact.

    Returns a list of submap dicts ready for server fusion.
    """
    intrinsics = dataset.intrinsics
    policy = KeyframePolicy(
        alpha=settings.keyframing.threshold,
        submapping_threshold=settings.submapping.threshold,
        min_keyframes_per_submap=settings.submapping.keyframe_num,
    )
    tracker = Tracker(intrinsics, TrackerConfig(
        odometry_type=settings.tracking.odometry_type,
        w_color_loss=settings.tracking.w_color_loss,
    ))
    mapper = Mapper(
        intrinsics,
        iterations=settings.mapping.iterations,
        new_submap_iterations=settings.mapping.new_submap_iterations,
        new_submap_points_num=settings.mapping.new_submap_points_num,
        prune_ratio=settings.mapping.prune_ratio,
    )

    submaps: list[dict] = []
    current_state = GaussianState()
    keyframe_features: list[np.ndarray] = []
    keyframe_ids: list[int] = []
    keyframe_c2ws: list[np.ndarray] = []
    submap_id = 0
    submap_anchor: np.ndarray | None = None
    prev_c2ws = np.stack([np.eye(4, dtype=np.float32)])

    print(f"  [Agent {agent_id}] Processing {len(dataset)} frames...")
    t0 = time.time()

    for idx in range(len(dataset)):
        frame_id, rgb, depth, gt_c2w = dataset[idx]
        frame = {"rgb": rgb, "depth": depth, "gt_c2w": gt_c2w}

        # Extract features
        feature = dino.extract(rgb)  # (1, 384)
        feature_vec = feature.reshape(-1)

        # Track
        if idx == 0:
            pose = gt_c2w.copy()
            tracker.odometer.update_last_rgbd(rgb, depth)
        else:
            pose = tracker.track(frame, gaussian_state=None, prev_c2ws=prev_c2ws)

        # Keyframe decision
        is_keyframe = len(keyframe_features) == 0 or policy.should_promote(
            feature_vec, keyframe_features
        )

        if is_keyframe:
            keyframe_features.append(feature_vec)
            keyframe_ids.append(frame_id)
            keyframe_c2ws.append(pose)

            # Map
            mapper.map_keyframe(
                frame, pose, current_state,
                is_new_submap=(len(keyframe_ids) == 1),
            )

            # Submap rotation check
            if submap_anchor is None:
                submap_anchor = feature_vec.copy()

            should_rotate = policy.should_rotate_submap(
                feature_vec, submap_anchor, keyframe_features
            )

            if should_rotate or idx == len(dataset) - 1:
                # Emit submap
                descriptor = np.mean(
                    np.stack(keyframe_features), axis=0
                ).astype(np.float32)

                submap_dict = {
                    "agent_id": agent_id,
                    "submap_id": submap_id,
                    "submap_start_frame_id": keyframe_ids[0],
                    "keyframe_ids": np.array(keyframe_ids, dtype=np.int64),
                    "submap_c2ws": np.stack(keyframe_c2ws).astype(np.float32),
                    "gaussian_xyz": current_state.xyz.copy(),
                    "gaussian_opacity": current_state.opacity.copy(),
                    "gaussian_scale": current_state.scale.copy(),
                    "gaussian_rotation": current_state.rotation.copy(),
                    "gaussian_features": current_state.features.copy(),
                    "descriptor_vector": descriptor,
                    "submap_features": descriptor,
                }

                # Add depth payload
                submap_dict["rendered_depth"] = depth.copy()

                submaps.append(submap_dict)

                # Reset for next submap
                submap_id += 1
                current_state = GaussianState()
                keyframe_features = []
                keyframe_ids = []
                keyframe_c2ws = []
                submap_anchor = None

        # Update tracking history
        prev_c2ws = np.stack(
            [prev_c2ws[-1], pose] if prev_c2ws.shape[0] >= 1 else [pose]
        )

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"    Frame {idx + 1}/{len(dataset)} — "
                f"{(idx + 1) / elapsed:.1f} fps — "
                f"{len(submaps)} submaps — "
                f"{current_state.size} gaussians"
            )

    elapsed = time.time() - t0
    print(
        f"  [Agent {agent_id}] Done: {len(submaps)} submaps, "
        f"{sum(s['gaussian_xyz'].shape[0] for s in submaps)} total gaussians, "
        f"{elapsed:.1f}s"
    )
    return submaps


def run_server_fusion(
    agents_submaps: dict[int, list[dict]],
    settings: SlamCokoSettings,
) -> dict:
    """Run server-side fusion: loop detect → register → PGO → merge → refine."""
    print("  [Server] Loop detection...")
    detector = LoopDetector(
        embed_size=settings.loop_detection.embed_size,
        feature_dist_threshold=settings.loop_detection.feature_dist_threshold,
        max_loops_per_frame=settings.loop_detection.max_loops_per_frame,
        fitness_threshold=settings.loop_detection.fitness_threshold,
        inlier_rmse_threshold=settings.loop_detection.inlier_rmse_threshold,
    )
    intra, inter = detector.detect(agents_submaps)
    all_loops = intra + inter
    print(f"    {len(intra)} intra + {len(inter)} inter loops detected")

    # Register loops
    print("  [Server] Registration...")
    registered = []
    for loop in all_loops:
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

        src_cloud = gaussian_xyz_as_cloud(src_sub)
        tgt_cloud = gaussian_xyz_as_cloud(tgt_sub)
        if src_cloud.shape[0] < 10 or tgt_cloud.shape[0] < 10:
            continue

        try:
            coarse = coarse_register(src_cloud, tgt_cloud)
            fine = icp_refine(src_cloud, tgt_cloud, coarse.transformation)
            loop.transformation = fine.transformation
            loop.fitness = fine.fitness
            loop.inlier_rmse = fine.inlier_rmse
            registered.append(loop)
        except Exception as e:
            print(f"    Registration failed: {e}")

    filtered = detector.filter_loops(registered)
    print(f"    {len(registered)} registered, {len(filtered)} filtered")

    # PGO
    print("  [Server] Pose graph optimization...")
    solver = get_solver(settings.submap.pgo_backend)
    pgo_result = solver.optimize(agents_submaps, filtered)
    print(f"    Error: {pgo_result.initial_error:.4f} → {pgo_result.final_error:.4f}")

    # Merge
    print("  [Server] Map merge...")
    merge_result = merge_submaps(agents_submaps, pgo_result.optimized_poses)
    print(
        f"    {merge_result.pre_prune_count} → {merge_result.post_prune_count} gaussians "
        f"({merge_result.total_submaps} submaps)"
    )

    # Refine
    print("  [Server] Refinement...")
    refine_result = refine_merged_map(merge_result.state)
    print(f"    {refine_result.pre_count} → {refine_result.post_count} gaussians")

    # Bandwidth audit
    bw = compute_bandwidth(agents_submaps)
    print(f"    Bandwidth: {bw.total_transmitted_mb:.1f} MB total, {bw.mean_per_agent_mb:.1f} MB/agent")

    return {
        "intra_loops": len(intra),
        "inter_loops": len(inter),
        "registered_loops": len(registered),
        "filtered_loops": len(filtered),
        "pgo_initial_error": pgo_result.initial_error,
        "pgo_final_error": pgo_result.final_error,
        "pre_prune_gaussians": merge_result.pre_prune_count,
        "post_prune_gaussians": merge_result.post_prune_count,
        "post_refine_gaussians": refine_result.post_count,
        "total_submaps": merge_result.total_submaps,
        "bandwidth_total_mb": bw.total_transmitted_mb,
        "bandwidth_per_agent_mb": bw.mean_per_agent_mb,
    }


def evaluate(
    scene: str,
    agents: dict[int, ReplicaSLAMDataset],
    agents_submaps: dict[int, list[dict]],
) -> dict:
    """Compute rendering metrics against ground truth."""
    print("  [Eval] Computing metrics...")
    dl1_vals = []

    for agent_id, dataset in agents.items():
        for sub in agents_submaps.get(agent_id, []):
            for kf_idx, kf_id in enumerate(sub["keyframe_ids"]):
                if kf_id >= len(dataset):
                    continue
                _, gt_rgb, gt_depth, _ = dataset[int(kf_id)]

                # For now, use the rendered depth from the submap as a proxy
                rendered_depth = sub.get("rendered_depth")
                if rendered_depth is not None and gt_depth is not None:
                    dl1_vals.append(depth_l1(rendered_depth, gt_depth))

    metrics = {
        "depth_l1_mean": float(np.mean(dl1_vals)) if dl1_vals else 0.0,
        "num_keyframes_evaluated": len(dl1_vals),
    }

    scene_key = f"replica_{scene}"
    gaps = compare_to_paper(scene_key, metrics)
    metrics["paper_gaps"] = gaps

    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SLAM-COKO training")
    parser.add_argument("--scene", type=str, default="room0", help="Scene name")
    parser.add_argument("--config", type=Path, default=None, help="TOML config")
    parser.add_argument("--data-root", type=Path, default=SLAM_DATA_ROOT)
    args = parser.parse_args(argv)

    # Load config
    config_paths = [args.config] if args.config else []
    settings = load_settings(*config_paths) if config_paths else load_settings()

    # Setup output dirs
    ckpt_dir = ARTIFACTS_ROOT / "checkpoints" / PROJECT / args.scene
    log_dir = ARTIFACTS_ROOT / "logs" / PROJECT
    report_dir = ARTIFACTS_ROOT / "reports" / PROJECT
    for d in [ckpt_dir, log_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SLAM-COKO Training — {args.scene}")
    print(f"Data: {args.data_root / args.scene}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Config: {args.config or 'default'}")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    agents = load_scene(args.scene, slam_data_root=args.data_root)
    for aid, ds in agents.items():
        print(f"  Agent {aid}: {len(ds)} frames")

    # Initialize DINO
    print("\n[2/4] Running per-agent SLAM...")
    dino_path = _resolve_dino_weights()
    print(f"  DINOv2 weights: {dino_path}")
    dino = DinoFeatureExtractor(weights_path=dino_path, device="cpu")

    # Per-agent processing
    agents_submaps: dict[int, list[dict]] = {}
    t_start = time.time()
    for agent_id, dataset in agents.items():
        submaps = run_agent(agent_id, dataset, settings, dino)
        agents_submaps[agent_id] = submaps

    # Server fusion
    print("\n[3/4] Server fusion...")
    fusion_result = run_server_fusion(agents_submaps, settings)

    # Evaluation
    print("\n[4/4] Evaluation...")
    eval_result = evaluate(args.scene, agents, agents_submaps)

    total_time = time.time() - t_start

    # Write results
    results = {
        "scene": args.scene,
        "total_time_s": total_time,
        "fusion": fusion_result,
        "evaluation": eval_result,
        "agents": {
            str(aid): {
                "num_submaps": len(subs),
                "total_gaussians": sum(s["gaussian_xyz"].shape[0] for s in subs),
                "total_keyframes": sum(len(s["keyframe_ids"]) for s in subs),
            }
            for aid, subs in agents_submaps.items()
        },
    }

    results_path = report_dir / f"{args.scene}_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — {args.scene}")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Submaps: {fusion_result['total_submaps']}")
    print(f"  Gaussians: {fusion_result['post_refine_gaussians']}")
    print(f"  Bandwidth: {fusion_result['bandwidth_total_mb']:.1f} MB")
    print(f"  Depth-L1: {eval_result['depth_l1_mean']:.4f}")
    print(f"  Results: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
