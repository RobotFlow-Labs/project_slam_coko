"""Tests for Replica evaluation metrics."""

from __future__ import annotations

import numpy as np

from anima_slam_coko.eval.metrics import compare_to_paper, depth_l1, psnr, ssim
from anima_slam_coko.eval.replica import SceneResult, evaluate_scene


def test_psnr_identical_images() -> None:
    img = np.random.rand(64, 64, 3).astype(np.float32)
    assert psnr(img, img) == float("inf")


def test_psnr_known_value() -> None:
    gt = np.ones((64, 64, 3), dtype=np.float32)
    pred = gt + 0.1
    result = psnr(pred, gt)
    # MSE = 0.01 → PSNR = 10*log10(1/0.01) = 20 dB
    assert abs(result - 20.0) < 0.5


def test_ssim_identical_images() -> None:
    img = np.random.rand(64, 64, 3).astype(np.float32) * 0.5 + 0.25
    result = ssim(img, img)
    assert result > 0.99


def test_ssim_different_images() -> None:
    img1 = np.zeros((64, 64, 3), dtype=np.float32)
    img2 = np.ones((64, 64, 3), dtype=np.float32)
    result = ssim(img1, img2)
    assert result < 0.1


def test_depth_l1_perfect() -> None:
    depth = np.ones((64, 64), dtype=np.float32)
    assert depth_l1(depth, depth) == 0.0


def test_depth_l1_known_error() -> None:
    gt = np.ones((64, 64), dtype=np.float32) * 2.0
    pred = np.ones((64, 64), dtype=np.float32) * 2.5
    assert abs(depth_l1(pred, gt) - 0.5) < 1e-6


def test_depth_l1_ignores_zero_gt() -> None:
    gt = np.zeros((64, 64), dtype=np.float32)
    gt[10:50, 10:50] = 2.0
    pred = np.ones((64, 64), dtype=np.float32) * 2.5
    result = depth_l1(pred, gt)
    assert abs(result - 0.5) < 1e-6


def test_evaluate_scene_produces_result() -> None:
    rng = np.random.RandomState(42)
    pred = [rng.rand(32, 32, 3).astype(np.float32) for _ in range(5)]
    gt = [rng.rand(32, 32, 3).astype(np.float32) for _ in range(5)]
    result = evaluate_scene("office_0", pred, gt)
    assert isinstance(result, SceneResult)
    assert result.scene == "office_0"
    assert result.num_frames == 5
    assert result.psnr_mean > 0


def test_compare_to_paper_known_scene() -> None:
    gaps = compare_to_paper("replica_office_0", {"psnr": 40.0})
    assert "psnr" in gaps
    assert gaps["psnr"] > 0  # 40 > 39.287


def test_compare_to_paper_unknown_scene() -> None:
    gaps = compare_to_paper("nonexistent_scene", {"psnr": 30.0})
    assert gaps == {}
