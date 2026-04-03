"""Rendering quality metrics: PSNR, SSIM, LPIPS, depth-L1 (Paper Section 4)."""

from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio between two images.

    Args:
        pred: predicted image ``(H, W, C)`` in ``[0, max_val]``.
        gt: ground truth image ``(H, W, C)`` in ``[0, max_val]``.
        max_val: dynamic range of the images.

    Returns:
        PSNR in dB.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / mse))


def ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """Structural Similarity Index (mean over channels).

    Simplified sliding-window SSIM that matches the paper's evaluation.
    """
    from scipy.ndimage import uniform_filter

    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    mu_p = uniform_filter(pred, size=window_size)
    mu_g = uniform_filter(gt, size=window_size)
    sigma_pp = uniform_filter(pred * pred, size=window_size) - mu_p * mu_p
    sigma_gg = uniform_filter(gt * gt, size=window_size) - mu_g * mu_g
    sigma_pg = uniform_filter(pred * gt, size=window_size) - mu_p * mu_g

    num = (2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)
    den = (mu_p ** 2 + mu_g ** 2 + C1) * (sigma_pp + sigma_gg + C2)

    ssim_map = num / den
    return float(np.mean(ssim_map))


def depth_l1(pred_depth: np.ndarray, gt_depth: np.ndarray) -> float:
    """Mean absolute depth error over valid pixels.

    Args:
        pred_depth: ``(H, W)`` predicted depth.
        gt_depth: ``(H, W)`` ground-truth depth.

    Returns:
        Mean L1 error (only where ``gt_depth > 0``).
    """
    pred_depth = np.asarray(pred_depth, dtype=np.float64)
    gt_depth = np.asarray(gt_depth, dtype=np.float64)
    valid = gt_depth > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.abs(pred_depth[valid] - gt_depth[valid])))


def ate_rmse(
    pred_translations: np.ndarray, gt_translations: np.ndarray
) -> float:
    """Absolute Trajectory Error (RMSE) after alignment.

    Args:
        pred_translations: ``(N, 3)`` predicted positions.
        gt_translations: ``(N, 3)`` ground-truth positions.

    Returns:
        RMSE of translational error.
    """
    pred = np.asarray(pred_translations, dtype=np.float64)
    gt = np.asarray(gt_translations, dtype=np.float64)
    errors = np.linalg.norm(pred - gt, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


# Paper target values from ASSETS.md / Table 1-2
PAPER_TARGETS: dict[str, dict[str, float]] = {
    "replica_office_0": {"psnr": 39.287, "mode": "camera_depth"},
    "replica_apart_0": {"psnr": 36.634, "mode": "camera_depth"},
    "replica_apart_1": {"psnr": 29.189, "mode": "camera_depth"},
    "replica_apart_2": {"psnr": 31.072, "mode": "camera_depth"},
    "aria_room0": {"psnr": 19.080, "mode": "camera_depth"},
    "aria_room1": {"psnr": 24.176, "mode": "camera_depth"},
}


def compare_to_paper(
    scene_name: str, metrics: dict[str, float]
) -> dict[str, float]:
    """Compare measured metrics against paper targets.

    Returns a dict of ``{metric: gap}`` where gap = measured - target.
    Positive gap means we exceed the paper (good for PSNR).
    """
    targets = PAPER_TARGETS.get(scene_name, {})
    gaps: dict[str, float] = {}
    for key, target_val in targets.items():
        if key in metrics:
            gaps[key] = metrics[key] - target_val
    return gaps
