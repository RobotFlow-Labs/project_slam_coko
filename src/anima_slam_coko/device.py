"""Compute backend selection for local MLX development and later CUDA deployment."""

from __future__ import annotations

import os
import platform


def get_backend() -> str:
    requested = os.environ.get("ANIMA_BACKEND", "auto").strip().lower()
    if requested in {"mlx", "cuda", "cpu"}:
        return requested

    if platform.system() == "Darwin":
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError:
            pass
        else:
            return "mlx"

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device():
    backend = get_backend()
    if backend == "mlx":
        import mlx.core as mx

        return mx.default_device()

    import torch

    return torch.device("cuda" if backend == "cuda" else "cpu")
