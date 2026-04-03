"""DINOv2-Small feature extraction wrapper used by the paper pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from anima_slam_coko.device import get_backend


class DinoFeatureExtractor:
    """Thin wrapper around the paper's DINOv2-Small feature extractor.

    The Mac prebuild path keeps this import-light. Runtime dependencies are loaded only
    when `extract()` is called on an environment that actually has Torch and Transformers.
    """

    def __init__(
        self,
        weights_path: str | Path = "./dinov2-small",
        *,
        device: str = "auto",
        embedding_dim: int = 384,
    ) -> None:
        self.weights_path = str(weights_path)
        self.embedding_dim = embedding_dim
        self.device = self._resolve_device(device)
        self._processor: Any | None = None
        self._model: Any | None = None

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        backend = get_backend()
        return "cuda" if backend == "cuda" else "cpu"

    def _load_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINO extraction requires the paper extras. Install with "
                "`uv sync --extra paper` and, on Linux GPU hosts, add the CUDA bootstrap."
            ) from exc

        self._processor = AutoImageProcessor.from_pretrained(self.weights_path)
        self._model = AutoModel.from_pretrained(self.weights_path).to(self.device)
        self._model.eval()
        self._torch = torch

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb input must have shape (H, W, 3)")

        self._load_model()

        from PIL import Image

        image = Image.fromarray(rgb.astype(np.uint8, copy=False))
        with self._torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            outputs = self._model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            features = self._torch.nn.functional.normalize(features, p=2, dim=1)

        vector = features.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
        if vector.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected a batched {self.embedding_dim}-D embedding, got {vector.shape[1]}"
            )
        return vector
