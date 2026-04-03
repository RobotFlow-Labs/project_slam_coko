"""Typed checkpoint payloads for compact SLAM-COKO submaps."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RuntimeMode = Literal["rendered_depth", "camera_depth"]

ARRAY_FIELDS = {
    "keyframe_ids",
    "submap_c2ws",
    "gaussian_xyz",
    "gaussian_opacity",
    "gaussian_scale",
    "gaussian_rotation",
    "gaussian_features",
    "descriptor_vector",
    "rendered_depth",
    "camera_depth",
}


def _coerce_array(value: Any, *, dtype: np.dtype) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    return np.asarray(value, dtype=dtype)


class SubmapRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    agent_id: int = Field(ge=0)
    submap_id: int = Field(ge=0)
    runtime_mode: RuntimeMode

    keyframe_ids: np.ndarray
    submap_c2ws: np.ndarray

    gaussian_xyz: np.ndarray
    gaussian_opacity: np.ndarray
    gaussian_scale: np.ndarray
    gaussian_rotation: np.ndarray
    gaussian_features: np.ndarray

    descriptor_vector: np.ndarray
    rendered_depth: np.ndarray | None = None
    camera_depth: np.ndarray | None = None

    @field_validator("keyframe_ids", mode="before")
    @classmethod
    def _validate_keyframe_ids(cls, value: Any) -> np.ndarray:
        return _coerce_array(value, dtype=np.int64)

    @field_validator(
        "submap_c2ws",
        "gaussian_xyz",
        "gaussian_opacity",
        "gaussian_scale",
        "gaussian_rotation",
        "gaussian_features",
        "descriptor_vector",
        "rendered_depth",
        "camera_depth",
        mode="before",
    )
    @classmethod
    def _validate_float_arrays(cls, value: Any) -> np.ndarray | None:
        return _coerce_array(value, dtype=np.float32)

    @model_validator(mode="after")
    def _validate_shapes(self) -> "SubmapRecord":
        if self.keyframe_ids.ndim != 1:
            raise ValueError("keyframe_ids must be a 1D int64 array")

        if self.submap_c2ws.ndim != 3 or self.submap_c2ws.shape[1:] != (4, 4):
            raise ValueError("submap_c2ws must have shape (K, 4, 4)")

        keyframe_count = self.keyframe_ids.shape[0]
        if self.submap_c2ws.shape[0] != keyframe_count:
            raise ValueError("submap_c2ws must match keyframe_ids length")

        gaussian_count = self.gaussian_xyz.shape[0]
        if self.gaussian_xyz.ndim != 2 or self.gaussian_xyz.shape[1] != 3:
            raise ValueError("gaussian_xyz must have shape (N, 3)")

        expected_shapes = {
            "gaussian_opacity": (gaussian_count, 1),
            "gaussian_scale": (gaussian_count, 3),
            "gaussian_rotation": (gaussian_count, 4),
        }
        for field_name, expected_shape in expected_shapes.items():
            field_value = getattr(self, field_name)
            if field_value.shape != expected_shape:
                raise ValueError(f"{field_name} must have shape {expected_shape}")

        if self.gaussian_features.ndim != 2 or self.gaussian_features.shape[0] != gaussian_count:
            raise ValueError("gaussian_features must have shape (N, F)")

        if self.descriptor_vector.ndim != 1:
            raise ValueError("descriptor_vector must be a 1D float32 array")

        if self.runtime_mode == "rendered_depth":
            if self.rendered_depth is None or self.rendered_depth.ndim != 2:
                raise ValueError("rendered_depth mode requires a 2D rendered_depth payload")
            if self.camera_depth is not None:
                raise ValueError("rendered_depth mode cannot include camera_depth payload")
        else:
            if self.camera_depth is None or self.camera_depth.ndim != 2:
                raise ValueError("camera_depth mode requires a 2D camera_depth payload")
            if self.rendered_depth is not None:
                raise ValueError("camera_depth mode cannot include rendered_depth payload")

        return self

    @property
    def gaussian_count(self) -> int:
        return int(self.gaussian_xyz.shape[0])

    def to_serializable(self) -> dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        for field_name in ARRAY_FIELDS.intersection(payload):
            array = payload[field_name]
            if isinstance(array, np.ndarray):
                payload[field_name] = {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "data": array.tolist(),
                }
        return payload

    @classmethod
    def from_serializable(cls, payload: dict[str, Any]) -> "SubmapRecord":
        hydrated = dict(payload)
        for field_name in ARRAY_FIELDS.intersection(hydrated):
            field_value = hydrated[field_name]
            if isinstance(field_value, dict) and "data" in field_value:
                hydrated[field_name] = field_value["data"]
        return cls.model_validate(hydrated)
