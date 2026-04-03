"""Adapters between ROS2 sensor messages and SLAM-COKO frame dicts."""

from __future__ import annotations

import numpy as np


def image_msg_to_numpy(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to (H, W, 3) uint8 numpy array."""
    if msg.encoding in ("rgb8", "RGB8"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding in ("bgr8", "BGR8"):
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return raw[:, :, ::-1].copy()
    if msg.encoding in ("mono8",):
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return np.stack([gray, gray, gray], axis=-1)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def depth_msg_to_numpy(msg, depth_scale: float = 1.0) -> np.ndarray:
    """Convert sensor_msgs/Image (16UC1 or 32FC1) to (H, W) float32 meters."""
    if msg.encoding in ("16UC1",):
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return raw.astype(np.float32) / depth_scale
    if msg.encoding in ("32FC1",):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


def camera_info_to_intrinsics(msg) -> np.ndarray:
    """Extract 3x3 intrinsics from sensor_msgs/CameraInfo."""
    K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
    return K


def numpy_to_image_msg(rgb: np.ndarray, header=None):
    """Convert (H, W, 3) uint8 to sensor_msgs/Image. Returns a dict if rclpy unavailable."""
    h, w = rgb.shape[:2]
    return {
        "height": h,
        "width": w,
        "encoding": "rgb8",
        "step": w * 3,
        "data": rgb.tobytes(),
        "header": header,
    }


def frame_dict_from_ros(rgb_msg, depth_msg, info_msg, depth_scale: float = 6553.5) -> dict:
    """Build a SLAM-COKO frame dict from ROS2 messages."""
    return {
        "rgb": image_msg_to_numpy(rgb_msg),
        "depth": depth_msg_to_numpy(depth_msg, depth_scale),
        "intrinsics": camera_info_to_intrinsics(info_msg),
    }
