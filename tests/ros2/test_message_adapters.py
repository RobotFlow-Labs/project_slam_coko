"""Tests for ROS2 message adapters (no rclpy dependency)."""

from __future__ import annotations

import numpy as np
import pytest

from anima_slam_coko.ros2.message_adapters import (
    depth_msg_to_numpy,
    image_msg_to_numpy,
    numpy_to_image_msg,
)


class FakeImageMsg:
    def __init__(self, data: bytes, height: int, width: int, encoding: str):
        self.data = data
        self.height = height
        self.width = width
        self.encoding = encoding


class FakeCameraInfo:
    def __init__(self, fx=600.0, fy=600.0, cx=320.0, cy=240.0):
        self.k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]


def test_rgb8_to_numpy() -> None:
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    msg = FakeImageMsg(rgb.tobytes(), 480, 640, "rgb8")
    result = image_msg_to_numpy(msg)
    assert result.shape == (480, 640, 3)
    np.testing.assert_array_equal(result, rgb)


def test_bgr8_to_numpy() -> None:
    bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    msg = FakeImageMsg(bgr.tobytes(), 480, 640, "bgr8")
    result = image_msg_to_numpy(msg)
    assert result.shape == (480, 640, 3)
    np.testing.assert_array_equal(result[:, :, 0], bgr[:, :, 2])


def test_depth_16uc1() -> None:
    depth_u16 = np.ones((480, 640), dtype=np.uint16) * 3000
    msg = FakeImageMsg(depth_u16.tobytes(), 480, 640, "16UC1")
    result = depth_msg_to_numpy(msg, depth_scale=6553.5)
    assert result.dtype == np.float32
    assert abs(result.mean() - 3000 / 6553.5) < 0.01


def test_depth_32fc1() -> None:
    depth_f32 = np.ones((480, 640), dtype=np.float32) * 2.5
    msg = FakeImageMsg(depth_f32.tobytes(), 480, 640, "32FC1")
    result = depth_msg_to_numpy(msg)
    assert result.dtype == np.float32
    assert abs(result.mean() - 2.5) < 0.01


def test_unsupported_encoding_raises() -> None:
    msg = FakeImageMsg(b"", 10, 10, "yuv422")
    with pytest.raises(ValueError, match="Unsupported"):
        image_msg_to_numpy(msg)


def test_numpy_to_image_msg_roundtrip() -> None:
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    msg_dict = numpy_to_image_msg(rgb)
    assert msg_dict["height"] == 64
    assert msg_dict["width"] == 64
    assert msg_dict["encoding"] == "rgb8"
    assert len(msg_dict["data"]) == 64 * 64 * 3


def test_camera_info_to_intrinsics() -> None:
    from anima_slam_coko.ros2.message_adapters import camera_info_to_intrinsics

    info = FakeCameraInfo(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    K = camera_info_to_intrinsics(info)
    assert K.shape == (3, 3)
    assert K[0, 0] == 600.0
    assert K[1, 1] == 600.0
    assert K[0, 2] == 320.0
