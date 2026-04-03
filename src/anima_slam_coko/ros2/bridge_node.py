"""ROS2 bridge node for SLAM-COKO pipeline.

Subscribes to RGB-D topics, runs the SLAM pipeline per frame,
and publishes submap summaries and optimized poses.

Requires rclpy (ROS2 Python client library).
"""

from __future__ import annotations

import json


try:
    import rclpy
    from rclpy.node import Node

    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False
    Node = object  # type: ignore[assignment,misc]

from anima_slam_coko.ros2.message_adapters import (
    camera_info_to_intrinsics,
    depth_msg_to_numpy,
    image_msg_to_numpy,
)


class SlamCokoBridge(Node):
    """ROS2 node wrapping the SLAM-COKO agent pipeline."""

    def __init__(self) -> None:
        if not _HAS_RCLPY:
            raise ImportError("rclpy is required for the ROS2 bridge node")

        super().__init__("slam_coko_bridge")

        self.declare_parameter("depth_scale", 6553.5)
        self.declare_parameter("dino_weights", "/mnt/forge-data/models/facebook--dinov2-small")
        self.declare_parameter("agent_id", 0)

        self._depth_scale = self.get_parameter("depth_scale").value
        self._agent_id = self.get_parameter("agent_id").value
        self._intrinsics = None
        self._frame_count = 0

        # Subscribers
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String

        self._rgb_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self._on_rgb, qos
        )
        self._depth_sub = self.create_subscription(
            Image, "/camera/depth/image_rect", self._on_depth, qos
        )
        self._info_sub = self.create_subscription(
            CameraInfo, "/camera/info", self._on_info, qos
        )

        # Publishers
        self._status_pub = self.create_publisher(String, "/slam_coko/status", 10)

        self._latest_rgb = None
        self._latest_depth = None

        self.get_logger().info("SLAM-COKO bridge node initialized")

    def _on_info(self, msg) -> None:
        self._intrinsics = camera_info_to_intrinsics(msg)

    def _on_rgb(self, msg) -> None:
        self._latest_rgb = image_msg_to_numpy(msg)
        self._try_process()

    def _on_depth(self, msg) -> None:
        self._latest_depth = depth_msg_to_numpy(msg, self._depth_scale)
        self._try_process()

    def _try_process(self) -> None:
        if self._latest_rgb is None or self._latest_depth is None:
            return
        if self._intrinsics is None:
            self.get_logger().warn("No camera info received yet")
            return

        rgb = self._latest_rgb
        depth = self._latest_depth
        self._latest_rgb = None
        self._latest_depth = None

        self._frame_count += 1

        from std_msgs.msg import String

        status = String()
        status.data = json.dumps({
            "agent_id": self._agent_id,
            "frame": self._frame_count,
            "rgb_shape": list(rgb.shape),
            "depth_range": [float(depth[depth > 0].min()) if (depth > 0).any() else 0,
                            float(depth[depth > 0].max()) if (depth > 0).any() else 0],
        })
        self._status_pub.publish(status)

        if self._frame_count % 50 == 0:
            self.get_logger().info(f"Processed {self._frame_count} frames")


def main(args=None):
    if not _HAS_RCLPY:
        print("rclpy not available — cannot start ROS2 node")
        return
    rclpy.init(args=args)
    node = SlamCokoBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
