"""ROS2 launch file for SLAM-COKO bridge node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("depth_scale", default_value="6553.5"),
        DeclareLaunchArgument("agent_id", default_value="0"),
        DeclareLaunchArgument(
            "dino_weights",
            default_value="/mnt/forge-data/models/facebook--dinov2-small",
        ),
        Node(
            package="anima_slam_coko",
            executable="slam_coko_bridge",
            name="slam_coko_bridge",
            parameters=[{
                "depth_scale": LaunchConfiguration("depth_scale"),
                "agent_id": LaunchConfiguration("agent_id"),
                "dino_weights": LaunchConfiguration("dino_weights"),
            }],
            remappings=[
                ("/camera/color/image_raw", "/camera/color/image_raw"),
                ("/camera/depth/image_rect", "/camera/depth/image_rect"),
                ("/camera/info", "/camera/camera_info"),
            ],
        ),
    ])
