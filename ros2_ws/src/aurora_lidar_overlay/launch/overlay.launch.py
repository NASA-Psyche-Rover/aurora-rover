# Copyright 2026 AURORA Rover Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Launch the aurora_lidar_overlay node.

Parameters are loaded from config/overlay_params.yaml, overridable from
the command line, e.g.:

    ros2 launch aurora_lidar_overlay overlay.launch.py \\
        use_tf:=true color_by:=intensity skip:=2

    ros2 launch aurora_lidar_overlay overlay.launch.py \\
        use_sim_time:=true \\
        extrinsic_yaml:=/tmp/synthetic_bag_calibration/extrinsics.yaml \\
        intrinsic_yaml:=/tmp/synthetic_bag_calibration/intrinsics.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("aurora_lidar_overlay")
    default_params = os.path.join(pkg_share, "config", "overlay_params.yaml")
    default_extrinsic = os.path.join(pkg_share, "config", "calibration", "extrinsics.yaml")
    default_intrinsic = os.path.join(pkg_share, "config", "calibration", "intrinsics.yaml")

    args = [
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("image_topic", default_value="/oak/rgb/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/oak/rgb/camera_info"),
        DeclareLaunchArgument("lidar_topic", default_value="/livox/lidar"),
        DeclareLaunchArgument("output_topic", default_value="/aurora/lidar_overlay/image"),
        DeclareLaunchArgument("extrinsic_yaml", default_value=default_extrinsic),
        DeclareLaunchArgument("intrinsic_yaml", default_value=default_intrinsic),
        DeclareLaunchArgument("invert_extrinsic", default_value="false"),
        DeclareLaunchArgument("use_tf", default_value="false"),
        DeclareLaunchArgument("lidar_frame", default_value="livox_frame"),
        DeclareLaunchArgument("camera_frame", default_value="oak_camera"),
        DeclareLaunchArgument("color_by", default_value="depth"),
        DeclareLaunchArgument("skip", default_value="1"),
    ]

    node = Node(
        package="aurora_lidar_overlay",
        executable="overlay_node",
        name="aurora_lidar_overlay_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "image_topic": LaunchConfiguration("image_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "lidar_topic": LaunchConfiguration("lidar_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
                "extrinsic_yaml": LaunchConfiguration("extrinsic_yaml"),
                "intrinsic_yaml": LaunchConfiguration("intrinsic_yaml"),
                "invert_extrinsic": LaunchConfiguration("invert_extrinsic"),
                "use_tf": LaunchConfiguration("use_tf"),
                "lidar_frame": LaunchConfiguration("lidar_frame"),
                "camera_frame": LaunchConfiguration("camera_frame"),
                "color_by": LaunchConfiguration("color_by"),
                "skip": LaunchConfiguration("skip"),
            },
        ],
    )

    return LaunchDescription(args + [node])
