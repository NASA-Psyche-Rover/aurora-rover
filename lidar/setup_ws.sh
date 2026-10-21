#!/usr/bin/env bash
# Build ~/ros2_ws with pinned upstream clones + our configs/patches. Run once.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; WS="$HOME/ros2_ws"; mkdir -p "$WS/src" "$WS/PCD"; cd "$WS/src"
[ -d livox_ros_driver2 ] || git clone https://github.com/Livox-SDK/livox_ros_driver2.git
git -C livox_ros_driver2 checkout 4a1def9
git -C livox_ros_driver2 apply "$HERE/patches/livox_ros_driver2_lds.patch" || echo "(patch already applied?)"
cp "$HERE/config/MID360_config.json" livox_ros_driver2/config/
[ -d FAST_LIO ] || git clone https://github.com/hku-mars/FAST_LIO --branch ROS2 --recursive
git -C FAST_LIO checkout a4743b0
cp "$HERE/config/mid360.yaml" FAST_LIO/config/; cp "$HERE/config/loam_livox_ros2.rviz" FAST_LIO/rviz_cfg/
cp "$HERE"/slam/{start_slam.sh,fastdds_localhost_unicast.xml} "$WS/"; chmod +x "$WS/start_slam.sh"
# livox_ros_driver2 ROS2 build: copy its package_ROS2.xml / launch_ROS2 per upstream README if colcon complains
cd "$WS" && source /opt/ros/humble/setup.bash && colcon build --symlink-install
