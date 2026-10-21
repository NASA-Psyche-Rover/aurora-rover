#!/usr/bin/env bash
# Live aim-and-tune GUI: rqt_reconfigure on the running overlay node.
# CLI alternative: ros2 param set /aurora_lidar_overlay_node tune_yaw_deg 3.0
source /opt/ros/humble/setup.bash
PERCEPTION="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PERCEPTION/ros2_ws/install/setup.bash"
exec ros2 run rqt_reconfigure rqt_reconfigure
