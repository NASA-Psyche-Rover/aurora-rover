#!/usr/bin/env bash
# Save the live-tuned LiDAR->camera extrinsic (tune_* params) into extrinsics.yaml.
source /opt/ros/humble/setup.bash
PERCEPTION="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PERCEPTION/ros2_ws/install/setup.bash"
ros2 service call /aurora_lidar_overlay_node/save_extrinsic std_srvs/srv/Trigger
