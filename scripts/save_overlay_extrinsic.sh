#!/usr/bin/env bash
# Save the live-tuned LiDAR->camera extrinsic (tune_* params) into extrinsics.yaml.
source /opt/ros/humble/setup.bash
source "$HOME/psyche-objdetec/ros2_ws/install/setup.bash"
ros2 service call /aurora_lidar_overlay_node/save_extrinsic std_srvs/srv/Trigger
