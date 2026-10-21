#!/bin/bash
# One-command Livox MID-360 + FAST-LIO SLAM bring-up (Jetson).
# Kills any previous run first, so every launch starts from an empty map.
#
#   ~/ros2_ws/start_slam.sh
#
# Ctrl+C stops everything. To save the map while it is running, in another shell:
#   source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
#   export FASTRTPS_DEFAULT_PROFILES_FILE=~/ros2_ws/fastdds_localhost_unicast.xml
#   ros2 service call /map_save std_srvs/srv/Trigger      # -> ~/ros2_ws/PCD/room_map.pcd

export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/ros2_ws/fastdds_localhost_unicast.xml
unset ROS_LOCALHOST_ONLY
export DISPLAY=${DISPLAY:-:1}

source /opt/ros/humble/setup.bash
source $HOME/ros2_ws/install/setup.bash

echo "[start_slam] clearing any previous run ..."
pkill -f "ros2 launch fast_lio"                 2>/dev/null
pkill -f "ros2 launch livox_ros_driver2"        2>/dev/null
pkill -f "install/fast_lio/lib/fast_lio"        2>/dev/null   # fastlio_mapping (comm name is truncated to 15 chars, so -x fails)
pkill -f "install/livox_ros_driver2/lib"        2>/dev/null   # livox_ros_driver2_node
pkill -x rviz2                                  2>/dev/null
sleep 3
ros2 daemon stop >/dev/null 2>&1

echo "[start_slam] starting Livox MID-360 driver ..."
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
LIVOX_PID=$!
sleep 4

echo "[start_slam] starting FAST-LIO + RViz (fresh map) ..."
ros2 launch fast_lio mapping.launch.py rviz:=true &
FASTLIO_PID=$!

trap 'echo; echo "[start_slam] stopping ..."; kill $LIVOX_PID $FASTLIO_PID 2>/dev/null; pkill -f "install/fast_lio/lib/fast_lio" 2>/dev/null; pkill -x rviz2 2>/dev/null; pkill -f "install/livox_ros_driver2/lib" 2>/dev/null' INT TERM
wait
