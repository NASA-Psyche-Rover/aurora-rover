#!/bin/bash
sudo ip addr add 192.168.1.5/24 dev enP8p1s0 2>/dev/null; true
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

echo "Starting Livox driver..."
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
DRIVER_PID=$!
sleep 8

echo "Starting FAST-LIO..."
ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml &
FASTLIO_PID=$!
sleep 5

echo "Starting RViz..."
ros2 run rviz2 rviz2 -d ~/fastlio.rviz

kill $DRIVER_PID $FASTLIO_PID 2>/dev/null
