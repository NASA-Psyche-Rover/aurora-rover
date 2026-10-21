# AURORA LiDAR (Livox Mid-360) + FAST-LIO SLAM

Separate from the camera/YOLO code; nothing outside `lidar/` is touched.

## Network
LiDAR `192.168.1.115`, Jetson `192.168.1.5/24` on `enP8p1s0` (UDP point data port 56301):
`sudo ip addr add 192.168.1.5/24 dev enP8p1s0`

## Setup (once)
`lidar/setup_ws.sh` clones pinned `livox_ros_driver2` (4a1def9) + `FAST_LIO` ROS2 (a4743b0), applies our
`patches/` (driver sets connect_state on data; without it no cloud is published) and `config/`, then `colcon build`.
Note: it is untested on a clean machine; compare against the upstream READMEs if the build complains.

## Run live SLAM map (RViz, fixed frame `camera_init`, `/cloud_registered`)
`~/ros2_ws/start_slam.sh`
Save map (other terminal): `ros2 service call /map_save std_srvs/srv/Trigger` -> `~/ros2_ws/PCD/room_map.pcd`
(needs `FASTRTPS_DEFAULT_PROFILES_FILE=~/ros2_ws/fastdds_localhost_unicast.xml`, see start_slam.sh).

## Standalone viewer (no ROS)
`python3 lidar/tools/lidar_mapv2.py` (needs pygame, numpy; don't run with the ROS driver).

## LiDAR->camera overlay
Lives on branch `overlay-one-command-run`: `scripts/run_lidar_overlay.sh`. Extrinsic calibration is still a placeholder.

## Known issues
Multicast DDS discovery is broken on the Jetson (hence the unicast FastDDS profile). Map save is untested on hardware.
