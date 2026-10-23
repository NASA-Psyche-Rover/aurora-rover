# Dashboard

A local web dashboard that shows the rover's perception stack live on one screen, for demos. Three panels; click one (or press `1` `2` `3`) to enlarge it, `Esc` to go back.

| Panel | Shows | Comes from |
|---|---|---|
| **LiDAR overlay** | Mid-360 points projected onto the OAK-D image, coloured by range | `perception/ros2_ws/src/aurora_lidar_overlay` |
| **Object detection** | YOLO boxes with stereo distances, drop/slope edge, PROCEED / CAUTION / STOP, depth heatmap inset | `perception/OAKD-obj-detec.py` (`Perception`, `annotate`) |
| **LiDAR point cloud** | Live 3D render of the Mid-360 cloud, last 8 s of returns, colour = height. Drag to rotate, scroll to zoom, double-click for auto-orbit | `cloud_render.py` |

Each panel has a metrics strip (frame rates, latency, nearest distance, LiDAR points/s, 60 s trend lines). The enlarged detection view also lists the decision thresholds read from `src/decision.py` / `src/metric_terrain.py`.

## Run

```bash
dashboard/run_dashboard.sh                 # starts drivers + overlay + dashboard, opens fullscreen (kiosk)
KIOSK=0 dashboard/run_dashboard.sh         # normal browser window instead
DASHBOARD_ARGS="--host 0.0.0.0" dashboard/run_dashboard.sh   # also viewable from other machines
```

It wraps `perception/scripts/run_lidar_overlay.sh` (same drivers, same pinned overlay parameters) and adds the dashboard server on <http://localhost:8080>. `Ctrl+C` stops everything it started. Logs are in the `/tmp/lidar_overlay.*` folder it prints.

## How it fits together

The dashboard never opens a sensor. The OAK-D bridge (`sensor_bridges/oak_image_publisher`) is the one process that owns the camera and publishes RGB, camera_info and, new, aligned stereo depth (`/oak/stereo/depth`, 16UC1 mm). `server.py` subscribes to those topics plus `/livox/lidar` and the overlay output, runs detection and the cloud render in background threads, and serves MJPEG streams and `/api/metrics` JSON. Because it only reads topics, it works the same on replayed data.

## Developing without hardware

```bash
source /opt/ros/humble/setup.bash && source perception/ros2_ws/install/setup.bash
python3 dashboard/tools/fake_sensors.py &     # synthetic camera, depth and LiDAR on the real topics
ros2 launch aurora_lidar_overlay overlay.launch.py &
python3 dashboard/server.py                   # then open http://localhost:8080
```

Tests: `python3 -m pytest dashboard/tests`.
