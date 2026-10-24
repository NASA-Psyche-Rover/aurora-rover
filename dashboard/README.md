# Dashboard

A local web dashboard that shows the rover's perception stack live on one screen, for demos. Three panels; click one (or press `1` `2` `3`) to enlarge it, `Esc` to go back.

| Panel | Shows | Comes from |
|---|---|---|
| **LiDAR overlay** | Mid-360 points projected onto the OAK-D image, coloured by range | `perception/ros2_ws/src/aurora_lidar_overlay` |
| **Object detection** | YOLO boxes with stereo distances, drop/slope edge, PROCEED / CAUTION / STOP, depth heatmap inset | `perception/OAKD-obj-detec.py` (`Perception`, `annotate`) |
| **LiDAR point cloud** | Live 3D render of the Mid-360 cloud, last 8 s of returns, colour = height, older points fade out over their last 2.5 s. Drag to rotate, scroll to zoom, double-click to reset the view (the cloud does not spin on its own). Rendered on the server (numpy + OpenCV) and streamed as MJPEG | `cloud_render.py` |

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

## Performance

The dashboard shares a Jetson with the camera, LiDAR and detection, so it is tuned to stay light:

- **Point cloud is rendered server-side, not in the browser.** A WebGL version existed (`d2007cf`, `b147fa6`) but Chromium has no GPU access on the Jetson, so it ran on software GL at ~145% CPU and starved the rest of the pipeline. To get it back: `git checkout b147fa6 -- dashboard`.
- **Cheaper cloud renderer** (28.6 -> 19.2 ms/frame): BGRA buffer with one write per point, a precomputed height x fade colour table, a cached background, a single projection for all rings, and stable subsampling with hysteresis so static scenes don't shimmer. The cloud JPEG uses 4:4:4 chroma for accurate colour.
- **`/api/metrics` is cheap**: it reads a point count instead of rebuilding the cloud, and camera frames are converted to numpy only when the detector actually takes one.
- **TensorRT detector**: `perception/src/detection.py` uses `models/<name>.engine` (FP16, 640x480) when it is newer than the weights, else falls back to `.pt`. Same detections, ~7 ms vs ~19 ms inference. Build command is in [`perception/README.md`](../perception/README.md). `*.engine` is git-ignored.
- **Faster LiDAR overlay**: `overlay_math.py` draws points with an index-layer + disk max-filter instead of a per-point `cv2.circle` loop (131 -> 42 ms/frame, pixel-identical), and projects with plain numpy when distortion is all zeros. After editing it, run `colcon build --packages-select aurora_lidar_overlay` (`install/` is a copy).

Measured on the same scene: detection 6.7-7.6 fps / 111-158 ms latency before, ~10 fps / 64-67 ms after. Depth and terrain code (`metric_terrain.py`) is deliberately untouched, since it is the rover's safety logic. Details and gotchas: [`OPTIMIZATION_NOTES.md`](OPTIMIZATION_NOTES.md).

## Troubleshooting

- `run_dashboard.sh` fails right after a restart with "camera image topic ... not publishing": the ROS graph still remembers the old camera for a few seconds. Wait ~10 s and run it again.
- Port 8080 stays busy after a stop: the server sometimes ignores plain `kill`; use `kill -9` on its PID.
- Closing AnyDesk while testing frees ~70% of a core, more than any code change.

## Developing without hardware

```bash
source /opt/ros/humble/setup.bash && source perception/ros2_ws/install/setup.bash
python3 dashboard/tools/fake_sensors.py &     # synthetic camera, depth and LiDAR on the real topics
ros2 launch aurora_lidar_overlay overlay.launch.py &
python3 dashboard/server.py                   # then open http://localhost:8080
```

Tests: `python3 -m pytest dashboard/tests`.
