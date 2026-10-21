# aurora_lidar_overlay

Waymo-style, depth-colored LiDAR-on-camera overlay for the AURORA rover
(Livox Mid-360 + OAK-D-Lite, ROS 2 Humble, Jetson). Runs identically live
or from a rosbag replay -- it just subscribes to topics.

## Architecture

- **`overlay_math.py`** -- pure numpy/OpenCV math (`project_points`,
  `draw_overlay`). No ROS imports; unit-tested standalone.
- **`pointcloud_utils.py`** -- vectorized `PointCloud2` -> numpy, reading
  each field's real offset/datatype from the message (works against
  `livox_ros_driver2`'s padded `x,y,z,intensity,tag,line,timestamp` layout
  and simpler `x,y,z`-only clouds alike).
- **`calibration_io.py`** -- YAML I/O. Extrinsic schema: a single
  `extrinsic_matrix` (4x4), applied as `p_cam = extrinsic_matrix @ p_lidar`
  (same convention as `ros2_camera_lidar_fusion`). Intrinsic schema:
  `camera_matrix` / `distortion_coefficients` (also shared with that repo).
- **`overlay_node.py`** -- the live/replay ROS 2 node.
- **`reprojection.py`, `get_extrinsics.py`, `check_calibration.py`,
  `get_intrinsics_*.py`** -- calibration tooling (below).
- **`synthetic_scene.py`, `make_synthetic_bag.py`, `visual_alignment_check.py`**
  -- verification without needing the live rover.

### Why this design

- **No depthai-ros.** The rig already has a working, hardware-validated v3
  DepthAI pipeline (`src/camera_stream.py`'s `OakDLiteCamera`) bridged into
  ROS 2 by `sensor_bridges/oak_image_publisher.py`. Standing up depthai-ros
  as a second camera driver on top of that would be pure duplication.
  `oak_image_publisher.py` was extended (not replaced) to also publish a
  matching `sensor_msgs/CameraInfo` on `/oak/rgb/camera_info` -- it was
  Image-only before, which is why the overlay node had nothing to project
  through.
- **LiDAR source: prefer the real `livox_ros_driver2` (xfer_format=0,
  PointCloud2/PointXYZRTLT), not the raw-UDP `sensor_bridges/livox_pointcloud_publisher.py`.**
  `livox_ros_driver2` is already built in `~/ros2_ws` and already proven
  against real hardware (it feeds FAST-LIO for SLAM) -- and critically, in
  PointCloud2 mode it publishes `x, y, z, intensity, tag, line, timestamp`
  per point, where `timestamp` is a real per-point float64 stamp. That's
  exactly what per-point motion deskew needs, for free, with zero new
  code. The raw-UDP bridge's packet struct has no per-point time (only a
  ~100ms batch timestamp) and was written under the mistaken assumption
  that no ROS 2 Livox driver was available on this box -- it is, in a
  different colcon workspace (`~/ros2_ws`, not `~/psyche-objdetec/perception/ros2_ws`).
  It's kept as a documented fallback for quick bring-up without sourcing
  that workspace, but treat the real driver as the primary path. See
  **Running live** below for the exact launch command
  (`msg_MID360_launch.py` uses `xfer_format=1`/CustomMsg for FAST-LIO;
  you'll want a *separate* driver instance or launch file at
  `xfer_format=0` for the overlay -- both can't come out of one running
  driver instance).
- **Extrinsic path: manual point-picking + `solvePnP`, not FAST-Calib, for
  now.** FAST-Calib (`hku-mars/FAST-Calib`) natively supports the Mid-360
  and would give a better, repeatable extrinsic, but needs its
  ArUco+circular-hole target printed first, which nobody has done yet. The
  manual path needs no new hardware and reuses
  `ros2_camera_lidar_fusion`'s already-installed, already-staged capture
  UI (`save_sensor_data.py` for synchronized image+cloud capture,
  `extract_points.py` for point picking) -- this package's own
  `get_extrinsics` script reads that same correspondence-file format.
  **Upgrade path:** once the target is printed, switch to FAST-Calib and
  feed its result straight into `extrinsics.yaml` (same 4x4 schema).
- **Intrinsics: checkerboard at your exact streaming resolution, not
  factory calibration, as the source of truth.** The OAK-D-Lite's factory
  calibration is a fine fallback/cross-check (`get_intrinsics_factory.py`),
  but `depthai-core` issue #751 documents a real several-pixel offset when
  querying it for anything other than the literal resolution/crop mode you
  stream at -- exactly the failure mode that makes projected points
  consistently miss edges. `get_intrinsics_checkerboard.py` calibrates
  through the *same* `OakDLiteCamera`/`requestOutput()` call the live
  pipeline uses, so there's no resolution mismatch to get wrong.
- **CameraInfo is best-effort, not strictly time-synced.** The node
  subscribes to `image`, `camera_info`, and `pointcloud` as asked, but
  only 2-way `ApproximateTimeSynchronizer`s `image` + `pointcloud`;
  `camera_info` is applied asynchronously (whenever the latest one
  arrives) with a static-YAML fallback until the first one shows up.
  Camera intrinsics don't change frame-to-frame, so an exact-time match
  isn't meaningful, and requiring one would make the node fragile to a
  camera_info publisher that ticks at a different rate or briefly stalls.
- **Deskew is opportunistic.** It only activates when the cloud has a
  per-point `timestamp` field *and* `use_tf: true` with a working
  `deskew_reference_frame -> lidar_frame` TF chain (e.g. from FAST-LIO's
  odometry). Otherwise it's a one-time log line and the raw points are
  used -- per the task brief ("when that data is available").

## Build

```bash
cd ~/psyche-objdetec/perception/ros2_ws
colcon build --packages-select aurora_lidar_overlay sensor_bridges --symlink-install
source install/setup.bash
```

Run the test suite (unit tests + ament lint -- flake8, pep257, copyright):

```bash
cd src/aurora_lidar_overlay
python3 -m pytest test/ -v
```

(`colcon test` reports "Ran 0 tests" for every ament_python package on this
machine, this one included -- it only has `colcon-python-setup-py`
installed, which discovers tests via `python3 -m unittest`, and none of
this package's tests (or `ros2_camera_lidar_fusion`'s, already in this
workspace) are `unittest.TestCase`-based. Install
`python3-colcon-pytest`/`ros-humble-ament-cmake-pytest`'s colcon
counterpart to make `colcon test` itself pytest-aware if you want that
path instead -- until then, the `pytest` invocation above is the reliable
one.)

## Calibration

All calibration files live under `config/calibration/` by default (or pass
`--output`/`extrinsic_yaml`/`intrinsic_yaml` explicitly). The two
placeholder files there are guesses -- replace them before trusting the
overlay.

### 1. Intrinsics (do this first)

Recommended (checkerboard, at your real streaming resolution):

```bash
ros2 run aurora_lidar_overlay get_intrinsics_checkerboard \
    --rows 6 --columns 9 --square-size-m 0.025 \
    --width 640 --height 480 \
    --output ~/psyche-objdetec/perception/ros2_ws/src/aurora_lidar_overlay/config/calibration/intrinsics.yaml
```
(`--rows`/`--columns` are INNER corner counts, not square counts. Capture
>=10 views covering the frame edges/corners and a few different angles,
press 'c' when the chessboard overlay lights up, 'q' when done.)

Quick fallback/cross-check (factory calibration, camera must be plugged in):

```bash
ros2 run aurora_lidar_overlay get_intrinsics_factory --width 640 --height 480 \
    --output /tmp/intrinsics_factory_check.yaml
```

Both report an RMS reprojection error in pixels: **<0.5px excellent, <1px
acceptable, >1-2px re-shoot.**

### 2. Extrinsics (LiDAR -> camera)

Capture synchronized image+cloud pairs and pick corresponding points with
the already-installed `ros2_camera_lidar_fusion` package (same
correspondence-file format `get_extrinsics` reads):

```bash
ros2 run ros2_camera_lidar_fusion save_sensor_data     # press Enter to snapshot a few pairs, Ctrl+C when done
ros2 run ros2_camera_lidar_fusion extract_points       # click 4+ matching points per pair (image window, then Open3D window)
```

Then solve + immediately see the reprojection error:

```bash
ros2 run aurora_lidar_overlay get_extrinsics \
    --correspondences ~/psyche-objdetec/perception/ros2_ws/src/ros2_camera_lidar_fusion/data/ros2_camera_lidar_fusion_correspondences.txt \
    --intrinsics ~/psyche-objdetec/perception/ros2_ws/src/aurora_lidar_overlay/config/calibration/intrinsics.yaml \
    --output ~/psyche-objdetec/perception/ros2_ws/src/aurora_lidar_overlay/config/calibration/extrinsics.yaml
```

Re-check an existing calibration at any time (e.g. after the mount gets
bumped) without re-solving:

```bash
ros2 run aurora_lidar_overlay check_calibration \
    --correspondences <same file> --intrinsics <same file> --extrinsics <same file>
```

### 3. Visual sanity check

```bash
ros2 run aurora_lidar_overlay visual_alignment_check \
    --image <a saved .png> --cloud <its matching .pcd> \
    --intrinsics config/calibration/intrinsics.yaml --extrinsics config/calibration/extrinsics.yaml \
    --output /tmp/alignment_check.png
```
Look at the right-hand panel (overlay + green Canny edges): dots should
sit ON the edges, not offset to one side.

## Running live

Terminal 1 -- camera bridge (now also publishes CameraInfo):
```bash
source ~/psyche-objdetec/perception/ros2_ws/install/setup.bash
ros2 run sensor_bridges oak_image_publisher --ros-args \
    -p intrinsics_yaml:=$HOME/psyche-objdetec/perception/ros2_ws/src/aurora_lidar_overlay/config/calibration/intrinsics.yaml
```

Terminal 2 -- LiDAR (real driver, recommended; needs `~/ros2_ws` sourced,
and `sudo ip addr add 192.168.1.5/24 dev enP8p1s0` if not already set --
see `~/launch_slam.sh`):
```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
# xfer_format=0 (PointCloud2) -- edit launch_ROS2/msg_MID360_launch.py's
# copy or pass the param directly:
ros2 launch livox_ros_driver2 msg_MID360_launch.py xfer_format:=0
```
(Fallback without the driver workspace: `ros2 run sensor_bridges livox_pointcloud_publisher` --
loses per-point timestamps, so deskew won't activate.)

Terminal 3 -- overlay:
```bash
source ~/psyche-objdetec/perception/ros2_ws/install/setup.bash
ros2 launch aurora_lidar_overlay overlay.launch.py
```

View it: `ros2 run rqt_image_view rqt_image_view /aurora/lidar_overlay/image`
(or add an Image display in rviz2).

## Replay from a rosbag

Record one live (once the above is running):
```bash
ros2 bag record /oak/rgb/image_raw /oak/rgb/camera_info /livox/lidar /tf /tf_static -o my_bag
```

Replay it:
```bash
ros2 launch aurora_lidar_overlay overlay.launch.py use_sim_time:=true &
ros2 bag play my_bag --clock
```

No real bag yet? Generate a synthetic one with a known scene (a
camera-facing wall at a known, slowly-breathing distance) and matching
ground-truth calibration files:
```bash
ros2 run aurora_lidar_overlay make_synthetic_bag --output /tmp/synthetic_bag --frames 40
# it prints the exact overlay_node command using the calibration it just wrote
ros2 bag play /tmp/synthetic_bag --clock
```

## Troubleshooting

- **Dots offset to one side, consistently:** translation error in the
  extrinsic. Re-run `check_calibration` -- a high RMS confirms it. Also
  double check `invert_extrinsic`: if you solved for `T_lidar_cam` instead
  of `T_cam_lidar` (e.g. the correspondence order was image-points-as-3D
  swapped with cloud-points-as-2D), flip this parameter.
- **Dots rotate/scale wrong with distance:** rotation error in the
  extrinsic, or wrong intrinsics (wrong resolution's K). Re-run
  `get_intrinsics_checkerboard` at the *exact* resolution `image_topic`
  actually streams, and re-run `get_extrinsics` after.
- **Dots upside-down or mirrored:** almost always a frame-convention bug,
  not a numeric calibration error -- check `lidar_frame`/`camera_frame`
  axis conventions match what you assumed when picking correspondence
  points (LiDAR: x-forward/y-left/z-up is Livox's convention; camera
  frames in this package are OpenCV-style x-right/y-down/z-forward).
- **Dots drift over time / while the rover moves:** check the logged
  image<->lidar time offset (`ros2 topic ... ` isn't needed --
  `overlay_node`'s own log prints it every `log_time_offset_period_sec`).
  A large, growing offset means clock sync or `use_sim_time` mismatch
  between nodes. If it's the LiDAR sweep-vs-image-instant skew during
  motion specifically: confirm `deskew_enabled: true`, `use_tf: true`, a
  per-point `timestamp` field is actually present (only the real
  `livox_ros_driver2` PointCloud2 mode has one), and that
  `deskew_reference_frame -> lidar_frame` TF is actually being published
  (FAST-LIO does this while running).
- **No dots at all:** check `ros2 topic hz` on all three input topics;
  check the node's startup log for "No intrinsics yet" (no CameraInfo, no
  fallback yaml found) or a `use_tf` TF-lookup warning.

## Assumptions made building this

- LiDAR: Livox Mid-360 via `livox_ros_driver2` (built in `~/ros2_ws`) is
  the primary intended source; the raw-UDP `sensor_bridges` bridge is a
  documented fallback, not the primary path.
- Camera: OAK-D-Lite via the existing custom v3 DepthAI pipeline
  (`src/camera_stream.py`), not depthai-ros.
- No calibration target exists yet, so the extrinsic path defaults to
  manual point-picking + solvePnP, not FAST-Calib.
- `lidar_frame`/`camera_frame` default names (`livox_frame`, `oak_camera`)
  match what's already used elsewhere in this repo
  (`general_configuration.yaml`, `oak_image_publisher.py`) -- update the
  params if your real TF tree uses different names.
- Deskew assumes a `deskew_reference_frame` (default `odom`) is actually
  being published relative to `lidar_frame` (e.g. by FAST-LIO); if nothing
  publishes that, deskew silently no-ops (by design).

## What still needs real hardware to confirm

- Actual calibration numbers (intrinsics + extrinsics) -- everything
  shipped here is either a placeholder or synthetic; see **Calibration**.
- Real-world FPS/point-density on the Jetson with the actual Mid-360
  point rate (tune `skip`/`max_points` against what you observe).
- Whether `sync_slop`/`sync_queue_size` defaults (0.1s / 10) are right for
  the real image/LiDAR publish-rate mismatch on this rig.
- Whether the TF tree needed for `use_tf`/deskew (odom -> ... ->
  `lidar_frame`, `camera_frame`) actually exists in your launch setup, or
  needs a `static_transform_publisher` added for the fixed camera<->LiDAR
  mount.
- The actual `livox_ros_driver2` `xfer_format=0` PointCloud2 field layout
  and per-point `timestamp` semantics were confirmed by reading its C++
  source on this machine, not by running it against live hardware yet.

## Live aim-and-tune (LiDAR front as reference)

`config/calibration/extrinsics.yaml` is the axis map LiDAR(x fwd, y left, z up) -> camera(x right,
y down, z fwd), zero translation, no tilt correction: **aim the camera the same way the LiDAR front
points.** The node logs every 2 s: `aim: N pts in image ... | LiDAR front +x deg above horizon, roll y deg`
(from `/livox/imu`) plus a hint of which way to rotate if few points are visible. A white cross marks
where the LiDAR front axis lands (magenta X = image centre).

```bash
~/psyche-objdetec/perception/scripts/tune_overlay_extrinsic.sh                       # rqt_reconfigure GUI
ros2 param set /aurora_lidar_overlay_node tune_pitch_deg 3.0              # CLI; also tune_roll_deg, tune_yaw_deg, tune_x_cm, tune_y_cm, tune_z_cm
~/psyche-objdetec/perception/scripts/save_overlay_extrinsic.sh                       # writes tuned T into extrinsics.yaml, zeroes the deltas
```
