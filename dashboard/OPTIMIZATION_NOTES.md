# Dashboard / perception optimization notes (2026-10-22)

Branch: `dashboard-pointcloud`. **Nothing here is committed yet.** The revert to the pre-WebGL
dashboard is staged; the later optimizations sit on top as unstaged changes.

## Where things stand

- The point cloud is **server-rendered again** (numpy + OpenCV, streamed as MJPEG). WebGL was
  dropped: Chromium has no GPU access on the Jetson, so it ran on software GL (~145% CPU) and starved
  the camera/LiDAR/detection pipeline. The WebGL work is still in history: `d2007cf`, `b147fa6`.
  To get it back: `git checkout b147fa6 -- dashboard`.
- Dashboard URL: http://localhost:8080. Start it with `KIOSK=0 dashboard/run_dashboard.sh`.

## Results (same scene, stack running)

| | Before | After |
|---|---|---|
| Detection | 6.7-7.6 fps, 111-158 ms latency | ~9.6-10 fps, 64-67 ms |
| Overlay draw per frame (synthetic) | 131 ms | 42 ms (pixel-identical output) |
| Cloud render per frame | 28.6 ms | 19.2 ms |
| Server threads (detect / cloud / ROS cb) | 42 / 38 / 21 % | 39 / 31 / 17 % |

Total CPU is about the same because the system now does more work per second. Overlay fps is tied to
the LiDAR rate (~8-10 Hz). **AnyDesk uses ~70% of a core**; closing it frees more than any code change.

## What changed

### Point cloud (`dashboard/cloud_render.py`, `dashboard/server.py`)
- Render into a BGRA buffer and write one 32-bit value per point (was 3 channel writes).
- One precomputed height x fade colour table instead of several numpy passes.
- Cached background canvas (copy instead of rebuild, saved 7.6 ms) and one projection for all 4 rings.
- `np.cross` replaced with plain math; range filter uses squared range.
- **Stable subsampling** with hysteresis. It used to pick a random offset per render, so static scenes
  shimmered.
- **Age fade**: points dim over their last 2.5 s instead of vanishing in chunks.
- Cloud JPEG uses **4:4:4 chroma** (colour error 14.9 -> 6.4, costs ~8 ms/frame).
- `/api/metrics` uses a cheap `n_points()` instead of rebuilding the 100k-point cloud on every poll.
- Camera frames are converted to numpy only when the detector takes one (~10/s, not 30/s).
- Tried and dropped: a fast PointCloud2 parser (saved only 0.1 ms), JPEG quality/flag tuning (no gain).

### YOLO (`perception/src/detection.py`)
- `ObjectDetector` prefers `models/<name>.engine` (TensorRT FP16, 640x480) when it sits next to the
  weights and is newer; it falls back to `.pt` if the engine is missing or fails to load.
- Same detections as PyTorch on a real frame (9/9 boxes, IoU > 0.9, same confidences). Inference 19 ms ->
  7 ms on an idle GPU (35-47 ms -> much less under full load).
- Results are copied off the GPU once per frame instead of one sync per box/field.
- `*.engine` is in `.gitignore` (device + TensorRT specific). Rebuild command is in `perception/README.md`.
  The export needed `onnx` + `onnxslim`; I installed them into a throwaway folder and used `PYTHONPATH`
  for the export only, so the normal Python env is untouched.
- **`models/best.pt` is an empty placeholder**, so the rover runs stock `yolov8n.pt`. A trained `best.pt`
  needs its own engine export.

### LiDAR overlay (`perception/ros2_ws/src/aurora_lidar_overlay/.../overlay_math.py`)
- Replaced the per-point Python `cv2.circle` loop with an index-layer + disk max-filter (`_draw_disks`).
  Pixel-identical to the original.
- Pinhole projection in numpy when distortion is all zeros (placeholder calibration); `cv2.projectPoints`
  is still used for any non-zero distortion.
- Added tests in `test/test_overlay_math.py` that compare both shortcuts against the original cv2 code.
  All 38 package tests pass (incl. flake8 / pep257).
- After editing the source, run `colcon build --packages-select aurora_lidar_overlay`: `install/` is a
  copy, not a symlink.

## Deliberately not changed
- Depth / terrain code (`metric_terrain.py` etc.): it is the rover's safety logic, with temporal blending
  and thresholds tuned to 640x480. Downscaling depth would change what the rover decides.
- No resolution was lowered. The camera pipeline is already 640x480 (below 720p), and YOLO input size
  barely changed inference time here.

## Gotchas
- `run_dashboard.sh` sometimes fails right after a restart ("camera image topic ... not publishing"). The
  ROS graph still remembers the old camera for a few seconds, so the script skips starting the driver.
  Wait ~10 s and run it again.
- Don't use `pkill -f` / `pgrep -f` with a pattern that appears in your own command line: it kills the
  shell. Kill by PID instead.
- The server ignores plain `kill` sometimes; if port 8080 stays busy use `kill -9` on that PID.
- A stale Python traceback (`NameError: name 'open'`) sits in an old terminal. It looks like an old
  shutdown error and doesn't affect the dashboard.

## Possible next steps
- Tune or throttle YOLO further (would need care: it drives the rover's decisions).
- Commit the work (suggest: dashboard revert + cloud optimizations, overlay speedup, and TensorRT
  detector as separate commits).
- Re-export the TensorRT engine once a trained `best.pt` exists.
- Close AnyDesk while testing for more CPU headroom.
