# One-command LiDAR -> OAK-D overlay

```bash
~/psyche-objdetec/scripts/run_lidar_overlay.sh     # NO_VIEWER=1 to skip rqt_image_view
```

Sources ROS 2 Humble + `ros2_ws`, builds `aurora_lidar_overlay`/`sensor_bridges` only if stale,
starts the OAK-D bridge and the Livox driver (PointCloud2, `xfer_format=0`; skipped if already
publishing), waits for the three topics, launches the overlay, opens `rqt_image_view`.

Pinned parameters: image `/oak/rgb/image_raw`, camera_info `/oak/rgb/camera_info`,
LiDAR `/livox/lidar`, output `/aurora/lidar_overlay/image`, `invert_extrinsic:=false`,
`use_tf:=false`, `color_by:=depth`, `skip:=1`; everything else from `config/overlay_params.yaml`
(`sync_slop` 0.1, range 0.3-25 m, point size 6->1 px).

**Calibration caveat:** `config/calibration/{extrinsics,intrinsics}.yaml` are still placeholders
(identity extrinsic). With identity only ~5 of ~20k points land in the image. The overlay will not
look good until a real extrinsic (README > Calibration in the package) is installed.
