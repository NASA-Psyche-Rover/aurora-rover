#!/usr/bin/env python3

# Copyright 2026 AURORA Rover Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Visual alignment check: overlays LiDAR dots on the image's own Canny edges.

If calibration is good, dots cluster tightly along the edge lines (object
boundaries); if it's off, you'll see a visible, roughly-constant offset or
rotation between the dots and the edges they should be sitting on.

This is a manual/eyeball check by design -- there's no real-hardware
ground truth to compare against automatically, so this tool's job is to
make misalignment *obvious*, not to score it. It works on any (image, .pcd)
pair, e.g. the ones ros2_camera_lidar_fusion's save_sensor_data.py already
captures into its data/ folder.

Usage:
    ros2 run aurora_lidar_overlay visual_alignment_check \\
        --image /path/to/frame.png --cloud /path/to/frame.pcd \\
        --intrinsics /path/to/intrinsics.yaml --extrinsics /path/to/extrinsics.yaml \\
        --output /path/to/alignment_check.png
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from aurora_lidar_overlay import calibration_io, overlay_math


def load_cloud_xyz(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).reshape(-1, 3)
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)


def run(image_path, cloud_path, intrinsics_path, extrinsics_path, output_path, invert_extrinsic=False,
        max_range_m=25.0, min_range_m=0.3):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    xyz = load_cloud_xyz(cloud_path)

    K, D, width, height = calibration_io.load_intrinsics(intrinsics_path)
    T_cam_lidar = calibration_io.load_extrinsic_matrix(extrinsics_path, invert=invert_extrinsic)

    overlay = overlay_math.draw_overlay(
        image, xyz, T_cam_lidar, K, D, color_by="range", max_range_m=max_range_m, min_range_m=min_range_m,
        point_size_near_px=4, point_size_far_px=1,
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges_bgr = np.zeros_like(image)
    edges_bgr[edges > 0] = (0, 255, 0)  # green edges

    combined = cv2.addWeighted(overlay, 0.75, edges_bgr, 1.0, 0)

    side_by_side = np.hstack([overlay, combined])
    cv2.imwrite(output_path, side_by_side)
    print(f"Wrote alignment check image to {output_path}")
    print("Left: depth-colored overlay alone. Right: overlay + green Canny edges of the image.")
    print("Look at the right panel: dots should sit ON the green edge lines at object boundaries, "
          "not offset to one side. A consistent offset in one direction usually means a translation "
          "error in the extrinsic; a rotating/scaling offset with distance usually means a rotation "
          "error or wrong intrinsics.")
    return output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--cloud", required=True, help=".pcd (via open3d) or .npy (N,3) array")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--extrinsics", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--invert-extrinsic", action="store_true")
    args = ap.parse_args()
    run(args.image, args.cloud, args.intrinsics, args.extrinsics, args.output, invert_extrinsic=args.invert_extrinsic)


if __name__ == "__main__":
    main()
