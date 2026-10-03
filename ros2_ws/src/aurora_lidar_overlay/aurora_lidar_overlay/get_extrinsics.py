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
Solve LiDAR->camera extrinsics from a manual 2D-3D correspondence file.

Uses solvePnP, and reports reprojection error immediately -- no separate
step needed to find out whether the result is any good.

This is the "no special target needed" calibration path recommended for
this rig (Livox Mid-360 + OAK-D-Lite, no printed calibration target on
hand yet). Upgrade path: hku-mars/FAST-Calib once someone prints its
ArUco+circular-hole target -- see the package README.

Usage (after capturing a correspondence file -- e.g. with
ros2_camera_lidar_fusion's extract_points.py, same file format):

    ros2 run aurora_lidar_overlay get_extrinsics \
        --correspondences /path/to/correspondences.txt \
        --intrinsics /path/to/intrinsics.yaml \
        --output /path/to/extrinsics.yaml
"""
from __future__ import annotations

import argparse

from aurora_lidar_overlay import calibration_io, reprojection


def run(correspondences_path, intrinsics_path, output_path, ransac=False):
    K, D, width, height = calibration_io.load_intrinsics(intrinsics_path)
    pts_2d, pts_3d = reprojection.load_correspondences(correspondences_path)
    print(f"Loaded {len(pts_2d)} correspondences from {correspondences_path}")

    T_cam_lidar, rvec, tvec, inliers = reprojection.solve_extrinsic_pnp(pts_3d, pts_2d, K, D, ransac=ransac)
    err, rms = reprojection.reprojection_errors(pts_3d, pts_2d, rvec, tvec, K, D)

    print("T_cam_lidar (p_cam = T @ p_lidar):")
    print(T_cam_lidar)
    print(f"Per-point reprojection error (px): min={err.min():.2f} max={err.max():.2f} mean={err.mean():.2f}")
    print(f"RMS reprojection error: {rms:.3f} px")
    if rms < 0.5:
        print("-> excellent, trust this for pixel-precise overlay work.")
    elif rms < 1.0:
        print("-> acceptable for typical robotics use, but expect visible edge misalignment.")
    elif rms < 2.0:
        print("-> marginal -- consider picking more/better-spread correspondence points and re-running.")
    else:
        print("-> poor -- do not trust this extrinsic. Re-pick points (check for mismatched picking order "
              "between the image and cloud) and re-run before using it live.")

    calibration_io.save_extrinsic_matrix(
        output_path,
        T_cam_lidar,
        meta={
            "rms_reprojection_error_px": rms,
            "num_correspondences": int(len(pts_2d)),
            "solver": "cv2.solvePnPRansac" if ransac else "cv2.solvePnP",
        },
    )
    print(f"Saved extrinsic_matrix to {output_path}")
    return T_cam_lidar, err, rms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correspondences", required=True, help="Path to the u,v,x,y,z correspondence file")
    ap.add_argument(
        "--intrinsics", required=True, help="Path to intrinsics.yaml (camera_matrix/distortion_coefficients)"
    )
    ap.add_argument("--output", required=True, help="Where to save the resulting extrinsics.yaml")
    ap.add_argument(
        "--ransac", action="store_true",
        help="Use solvePnPRansac instead of plain solvePnP (robust to a bad pick or two)",
    )
    args = ap.parse_args()
    run(args.correspondences, args.intrinsics, args.output, ransac=args.ransac)


if __name__ == "__main__":
    main()
