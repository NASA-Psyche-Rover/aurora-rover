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
Independent calibration sanity check.

Reloads a saved extrinsics.yaml + intrinsics.yaml and a correspondence
file, and reports reprojection error -- without re-solving anything. Use
this any time you want to re-validate a calibration you already have (e.g.
after the rig gets bumped) before trusting the live overlay.

Usage:
    ros2 run aurora_lidar_overlay check_calibration \
        --correspondences /path/to/correspondences.txt \
        --intrinsics /path/to/intrinsics.yaml \
        --extrinsics /path/to/extrinsics.yaml
"""
from __future__ import annotations

import argparse

from aurora_lidar_overlay import calibration_io, reprojection


def run(correspondences_path, intrinsics_path, extrinsics_path, invert_extrinsic=False):
    K, D, width, height = calibration_io.load_intrinsics(intrinsics_path)
    T_cam_lidar = calibration_io.load_extrinsic_matrix(extrinsics_path, invert=invert_extrinsic)
    pts_2d, pts_3d = reprojection.load_correspondences(correspondences_path)

    err, rms = reprojection.reprojection_errors_from_matrix(pts_3d, pts_2d, T_cam_lidar, K, D)
    print(f"{len(pts_2d)} correspondences from {correspondences_path}")
    print(f"Per-point reprojection error (px): min={err.min():.2f} max={err.max():.2f} mean={err.mean():.2f}")
    print(f"RMS reprojection error: {rms:.3f} px")
    for i, e in enumerate(err):
        flag = "  <-- outlier?" if e > rms * 2 and e > 2.0 else ""
        print(f"  point {i}: {e:.2f} px{flag}")
    return err, rms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correspondences", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--extrinsics", required=True)
    ap.add_argument(
        "--invert-extrinsic", action="store_true",
        help="Invert the loaded extrinsic before checking (matches the node's invert_extrinsic param)",
    )
    args = ap.parse_args()
    run(args.correspondences, args.intrinsics, args.extrinsics, invert_extrinsic=args.invert_extrinsic)


if __name__ == "__main__":
    main()
