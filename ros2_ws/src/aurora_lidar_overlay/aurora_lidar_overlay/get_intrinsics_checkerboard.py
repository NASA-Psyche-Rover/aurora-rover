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
Checkerboard intrinsic calibration at your exact live streaming resolution.

This is the recommended source of truth for intrinsics (see README >
Calibration for why factory calibration alone isn't good enough for
pixel-precise overlay work).

Captures live from the OAK-D-Lite via the same OakDLiteCamera class the
rest of the rover uses (src/camera_stream.py), so the resolution you
calibrate at is guaranteed to match what you stream live -- no separate
camera driver, no resolution mismatch.

Controls: 'c' to capture the current frame (only works when a chessboard
is detected), 'q' or ESC to finish and run the calibration.

Usage:
    ros2 run aurora_lidar_overlay get_intrinsics_checkerboard \
        --rows 6 --columns 9 --square-size-m 0.025 \
        --width 640 --height 480 --output /path/to/intrinsics.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from aurora_lidar_overlay import calibration_io

MIN_CAPTURES = 10


def run(rows, columns, square_size_m, width, height, output_path, repo_root=None):
    repo_root = repo_root or os.environ.get("PSYCHE_OBJDETEC_ROOT", "/home/aurora/psyche-objdetec")
    sys.path.insert(0, repo_root)
    from src.camera_stream import OakDLiteCamera  # noqa: E402

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points, img_points = [], []
    cam = OakDLiteCamera(rgb_size=(width, height), fps=30)
    print("Point the camera at the checkerboard. Press 'c' to capture, 'q'/ESC to finish.")
    try:
        while True:
            rgb, _depth = cam.read()
            if rgb is None:
                continue
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (columns, rows), None)

            display = rgb.copy()
            if found:
                cv2.drawChessboardCorners(display, (columns, rows), corners, found)
            cv2.putText(
                display, f"captures: {len(obj_points)} (need >= {MIN_CAPTURES})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("checkerboard calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("c") and found:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                obj_points.append(objp)
                img_points.append(refined)
                print(f"Captured view {len(obj_points)}")
    finally:
        cam.close()
        cv2.destroyAllWindows()

    if len(obj_points) < MIN_CAPTURES:
        print(
            f"Only {len(obj_points)} captures (< {MIN_CAPTURES}) -- "
            f"calibration would be unreliable. Aborting without saving."
        )
        return None

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (width, height), None, None)
    print(f"RMS reprojection error: {rms:.3f} px")
    if rms < 0.5:
        print("-> excellent.")
    elif rms < 1.0:
        print("-> acceptable, but consider a few more varied-angle captures for a pixel-precise overlay.")
    else:
        print("-> poor -- recapture with more views covering the frame edges/corners and varied angles/distances.")

    calibration_io.save_intrinsics(
        output_path, K, D, width, height,
        extra={
            "source": "cv2.calibrateCamera (checkerboard)",
            "rms_reprojection_error_px": float(rms),
            "num_captures": len(obj_points),
            "chessboard_rows": rows,
            "chessboard_columns": columns,
            "square_size_meters": square_size_m,
        },
    )
    print(f"Saved intrinsics to {output_path}")
    return K, D, rms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, required=True, help="Inner corner rows (NOT square count)")
    ap.add_argument("--columns", type=int, required=True, help="Inner corner columns (NOT square count)")
    ap.add_argument("--square-size-m", type=float, required=True, help="Real physical square size, meters")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    run(args.rows, args.columns, args.square_size_m, args.width, args.height, args.output)


if __name__ == "__main__":
    main()
