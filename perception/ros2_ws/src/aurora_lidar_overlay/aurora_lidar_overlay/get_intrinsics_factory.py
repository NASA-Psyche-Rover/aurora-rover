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

"""
Read the OAK-D-Lite's factory-calibrated intrinsics straight off the device.

Uses depthai (no chessboard needed). Fast, and a good fallback/cross-check,
but per our own research (see README > Calibration), NOT guaranteed to
match the *exact* resolution/crop your live pipeline streams at --
depthai-core issue #751 documents a real several-pixel offset when the
requested resolution's intrinsics are queried incorrectly. Prefer
get_intrinsics_checkerboard for the intrinsics you actually trust; use this
one to sanity-check that result (they should agree to within ~1-2px scaled
to your resolution) or as a quick placeholder before you've done a proper
checkerboard capture.

Usage (camera must be plugged in):
    ros2 run aurora_lidar_overlay get_intrinsics_factory --width 640 --height 480 --output /path/to/intrinsics.yaml
"""
from __future__ import annotations

import argparse

from aurora_lidar_overlay import calibration_io


def run(width, height, output_path, socket_name="CAM_A"):
    import depthai as dai

    socket = getattr(dai.CameraBoardSocket, socket_name)
    with dai.Device() as device:
        calib = device.readCalibration()
        # Pass the *exact* output resolution -- see module docstring.
        K = calib.getCameraIntrinsics(socket, width, height)
        D = calib.getDistortionCoefficients(socket)

    print("K =", K)
    print("D =", D)
    calibration_io.save_intrinsics(
        output_path, K, D, width, height, extra={"source": "depthai_factory_calibration", "socket": socket_name}
    )
    print(f"Saved factory intrinsics to {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--socket", default="CAM_A", help="dai.CameraBoardSocket name (CAM_A = RGB on OAK-D-Lite)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    run(args.width, args.height, args.output, socket_name=args.socket)


if __name__ == "__main__":
    main()
