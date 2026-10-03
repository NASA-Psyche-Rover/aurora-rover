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
Shared reprojection-error math for the calibration helper scripts.

Kept separate from overlay_math.py so the calibration tools (and their
tests) don't need to pull in the drawing code, and vice versa.
"""
from __future__ import annotations

import numpy as np
import cv2


def load_correspondences(path: str):
    """
    Read a `# u, v, x, y, z` correspondence file.

    Same format written by ros2_camera_lidar_fusion's extract_points.py --
    the two tools are interchangeable.

    Returns (pts_2d (N,2) float64, pts_3d (N,3) float64).
    """
    pts_2d, pts_3d = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            u, v, x, y, z = (float(p) for p in parts)
            pts_2d.append((u, v))
            pts_3d.append((x, y, z))
    return np.array(pts_2d, dtype=np.float64), np.array(pts_3d, dtype=np.float64)


def solve_extrinsic_pnp(pts_3d, pts_2d, K, D, ransac=False):
    """
    Solve for T_cam_lidar (4x4) via PnP given >=4 3D-2D correspondences.

    cv2.SOLVEPNP_ITERATIVE without an extrinsic guess uses a DLT
    initialization that silently *requires >=6 non-planar points*
    (OpenCV raises "DLT algorithm needs at least 6 points" below that) --
    it does not actually work down to 4 despite that being the textbook
    minimum for PnP in general. So: seed with SOLVEPNP_EPNP (valid from 4
    points, no planarity assumption) when there are fewer than 6, then
    always refine with one ITERATIVE (Gauss-Newton) pass using that as the
    extrinsic guess, for the best accuracy of either method.

    Returns (T_cam_lidar, rvec, tvec, inliers_or_None).
    """
    pts_3d = np.asarray(pts_3d, dtype=np.float64)
    pts_2d = np.asarray(pts_2d, dtype=np.float64)
    n = len(pts_3d)
    if n < 4:
        raise ValueError(f"Need at least 4 correspondences, got {n}")

    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    seed_flag = cv2.SOLVEPNP_ITERATIVE if n >= 6 else cv2.SOLVEPNP_EPNP

    if ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K, D, flags=seed_flag)
    else:
        success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, D, flags=seed_flag)
        inliers = None

    if not success:
        raise RuntimeError("solvePnP failed to find a solution")

    if seed_flag != cv2.SOLVEPNP_ITERATIVE:
        success, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d, K, D, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            raise RuntimeError("ITERATIVE refinement failed after a successful EPnP seed")

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec[:, 0]
    return T, rvec, tvec, inliers


def reprojection_errors(pts_3d, pts_2d, rvec, tvec, K, D):
    """
    Compute per-point pixel error between `pts_2d` and reprojected pts_3d.

    Reprojects pts_3d through (rvec, tvec, K, D).
    Returns (per_point_err (N,), rms_err (float)).
    """
    projected, _ = cv2.projectPoints(
        pts_3d, rvec, tvec, np.asarray(K, dtype=np.float64), np.asarray(D, dtype=np.float64)
    )
    projected = projected.reshape(-1, 2)
    err = np.linalg.norm(projected - pts_2d, axis=1)
    rms = float(np.sqrt(np.mean(err**2))) if err.size else float("nan")
    return err, rms


def reprojection_errors_from_matrix(pts_3d, pts_2d, T_cam_lidar, K, D):
    """
    Compute reprojection error starting from a 4x4 T_cam_lidar.

    Equivalent to reprojection_errors, but takes a 4x4 T_cam_lidar instead
    of an (rvec, tvec) pair -- used by check_calibration.py to re-validate
    a saved extrinsic_matrix.yaml against a correspondence file.
    """
    R = np.asarray(T_cam_lidar, dtype=np.float64)[:3, :3]
    t = np.asarray(T_cam_lidar, dtype=np.float64)[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    return reprojection_errors(pts_3d, pts_2d, rvec, tvec, K, D)
