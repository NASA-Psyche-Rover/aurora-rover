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
Generate a synthetic LiDAR+camera scene with a *known* ground truth.

Lets projection correctness be verified without any real hardware or a
recorded rosbag.

The scene is a flat rectangular "wall" placed at a known distance directly
in the camera frame, then transformed back into the LiDAR frame through
whatever T_cam_lidar you give it -- so the test is meaningful for any
extrinsic, not just identity.
"""
from __future__ import annotations

import numpy as np


def make_synthetic_wall_cloud(
    T_cam_lidar,
    wall_distance_m: float = 5.0,
    half_size_m: float = 1.5,
    grid_step_m: float = 0.05,
    jitter_std_m: float = 0.0,
    seed: int = 0,
):
    """
    Build a flat, camera-facing square grid of points, in the LiDAR frame.

    The grid sits at `wall_distance_m` (camera-frame Z).

    Returns
    -------
    xyz_lidar : array
        (N,3) points in the LiDAR frame.
    xyz_cam : array
        (N,3) the same points in the camera frame (ground truth -- by
        construction, xyz_cam[:, 2] == wall_distance_m (+/- jitter)).

    """
    rng = np.random.default_rng(seed)
    xs = np.arange(-half_size_m, half_size_m, grid_step_m)
    ys = np.arange(-half_size_m, half_size_m, grid_step_m)
    xx, yy = np.meshgrid(xs, ys)
    n = xx.size
    xyz_cam = np.stack([xx.ravel(), yy.ravel(), np.full(n, wall_distance_m)], axis=1)
    if jitter_std_m > 0:
        xyz_cam = xyz_cam + rng.normal(scale=jitter_std_m, size=xyz_cam.shape)

    T_cam_lidar = np.asarray(T_cam_lidar, dtype=np.float64)
    T_lidar_cam = np.linalg.inv(T_cam_lidar)
    xyz_lidar = xyz_cam @ T_lidar_cam[:3, :3].T + T_lidar_cam[:3, 3]
    return xyz_lidar, xyz_cam


def wall_corners_lidar(T_cam_lidar, wall_distance_m: float = 5.0, half_size_m: float = 1.5):
    """
    Return the 4 corners of the same wall, in the LiDAR frame.

    Useful for drawing an expected outline / computing expected pixel
    bounds.
    """
    corners_cam = np.array(
        [
            [-half_size_m, -half_size_m, wall_distance_m],
            [half_size_m, -half_size_m, wall_distance_m],
            [half_size_m, half_size_m, wall_distance_m],
            [-half_size_m, half_size_m, wall_distance_m],
        ]
    )
    T_cam_lidar = np.asarray(T_cam_lidar, dtype=np.float64)
    T_lidar_cam = np.linalg.inv(T_cam_lidar)
    return corners_cam @ T_lidar_cam[:3, :3].T + T_lidar_cam[:3, 3]


def blank_image(width=640, height=480, value=40):
    import numpy as np

    return np.full((height, width, 3), value, dtype=np.uint8)
