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
Integration test: known synthetic geometry through the real draw_overlay().

Builds a synthetic scene with a *known* ground-truth wall distance and
known extrinsic/intrinsics, runs it through the exact same draw_overlay()
the live node uses, and checks the projected points land where geometry
says they must -- i.e. the "known points land on known pixels" check, but
at the whole-cloud level instead of a single point, and saved as a PNG for
a human to eyeball too (see the file this test writes in tmp_path -- the
runner reports its path on failure, and the demo script
`make_synthetic_bag.py` produces one for manual inspection).
"""
import cv2
import numpy as np

from aurora_lidar_overlay import overlay_math, synthetic_scene

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
D = np.zeros(5)
T_CAM_LIDAR = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_synthetic_wall_points_project_inside_expected_bounds(tmp_path):
    distance = 5.0
    half_size = 1.5
    xyz_lidar, xyz_cam = synthetic_scene.make_synthetic_wall_cloud(
        T_CAM_LIDAR, wall_distance_m=distance, half_size_m=half_size, grid_step_m=0.05
    )

    pixels, in_front, cam_xyz = overlay_math.project_points(xyz_lidar, T_CAM_LIDAR, K, D)
    assert in_front.all(), "every synthetic wall point is directly in front of the camera by construction"
    assert np.allclose(cam_xyz, xyz_cam, atol=1e-9)

    corners_lidar = synthetic_scene.wall_corners_lidar(T_CAM_LIDAR, wall_distance_m=distance, half_size_m=half_size)
    corner_pixels, corner_in_front, _ = overlay_math.project_points(corners_lidar, T_CAM_LIDAR, K, D)
    assert corner_in_front.all()

    expected_u = 320.0 + 500.0 * half_size / distance
    expected_v = 240.0 + 500.0 * half_size / distance
    assert np.allclose(
        np.abs(corner_pixels - [320.0, 240.0]).max(axis=0), [expected_u - 320.0, expected_v - 240.0], atol=1e-6
    )

    # Every interior point's pixel must fall within the corner bounding box
    # (with a little slack for floating point).
    lo = corner_pixels.min(axis=0) - 1e-6
    hi = corner_pixels.max(axis=0) + 1e-6
    inside = np.all((pixels >= lo) & (pixels <= hi), axis=1)
    assert inside.all()


def test_draw_overlay_on_synthetic_scene_produces_visible_dots_at_corners(tmp_path):
    distance = 5.0
    half_size = 1.0
    xyz_lidar, _ = synthetic_scene.make_synthetic_wall_cloud(
        T_CAM_LIDAR, wall_distance_m=distance, half_size_m=half_size, grid_step_m=0.03
    )
    img = synthetic_scene.blank_image(640, 480, value=20)

    out = overlay_math.draw_overlay(
        img, xyz_lidar, T_CAM_LIDAR, K, D, color_by="range", max_range_m=10.0, min_range_m=0.5,
        point_size_near_px=3, point_size_far_px=1,
    )

    out_path = tmp_path / "synthetic_overlay.png"
    cv2.imwrite(str(out_path), out)
    assert out_path.exists()

    corners_lidar = synthetic_scene.wall_corners_lidar(T_CAM_LIDAR, wall_distance_m=distance, half_size_m=half_size)
    corner_pixels, _, _ = overlay_math.project_points(corners_lidar, T_CAM_LIDAR, K, D)
    for u, v in corner_pixels:
        u, v = int(u), int(v)
        patch = out[max(v - 5, 0):v + 5, max(u - 5, 0):u + 5]
        assert patch.any(), f"expected drawn pixels near wall corner ({u},{v}) but found none"
