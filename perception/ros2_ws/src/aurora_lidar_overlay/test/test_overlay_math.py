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

import cv2
import numpy as np
import pytest

from aurora_lidar_overlay import overlay_math

IDENTITY = np.eye(4)


def _simple_intrinsics():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5)
    return K, D


def test_known_point_projects_to_known_pixel():
    K, D = _simple_intrinsics()
    # A point 2m straight ahead, on-axis, must land exactly on the
    # principal point (cx, cy) regardless of K's focal length.
    xyz = np.array([[0.0, 0.0, 2.0]])
    pixels, in_front, cam_xyz = overlay_math.project_points(xyz, IDENTITY, K, D)
    assert in_front[0]
    assert np.allclose(pixels[0], [320.0, 240.0], atol=1e-6)

    # A point offset by X=1m at Z=2m should land at u = cx + fx * X / Z.
    xyz2 = np.array([[1.0, 0.0, 2.0]])
    pixels2, in_front2, _ = overlay_math.project_points(xyz2, IDENTITY, K, D)
    expected_u = 320.0 + 500.0 * 1.0 / 2.0
    assert in_front2[0]
    assert np.allclose(pixels2[0], [expected_u, 240.0], atol=1e-6)


def test_known_point_with_nontrivial_extrinsic():
    K, D = _simple_intrinsics()
    # LiDAR frame: x-forward, y-left, z-up. Camera frame: x-right, y-down,
    # z-forward. A point 3m directly ahead of the LiDAR, on its centerline,
    # should still land on the principal point after the axis remap.
    T_cam_lidar = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    xyz = np.array([[3.0, 0.0, 0.0]])
    pixels, in_front, cam_xyz = overlay_math.project_points(xyz, T_cam_lidar, K, D)
    assert in_front[0]
    assert np.allclose(cam_xyz[0], [0.0, 0.0, 3.0], atol=1e-9)
    assert np.allclose(pixels[0], [320.0, 240.0], atol=1e-6)


def test_point_behind_camera_is_dropped():
    K, D = _simple_intrinsics()
    xyz = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]])
    pixels, in_front, _ = overlay_math.project_points(xyz, IDENTITY, K, D)
    assert not in_front[0]
    assert in_front[1]


def test_point_exactly_at_camera_is_dropped():
    K, D = _simple_intrinsics()
    xyz = np.array([[0.0, 0.0, 0.0]])
    _, in_front, _ = overlay_math.project_points(xyz, IDENTITY, K, D)
    assert not in_front[0]


def test_empty_cloud_project_points_does_not_crash():
    K, D = _simple_intrinsics()
    xyz = np.zeros((0, 3))
    pixels, in_front, cam_xyz = overlay_math.project_points(xyz, IDENTITY, K, D)
    assert pixels.shape == (0, 2)
    assert in_front.shape == (0,)
    assert cam_xyz.shape == (0, 3)


def test_empty_cloud_draw_overlay_does_not_crash_and_returns_unmodified_copy():
    K, D = _simple_intrinsics()
    img = np.full((480, 640, 3), 100, dtype=np.uint8)
    out = overlay_math.draw_overlay(img, np.zeros((0, 3)), IDENTITY, K, D)
    assert out.shape == img.shape
    assert np.array_equal(out, img)
    assert out is not img  # must not mutate/alias the input


def test_draw_overlay_never_mutates_input_image():
    K, D = _simple_intrinsics()
    img = np.full((480, 640, 3), 100, dtype=np.uint8)
    img_copy = img.copy()
    xyz = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]])
    overlay_math.draw_overlay(img, xyz, IDENTITY, K, D)
    assert np.array_equal(img, img_copy)


def test_all_points_behind_camera_returns_unmodified_copy():
    K, D = _simple_intrinsics()
    img = np.full((480, 640, 3), 100, dtype=np.uint8)
    xyz = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
    out = overlay_math.draw_overlay(img, xyz, IDENTITY, K, D)
    assert np.array_equal(out, img)


def test_draw_overlay_draws_something_for_a_visible_point():
    K, D = _simple_intrinsics()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    xyz = np.array([[0.0, 0.0, 2.0]])
    out = overlay_math.draw_overlay(img, xyz, IDENTITY, K, D, point_size_near_px=6, point_size_far_px=1)
    assert not np.array_equal(out, img)
    # Something was drawn at/near the principal point.
    region = out[235:246, 315:326]
    assert region.any()


def test_out_of_bounds_projection_is_not_drawn():
    K, D = _simple_intrinsics()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Far off to the side -- projects way outside a 640x480 image.
    xyz = np.array([[1000.0, 0.0, 1.0]])
    out = overlay_math.draw_overlay(img, xyz, IDENTITY, K, D)
    assert np.array_equal(out, img)


def test_color_by_intensity_requires_intensity_array():
    K, D = _simple_intrinsics()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    xyz = np.array([[0.0, 0.0, 2.0]])
    with pytest.raises(ValueError):
        overlay_math.draw_overlay(img, xyz, IDENTITY, K, D, color_by="intensity", intensity=None)


def test_near_points_get_bigger_dots_than_far_points():
    K, D = _simple_intrinsics()
    img_near = np.zeros((480, 640, 3), dtype=np.uint8)
    img_far = np.zeros((480, 640, 3), dtype=np.uint8)

    out_near = overlay_math.draw_overlay(
        img_near, np.array([[0.0, 0.0, 1.0]]), IDENTITY, K, D,
        min_range_m=1.0, max_range_m=20.0, point_size_near_px=8, point_size_far_px=1,
    )
    out_far = overlay_math.draw_overlay(
        img_far, np.array([[0.0, 0.0, 19.0]]), IDENTITY, K, D,
        min_range_m=1.0, max_range_m=20.0, point_size_near_px=8, point_size_far_px=1,
    )
    near_pixel_count = int(np.count_nonzero(out_near.sum(axis=2)))
    far_pixel_count = int(np.count_nonzero(out_far.sum(axis=2)))
    assert near_pixel_count > far_pixel_count


def test_skip_reduces_point_count_drawn():
    K, D = _simple_intrinsics()
    rng = np.random.default_rng(0)
    xyz = np.stack([rng.uniform(-1, 1, 200), rng.uniform(-1, 1, 200), np.full(200, 5.0)], axis=1)

    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    out_full = overlay_math.draw_overlay(img1, xyz, IDENTITY, K, D, skip=1)
    out_skipped = overlay_math.draw_overlay(img2, xyz, IDENTITY, K, D, skip=4)

    full_count = int(np.count_nonzero(out_full.sum(axis=2)))
    skipped_count = int(np.count_nonzero(out_skipped.sum(axis=2)))
    assert skipped_count <= full_count


def _scene(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    xyz = np.c_[rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), rng.uniform(0.5, 14, n)]
    K = np.array([[520.0, 0, 322.0], [0, 518.0, 241.0], [0, 0, 1.0]])
    return xyz, K


def test_pinhole_fast_path_matches_cv2_projection():
    xyz, K = _scene()
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([0.02, -0.03, 0.01]))[0]
    T[:3, 3] = [0.05, -0.02, 0.1]
    fast, front_f, _ = overlay_math.project_points(xyz, T, K, np.zeros(5))
    ref, front_r, _ = overlay_math.project_points(xyz, T, K, np.full(5, 1e-15))   # non-zero D takes the cv2 path
    assert np.array_equal(front_f, front_r)
    assert np.abs(fast[front_f] - ref[front_r]).max() < 1e-6


def test_disk_drawing_matches_per_point_circles():
    xyz, K = _scene()
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    out = overlay_math.draw_overlay(img, xyz, np.eye(4), K, np.zeros(5))

    # reference: the original per-point cv2.circle painter's algorithm
    pixels, in_front, cam = overlay_math.project_points(xyz, np.eye(4), K, np.zeros(5))
    ok = in_front & overlay_math.in_bounds_mask(pixels, 640, 480)
    px, cam_xyz, xyz_v = pixels[ok], cam[ok], xyz[ok]
    colors = overlay_math._turbo_colors(cam_xyz[:, 2], 0.3, 25.0)
    rng_ = np.linalg.norm(xyz_v, axis=1)
    order = np.argsort(-rng_)
    px, colors, rng_ = px[order], colors[order], rng_[order]
    sizes = np.clip(np.round(1 + np.clip((25.0 - rng_) / 24.7, 0, 1) * 5).astype(int), 1, None)
    ref = img.copy()
    u = np.clip(np.round(px[:, 0]).astype(int), 0, 639)
    v = np.clip(np.round(px[:, 1]).astype(int), 0, 479)
    ref[v[sizes <= 1], u[sizes <= 1]] = colors[sizes <= 1]
    for i in np.where(sizes > 1)[0]:
        cv2.circle(ref, (int(u[i]), int(v[i])), int(sizes[i]), tuple(int(c) for c in colors[i]), -1)
    assert np.array_equal(out, ref)
