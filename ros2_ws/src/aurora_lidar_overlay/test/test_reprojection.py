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

import numpy as np

from aurora_lidar_overlay import reprojection


def _simple_intrinsics():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5)
    return K, D


def test_solve_extrinsic_pnp_recovers_known_identity_transform():
    K, D = _simple_intrinsics()
    pts_3d = np.array(
        [[0.0, 0.0, 2.0], [0.5, 0.0, 2.0], [0.0, 0.5, 2.0], [-0.5, -0.5, 3.0], [0.3, -0.2, 4.0]]
    )
    import cv2

    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, D)
    pts_2d = pts_2d.reshape(-1, 2)

    T, rvec_out, tvec_out, _ = reprojection.solve_extrinsic_pnp(pts_3d, pts_2d, K, D)
    assert np.allclose(T, np.eye(4), atol=1e-4)

    err, rms = reprojection.reprojection_errors(pts_3d, pts_2d, rvec_out, tvec_out, K, D)
    assert rms < 1e-3


def test_reprojection_errors_from_matrix_matches_rvec_tvec_path():
    K, D = _simple_intrinsics()
    pts_3d = np.array([[0.1, 0.0, 2.0], [0.0, 0.1, 3.0], [-0.1, -0.1, 4.0], [0.2, 0.2, 5.0]])
    pts_2d = np.array([[330.0, 240.0], [320.0, 257.0], [297.0, 215.0], [340.0, 260.0]])

    T, rvec, tvec, _ = reprojection.solve_extrinsic_pnp(pts_3d, pts_2d, K, D)
    err_a, rms_a = reprojection.reprojection_errors(pts_3d, pts_2d, rvec, tvec, K, D)
    err_b, rms_b = reprojection.reprojection_errors_from_matrix(pts_3d, pts_2d, T, K, D)
    assert np.allclose(err_a, err_b, atol=1e-6)
    assert abs(rms_a - rms_b) < 1e-6


def test_load_correspondences_parses_and_skips_comments(tmp_path):
    path = tmp_path / "corr.txt"
    path.write_text("# u, v, x, y, z\n1.0,2.0,3.0,4.0,5.0\n\n6,7,8,9,10\n")
    pts_2d, pts_3d = reprojection.load_correspondences(str(path))
    assert pts_2d.shape == (2, 2)
    assert pts_3d.shape == (2, 3)
    assert np.allclose(pts_2d[0], [1.0, 2.0])
    assert np.allclose(pts_3d[1], [8.0, 9.0, 10.0])


def test_solve_extrinsic_pnp_requires_at_least_four_points():
    import pytest

    K, D = _simple_intrinsics()
    pts_3d = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]])
    pts_2d = np.array([[320.0, 240.0], [345.0, 240.0]])
    with pytest.raises(ValueError):
        reprojection.solve_extrinsic_pnp(pts_3d, pts_2d, K, D)
