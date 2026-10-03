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

from aurora_lidar_overlay import tf_utils


def test_quaternion_identity_gives_identity_matrix():
    R = tf_utils.quaternion_to_matrix(0.0, 0.0, 0.0, 1.0)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_matrix_quaternion_round_trip():
    # 90 deg about Z
    theta = np.pi / 2
    R = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    q = tf_utils.matrix_to_quaternion(R)
    R2 = tf_utils.quaternion_to_matrix(*q)
    assert np.allclose(R, R2, atol=1e-6)


def test_slerp_endpoints():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = tf_utils.matrix_to_quaternion(
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    out = tf_utils.slerp(q0, q1, np.array([0.0, 1.0]))
    assert np.allclose(out[0], q0, atol=1e-6) or np.allclose(out[0], -q0, atol=1e-6)
    assert np.allclose(out[1], q1, atol=1e-6) or np.allclose(out[1], -q1, atol=1e-6)


def test_interpolate_transforms_midpoint_translation():
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, 3] = [2.0, 4.0, 6.0]
    out = tf_utils.interpolate_transforms(T0, T1, np.array([0.5]))
    assert np.allclose(out[0, :3, 3], [1.0, 2.0, 3.0], atol=1e-6)


def test_apply_transforms_batch():
    n = 3
    transforms = np.tile(np.eye(4), (n, 1, 1))
    transforms[:, 0, 3] = [1.0, 2.0, 3.0]
    points = np.zeros((n, 3))
    out = tf_utils.apply_transforms(transforms, points)
    assert np.allclose(out[:, 0], [1.0, 2.0, 3.0])
