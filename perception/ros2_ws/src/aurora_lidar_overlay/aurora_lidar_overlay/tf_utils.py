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
Small, dependency-light TF2-to-matrix and rigid-transform helpers.

Turns TF2 transforms into 4x4 matrices and interpolates rigid transforms --
used by the overlay node for both the optional TF2 extrinsic path and
per-point motion deskew.

Deliberately does not depend on tf_transformations (not guaranteed to be
installed) -- quaternion math is ~10 lines and easy to unit test on its own.
"""
from __future__ import annotations

import numpy as np


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """ROS (x,y,z,w) quaternion -> 3x3 rotation matrix."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform_msg_to_matrix(translation, rotation) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform matrix from translation + rotation.

    Takes a geometry_msgs Vector3/Point + Quaternion (or anything with
    matching .x/.y/.z[/.w] attributes).
    """
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
    T[:3, 3] = [translation.x, translation.y, translation.z]
    return T


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Spherically interpolate between two (x,y,z,w) quaternions.

    Parameters
    ----------
    q0 : array
        (4,) start quaternion.
    q1 : array
        (4,) end quaternion.
    t : array
        (N,) interpolation fractions in [0, 1].

    Returns
    -------
    array
        (N,4) interpolated, normalized quaternions.

    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    n = t.shape[0]
    out = np.zeros((n, 4), dtype=np.float64)

    if dot > 0.9995:
        # Nearly identical rotations: linear interpolation is numerically
        # safer than sin(theta)-based slerp when theta ~ 0.
        out = q0[None, :] + t[:, None] * (q1 - q0)[None, :]
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        out = s0[:, None] * q0[None, :] + s1[:, None] * q1[None, :]

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return out / norms


def matrix_to_quaternion(R: np.ndarray):
    """Convert a 3x3 rotation matrix to an (x,y,z,w) quaternion."""
    m = np.asarray(R, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def interpolate_transforms(T0: np.ndarray, T1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Interpolate rigid transforms between T0 (t=0) and T1 (t=1).

    Parameters
    ----------
    T0 : array
        (4,4) homogeneous transform at t=0.
    T1 : array
        (4,4) homogeneous transform at t=1.
    t : array
        (N,) fractions in [0, 1] (per-point normalized time).

    Returns
    -------
    array
        (N,4,4) interpolated transforms.

    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    q0 = matrix_to_quaternion(T0[:3, :3])
    q1 = matrix_to_quaternion(T1[:3, :3])
    q_interp = slerp(q0, q1, t)  # (N,4)

    n = t.shape[0]
    out = np.tile(np.eye(4), (n, 1, 1))
    p0, p1 = T0[:3, 3], T1[:3, 3]
    out[:, :3, 3] = p0[None, :] + t[:, None] * (p1 - p0)[None, :]

    x, y, z, w = q_interp[:, 0], q_interp[:, 1], q_interp[:, 2], q_interp[:, 3]
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def apply_transforms(transforms: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a per-point (N,4,4) transform batch to (N,3) points."""
    R = transforms[:, :3, :3]
    t = transforms[:, :3, 3]
    return np.einsum("nij,nj->ni", R, points) + t
