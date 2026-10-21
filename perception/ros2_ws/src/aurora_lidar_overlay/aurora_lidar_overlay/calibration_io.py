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
YAML I/O for camera intrinsics and LiDAR->camera extrinsics.

Extrinsic schema matches the task's own spec (and CDonosoK/
ros2_camera_lidar_fusion): a single top-level `extrinsic_matrix`, 4x4,
applied as p_cam = extrinsic_matrix @ p_lidar.

Intrinsic schema matches ros2_camera_lidar_fusion's `camera_matrix` /
`distortion_coefficients` keys (so files are interchangeable with that
package) plus explicit `image_width` / `image_height`.
"""
from __future__ import annotations

import os

import numpy as np
import yaml


def load_extrinsic_matrix(path: str, invert: bool = False) -> np.ndarray:
    """
    Load the 4x4 `extrinsic_matrix` from `path`.

    Parameters
    ----------
    path : str
        Path to a YAML file with a top-level `extrinsic_matrix` key.
    invert : bool
        If True, returns the matrix inverse. Use this when the file on disk
        stores T_lidar_cam (camera->lidar solved the other way round) but
        the node needs T_cam_lidar, or vice versa.

    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No extrinsic calibration file: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "extrinsic_matrix" not in data:
        raise KeyError(f"{path} has no 'extrinsic_matrix' key")
    T = np.array(data["extrinsic_matrix"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"extrinsic_matrix in {path} is not 4x4 (got {T.shape})")
    if invert:
        T = np.linalg.inv(T)
    return T


def save_extrinsic_matrix(path: str, T, meta: dict | None = None) -> None:
    data = {"extrinsic_matrix": np.asarray(T, dtype=np.float64).tolist()}
    if meta:
        data.update(meta)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_intrinsics(path: str):
    """Return (K (3,3), D (1,N), width, height)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No intrinsic calibration file: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(1, -1)
    width = int(data.get("image_width") or data["image_size"]["width"])
    height = int(data.get("image_height") or data["image_size"]["height"])
    return K, D, width, height


def save_intrinsics(path: str, K, D, width: int, height: int, extra: dict | None = None) -> None:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    data = {
        "camera_matrix": {"rows": 3, "columns": 3, "data": K.tolist()},
        "distortion_coefficients": {"rows": 1, "columns": int(D.size), "data": D.tolist()},
        "image_width": int(width),
        "image_height": int(height),
    }
    if extra:
        data.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
