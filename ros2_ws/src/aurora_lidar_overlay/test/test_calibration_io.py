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

from aurora_lidar_overlay import calibration_io


def test_extrinsic_round_trip(tmp_path):
    path = str(tmp_path / "extrinsics.yaml")
    T = np.array(
        [[0.0, -1.0, 0.0, 0.1], [0.0, 0.0, -1.0, 0.2], [1.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    )
    calibration_io.save_extrinsic_matrix(path, T, meta={"note": "test"})
    loaded = calibration_io.load_extrinsic_matrix(path)
    assert np.allclose(loaded, T)


def test_extrinsic_invert_round_trip(tmp_path):
    path = str(tmp_path / "extrinsics.yaml")
    T = np.array(
        [[0.0, -1.0, 0.0, 0.1], [0.0, 0.0, -1.0, 0.2], [1.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    )
    calibration_io.save_extrinsic_matrix(path, T)
    loaded_inv = calibration_io.load_extrinsic_matrix(path, invert=True)
    assert np.allclose(loaded_inv, np.linalg.inv(T))


def test_intrinsics_round_trip(tmp_path):
    path = str(tmp_path / "intrinsics.yaml")
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.array([0.1, -0.05, 0.0, 0.0, 0.0])
    calibration_io.save_intrinsics(path, K, D, 640, 480)
    K2, D2, w, h = calibration_io.load_intrinsics(path)
    assert np.allclose(K2, K)
    assert np.allclose(D2.reshape(-1), D)
    assert (w, h) == (640, 480)


def test_missing_extrinsic_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        calibration_io.load_extrinsic_matrix(str(tmp_path / "nope.yaml"))
