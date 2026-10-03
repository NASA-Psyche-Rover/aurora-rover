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
from sensor_msgs.msg import PointCloud2, PointField

from aurora_lidar_overlay import pointcloud_utils


def _make_cloud_xyz_only(points_xyz: np.ndarray) -> PointCloud2:
    msg = PointCloud2()
    msg.height = 1
    msg.width = points_xyz.shape[0]
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points_xyz.astype(np.float32).tobytes()
    return msg


def _make_cloud_livox_style(points_xyz, intensity, timestamp) -> PointCloud2:
    """
    Mirror livox_ros_driver2's real PointCloud2 layout.

    x,y,z,intensity (float32), then tag,line (uint8), then a float64
    timestamp -- verified against the driver's actual `LivoxPointXyzrtlt`
    struct (`src/comm/comm.h`), which is `#pragma pack(1)`: no alignment
    padding, so `timestamp` sits at a non-8-byte-aligned offset (18, not a
    compiler-padded 24). This exercises the generic offset handling against
    a real, non-obvious layout instead of a happy-path contiguous one.
    """
    n = points_xyz.shape[0]
    point_step = 26  # sizeof(LivoxPointXyzrtlt): tightly packed, no padding
    msg = PointCloud2()
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="tag", offset=16, datatype=PointField.UINT8, count=1),
        PointField(name="line", offset=17, datatype=PointField.UINT8, count=1),
        PointField(name="timestamp", offset=18, datatype=PointField.FLOAT64, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.is_dense = True

    buf = np.zeros(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
        ("tag", "u1"), ("line", "u1"), ("timestamp", "<f8"),
    ])
    assert buf.itemsize == point_step
    buf["x"] = points_xyz[:, 0]
    buf["y"] = points_xyz[:, 1]
    buf["z"] = points_xyz[:, 2]
    buf["intensity"] = intensity
    buf["timestamp"] = timestamp
    msg.data = buf.tobytes()
    return msg


def test_xyz_only_cloud_round_trips():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cloud = _make_cloud_xyz_only(pts)
    out = pointcloud_utils.pointcloud2_to_arrays(cloud, fields=("x", "y", "z"))
    assert np.allclose(out["xyz"], pts, atol=1e-5)


def test_empty_cloud_does_not_crash():
    cloud = _make_cloud_xyz_only(np.zeros((0, 3), dtype=np.float32))
    out = pointcloud_utils.pointcloud2_to_arrays(cloud, fields=("x", "y", "z", "intensity"))
    assert out["xyz"].shape == (0, 3)
    assert out["intensity"].shape == (0,)


def test_livox_style_padded_layout_with_intensity_and_timestamp():
    pts = np.array([[1.0, -1.0, 0.5], [2.0, 0.0, 1.5], [3.0, 3.0, -2.0]], dtype=np.float32)
    intensity = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    timestamp = np.array([100.0, 100.01, 100.02], dtype=np.float64)
    cloud = _make_cloud_livox_style(pts, intensity, timestamp)

    out = pointcloud_utils.pointcloud2_to_arrays(cloud, fields=("x", "y", "z", "intensity", "timestamp"))
    assert np.allclose(out["xyz"], pts, atol=1e-4)
    assert np.allclose(out["intensity"], intensity, atol=1e-4)
    assert np.allclose(out["timestamp"], timestamp, atol=1e-9)


def test_skip_downsamples():
    pts = np.arange(30, dtype=np.float32).reshape(10, 3)
    cloud = _make_cloud_xyz_only(pts)
    out = pointcloud_utils.pointcloud2_to_arrays(cloud, skip=3)
    assert out["xyz"].shape[0] == len(range(0, 10, 3))


def test_finite_mask_drops_nan_and_inf():
    xyz = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, np.inf, 0.0], [1.0, 2.0, 3.0]])
    mask = pointcloud_utils.finite_mask(xyz)
    assert list(mask) == [True, False, False, True]
