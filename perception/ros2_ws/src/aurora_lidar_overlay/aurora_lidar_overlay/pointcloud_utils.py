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
Vectorized PointCloud2 <-> numpy conversion.

Reads each field's actual (name, offset, datatype, count) from the message
instead of assuming a fixed x,y,z-then-junk layout, so it works unmodified
against any publisher's field layout -- e.g. livox_ros_driver2's PointCloud2
mode (x, y, z, intensity, tag, line, timestamp, packed with no padding and
a timestamp offset that isn't a fixed multiple of 4) as well as the
simpler x,y,z-only clouds other tools emit.
"""
from __future__ import annotations

import numpy as np

# sensor_msgs/PointField datatype constants (avoids importing the ROS
# message type here, keeping this module import-light / testable without a
# full ROS environment).
_DATATYPE_TO_NUMPY = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def _empty_result(fields):
    out = {"xyz": np.zeros((0, 3), dtype=np.float64)}
    for f in fields:
        if f not in ("x", "y", "z"):
            out[f] = np.zeros((0,), dtype=np.float64)
    return out


def pointcloud2_to_arrays(cloud_msg, fields=("x", "y", "z"), skip=1):
    """
    Extract named fields from a PointCloud2 as numpy arrays.

    Parameters
    ----------
    cloud_msg : PointCloud2
        A sensor_msgs/PointCloud2 (duck-typed: anything with .height,
        .width, .point_step, .is_bigendian, .fields, .data).
    fields : tuple of str
        Field names to extract in addition to x/y/z (which are always
        combined into the "xyz" key when all three are present).
    skip : int
        Keep every `skip`-th point.

    Returns
    -------
    dict
        Key "xyz" -> (N,3) float64, plus one (N,) float64 array per
        requested non-xyz field that's actually present in the message.
        Never raises on an empty/degenerate cloud -- returns empty arrays.

    """
    if cloud_msg.height == 0 or cloud_msg.width == 0 or cloud_msg.point_step == 0:
        return _empty_result(fields)

    field_by_name = {f.name: f for f in cloud_msg.fields}
    if not field_by_name:
        return _empty_result(fields)

    endian = ">" if cloud_msg.is_bigendian else "<"
    ordered = sorted(field_by_name.values(), key=lambda f: f.offset)

    dtype_spec = []
    pos = 0
    names_seen = set()
    name_map = {}
    for f in ordered:
        if f.offset < pos:
            # Overlapping field layout -- can't build a simple structured
            # dtype from it safely, bail out rather than silently misread.
            raise ValueError(
                f"PointField '{f.name}' at offset {f.offset} overlaps a "
                f"previous field (current position {pos})"
            )
        if f.offset > pos:
            dtype_spec.append((f"__pad{pos}", f"{endian}V{f.offset - pos}"))
        np_type = _DATATYPE_TO_NUMPY.get(f.datatype)
        if np_type is None:
            raise ValueError(f"Unsupported PointField datatype {f.datatype} ('{f.name}')")
        count = max(f.count, 1)
        elem_size = np.dtype(np_type).itemsize
        safe_name = f.name if f.name not in names_seen else f"{f.name}__{f.offset}"
        names_seen.add(safe_name)
        name_map[f.name] = safe_name
        type_char = np.dtype(np_type).char
        if count == 1:
            dtype_spec.append((safe_name, f"{endian}{type_char}"))
        else:
            dtype_spec.append((safe_name, f"{endian}{type_char}", (count,)))
        pos = f.offset + elem_size * count

    if cloud_msg.point_step > pos:
        dtype_spec.append((f"__pad{pos}", f"{endian}V{cloud_msg.point_step - pos}"))
    elif cloud_msg.point_step < pos:
        raise ValueError(
            f"PointCloud2 point_step ({cloud_msg.point_step}) is smaller than the "
            f"declared fields require ({pos} bytes)"
        )

    struct_dtype = np.dtype(dtype_spec)
    n_points = cloud_msg.width * cloud_msg.height
    raw = np.frombuffer(cloud_msg.data, dtype=struct_dtype, count=n_points)
    if skip > 1:
        raw = raw[::skip]

    out = {}
    have_xyz = all(k in field_by_name for k in ("x", "y", "z"))
    if have_xyz:
        out["xyz"] = np.stack(
            [
                raw[name_map["x"]].astype(np.float64),
                raw[name_map["y"]].astype(np.float64),
                raw[name_map["z"]].astype(np.float64),
            ],
            axis=1,
        )
    else:
        out["xyz"] = np.zeros((raw.shape[0], 3), dtype=np.float64)

    for f in fields:
        if f in ("x", "y", "z"):
            continue
        if f in field_by_name:
            out[f] = raw[name_map[f]].astype(np.float64)

    return out


def finite_mask(xyz):
    """(N,) bool mask of rows in an (N,3) array with no NaN/inf."""
    return np.all(np.isfinite(xyz), axis=1)
