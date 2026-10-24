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
Pure, ROS-free math for the LiDAR-on-camera overlay.

Projects a LiDAR point cloud onto a camera image and renders it
Waymo-style: dense, depth-colored dots that line up with the scene.

Kept free of rclpy/message types on purpose so it can be unit-tested without
ROS running and reused by the calibration/reprojection-check scripts and by
the live overlay node alike.

Convention (matches ros2_camera_lidar_fusion and the task's extrinsic YAML):
    p_cam = T_cam_lidar @ p_lidar   (T_cam_lidar is 4x4, homogeneous)
"""
from __future__ import annotations

import cv2
import numpy as np

# A point this close to (or behind) the camera's lens plane is treated as
# "behind the camera" -- cv2.projectPoints will happily project it to a
# bogus pixel instead of erroring, so this check has to happen ourselves.
MIN_FORWARD_DEPTH_M = 0.05


def project_points(xyz_lidar, T_cam_lidar, K, D):
    """
    Project LiDAR-frame points into pixel coordinates.

    Parameters
    ----------
    xyz_lidar : array
        (N,3) array, LiDAR-frame points in meters.
    T_cam_lidar : array
        (4,4) rigid transform, p_cam = T_cam_lidar @ p_lidar.
    K : array
        (3,3) camera intrinsic matrix.
    D : array
        Camera distortion coefficients (any shape OpenCV accepts).

    Returns
    -------
    pixels : array
        (N,2) float64 pixel (u, v). Undefined (but finite) for rows where
        in_front is False -- check the mask before using them.
    in_front : array
        (N,) bool, True where the point is far enough in front of the
        camera to be meaningfully projectable.
    cam_xyz : array
        (N,3) the same points expressed in the camera frame (camera forward
        is +Z; used as "depth" for coloring/draw order).

    """
    xyz_lidar = np.asarray(xyz_lidar, dtype=np.float64).reshape(-1, 3)
    n = xyz_lidar.shape[0]
    if n == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=bool),
            np.zeros((0, 3), dtype=np.float64),
        )

    T = np.asarray(T_cam_lidar, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]

    cam_xyz = xyz_lidar @ R.T + t
    in_front = cam_xyz[:, 2] > MIN_FORWARD_DEPTH_M

    pixels = np.zeros((n, 2), dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if in_front.any() and not np.any(D):
        # No distortion: cv2.projectPoints reduces to the pinhole model, which numpy
        # does several times faster on a dense cloud.
        p = cam_xyz[in_front]
        pixels[in_front] = np.stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2],
                                     K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], axis=1)
    elif in_front.any():
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        projected, _ = cv2.projectPoints(
            xyz_lidar[in_front],
            rvec,
            tvec,
            K,
            np.asarray(D, dtype=np.float64),
        )
        pixels[in_front] = projected.reshape(-1, 2)

    return pixels, in_front, cam_xyz


def in_bounds_mask(pixels, width, height):
    return (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )


def _turbo_colors(values, vmin, vmax):
    """Map scalar values to BGR uint8 colors via OpenCV's turbo colormap."""
    values = np.asarray(values, dtype=np.float64)
    span = max(vmax - vmin, 1e-6)
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    colors = cv2.applyColorMap(norm_u8.reshape(-1, 1), cv2.COLORMAP_TURBO)
    return colors.reshape(-1, 3)  # (N,3) BGR


def draw_overlay(
    img,
    xyz,
    T_cam_lidar,
    K,
    D,
    *,
    intensity=None,
    color_by="depth",
    max_range_m=25.0,
    min_range_m=0.3,
    point_size_near_px=6,
    point_size_far_px=1,
    skip=1,
):
    """
    Draw a Waymo-style depth-colored LiDAR overlay on a copy of `img`.

    Never mutates `img`; always returns a new array, even for an empty or
    all-invalid point cloud (in which case it's just `img.copy()`).

    Parameters
    ----------
    img : array
        HxWx3 uint8 BGR image.
    xyz : array
        (N,3) LiDAR-frame points, meters. May be empty.
    T_cam_lidar : array
        (4,4), p_cam = T_cam_lidar @ p_lidar.
    K : array
        (3,3) camera intrinsic matrix.
    D : array
        Camera distortion coefficients (OpenCV convention).
    intensity : array, optional
        (N,) array, required if color_by="intensity".
    color_by : str
        "depth" (camera-frame forward distance, default), "range"
        (Euclidean distance from the LiDAR origin), "intensity", or
        "height" (LiDAR-frame Z).
    max_range_m : float
        Upper end of the color/size normalization range, meters.
    min_range_m : float
        Lower end of the color/size normalization range, meters.
    point_size_near_px : int
        Dot radius (px) at min_range_m.
    point_size_far_px : int
        Dot radius (px) at max_range_m; linearly interpolated between the
        near/far sizes in between. Near points are drawn last (on top of
        far points).
    skip : int
        Keep every `skip`-th point (after any upstream downsampling) -- a
        cheap FPS knob for dense clouds.

    """
    out = img.copy()
    h, w = out.shape[:2]

    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3) if np.size(xyz) else np.zeros((0, 3))
    if xyz.shape[0] == 0:
        return out

    if skip > 1:
        xyz = xyz[::skip]
        if intensity is not None:
            intensity = np.asarray(intensity).reshape(-1)[::skip]

    pixels, in_front, cam_xyz = project_points(xyz, T_cam_lidar, K, D)
    valid = in_front & in_bounds_mask(pixels, w, h)
    if not valid.any():
        return out

    pixels = pixels[valid]
    cam_xyz = cam_xyz[valid]
    xyz_v = xyz[valid]
    depth = cam_xyz[:, 2]
    rng = np.linalg.norm(xyz_v, axis=1)

    if color_by == "intensity":
        if intensity is None:
            raise ValueError("color_by='intensity' requires an intensity array")
        metric = np.asarray(intensity, dtype=np.float64).reshape(-1)[valid]
        vmin, vmax = float(np.min(metric)), float(np.max(metric))
    elif color_by == "height":
        metric = xyz_v[:, 2]
        vmin, vmax = float(np.min(metric)), float(np.max(metric))
    elif color_by == "range":
        metric, vmin, vmax = rng, min_range_m, max_range_m
    else:  # "depth" (default)
        metric, vmin, vmax = depth, min_range_m, max_range_m

    colors = _turbo_colors(metric, vmin, vmax)

    # Painter's algorithm: farthest first, so nearer dots draw on top.
    order = np.argsort(-rng)
    pixels = pixels[order]
    colors = colors[order]
    rng_sorted = rng[order]

    size_span = max(max_range_m - min_range_m, 1e-6)
    size_scale = np.clip((max_range_m - rng_sorted) / size_span, 0.0, 1.0)
    sizes = np.round(
        point_size_far_px + size_scale * (point_size_near_px - point_size_far_px)
    ).astype(np.int32)
    sizes = np.clip(sizes, 1, None)

    u = np.round(pixels[:, 0]).astype(np.int32)
    v = np.round(pixels[:, 1]).astype(np.int32)
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    # Vectorized fast path: 1px dots (typically the bulk of far-away points)
    # are written in one fancy-index assignment instead of a Python loop.
    # Fancy-index writes with duplicate (v,u) resolve to the *last* entry in
    # iteration order, and since `order` is far-to-near, that's the nearer
    # duplicate -- so this still respects the painter's-algorithm ordering.
    tiny = sizes <= 1
    if tiny.any():
        out[v[tiny], u[tiny]] = colors[tiny]

    _draw_disks(out, u, v, sizes, colors, ~tiny)
    return out


_DISK_KERNELS = {}


def _draw_disks(out, u, v, sizes, colors, big):
    """
    Draw filled disks of radius `sizes` (px) in painter's order, with no per-point loop.

    Points arrive far-to-near and size grows as range shrinks, so points of one size are
    contiguous. For each size, point indices go into a layer and a disk-shaped max filter
    spreads them; each pixel then holds the highest index within the radius, i.e. the
    nearest point, which is the pixel the painter's algorithm would leave. Groups are
    applied far-to-near so nearer sizes overwrite farther ones.

    """
    big = np.flatnonzero(big)
    if big.size == 0:
        return
    h, w = out.shape[:2]
    sz = sizes[big]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sz)) + 1, [big.size]])
    best = np.zeros((h, w), np.float32)          # per pixel: index of the nearest point covering it
    for a, b in zip(starts[:-1], starts[1:]):
        r = int(sz[a])
        idx = big[a:b]
        gu, gv = u[idx], v[idx]
        x0, x1 = max(int(gu.min()) - r, 0), min(int(gu.max()) + r + 1, w)
        y0, y1 = max(int(gv.min()) - r, 0), min(int(gv.max()) + r + 1, h)
        layer = np.zeros((y1 - y0, x1 - x0), np.float32)
        layer[gv - y0, gu - x0] = idx + 1          # float32 holds integers exactly up to 2**24
        kernel = _DISK_KERNELS.get(r)
        if kernel is None:
            kernel = _DISK_KERNELS[r] = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
            cv2.circle(kernel, (r, r), r, 1, -1)    # same footprint cv2.circle would draw
        roi = best[y0:y1, x0:x1]
        np.maximum(roi, cv2.dilate(layer, kernel), out=roi)
    painted = best > 0
    out[painted] = colors[best[painted].astype(np.int64) - 1]
