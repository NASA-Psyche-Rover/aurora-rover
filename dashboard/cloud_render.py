"""
Software renderer for the dashboard's LiDAR point-cloud panel.

Takes recent Livox points (LiDAR frame: x forward, y left, z up, metres) and
draws them from a manually steered orbit camera into a BGR image. Pure numpy +
OpenCV, no ROS and no GPU, so it runs the same on the Jetson and in tests.

  CloudAccumulator   rolling window of recent frames -> one (N, 3) array,
                     so a static scene visibly fills in while a moving one
                     shows only its recent past
  CloudRenderer      orbit camera (yaw / pitch / zoom, moves only when steered), height colouring,
                     distance fade, ground rings, sensor marker
"""
import threading
import time
from collections import deque

import cv2
import numpy as np


def _height_lut():
    """256-entry BGR colour ramp (cyan -> green -> yellow -> orange) used for
    height. Built from cv2's TURBO map with the dark blue end dropped, since
    that end vanishes against the dark background."""
    ramp = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    full = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO).reshape(-1, 3)
    lut = full[int(0.12 * 255):int(0.92 * 255) + 1]
    idx = np.linspace(0, len(lut) - 1, 256).astype(int)
    return lut[idx]


HEIGHT_LUT = _height_lut()

FADE_LEVELS, FADE_MIN = 64, 0.08    # darkest a point can fade to (far + about to expire)
AGE_FADE_S = 2.5                     # points dim out over their last 2.5 s in the window


def _colour_table():
    """(256 heights x FADE_LEVELS fades) -> BGRA packed in a uint32, so a point's
    colour is one gather and the image is written one 32-bit word per point."""
    fade = np.linspace(FADE_MIN, 1.0, FADE_LEVELS)
    bgr = (HEIGHT_LUT[:, None, :].astype(np.uint16) * (fade * 256).astype(np.uint16)[None, :, None] >> 8).astype(np.uint8)
    bgra = np.concatenate([bgr, np.full((256, FADE_LEVELS, 1), 255, np.uint8)], axis=2)
    return np.ascontiguousarray(bgra).view(np.uint32).reshape(256 * FADE_LEVELS)


COLOUR_TABLE = _colour_table()


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], np.float32)


class CloudAccumulator:
    """Thread-safe rolling window of point-cloud frames.

    `add` is called from the ROS callback, `points` from the render thread.
    Old frames are dropped by age; if the window still holds more than
    `max_points`, a random subset is returned (rendering cost stays bounded
    and the subset keeps the cloud's overall shape)."""

    def __init__(self, window_s=8.0, max_points=150_000, max_range_m=40.0):
        self.window_s, self.max_points, self.max_range_m = window_s, max_points, max_range_m
        self._frames = deque()    # (t, (n, 3) float32)
        self._lock = threading.Lock()
        self._k = 1               # current subsampling stride; only changes when the point count drifts far from the budget
        self._arrivals = deque(maxlen=50)   # (t, n_points) for rate metrics
        self.version = 0          # bumped on every add, so the renderer can skip unchanged frames

    def add(self, xyz, t=None):
        t = time.time() if t is None else t
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        r2 = np.einsum("ij,ij->i", xyz, xyz)     # squared range; NaN / inf fail both comparisons
        xyz = xyz[(r2 > 0.05 ** 2) & (r2 < self.max_range_m ** 2)]
        with self._lock:
            self._frames.append((t, xyz))
            self._arrivals.append((t, len(xyz)))
            self.version += 1
            self._expire(t)

    def _expire(self, now):
        while self._frames and now - self._frames[0][0] > self.window_s:
            self._frames.popleft()

    def _stride(self, total, commit=False):
        """Subsampling stride for `total` points. It is kept (hysteresis) until the count
        drifts far from the budget, so the same points are drawn from one render to the
        next instead of a new random subset each time, which made static scenes shimmer."""
        need = -(-total // self.max_points)
        k = self._k
        if need > k or total < 0.7 * self.max_points * k:
            k = max(need, 1)
        if commit:
            self._k = k
        return k

    def points(self, now=None, with_age=False):
        """Recent points as (N, 3) float32; with_age=True also returns each point's age
        in seconds (N,) so the renderer can fade old points out smoothly."""
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            frames = list(self._frames)
        if not frames:
            empty = np.zeros((0, 3), np.float32)
            return (empty, np.zeros(0, np.float32)) if with_age else empty
        k = self._stride(sum(len(f) for _, f in frames), commit=True)
        parts = [f[::k] for _, f in frames]     # always the same points of a given frame
        pts = np.concatenate(parts)
        if not with_age:
            return pts
        age = np.repeat(np.array([now - t for t, _ in frames], np.float32), [len(p) for p in parts])
        return pts, age

    def n_points(self, now=None):
        """How many points `points()` would return, without building the array."""
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            sizes = [len(f) for _, f in self._frames]
        k = self._stride(sum(sizes))
        return sum(-(-n // k) for n in sizes)

    def rates(self, now=None):
        """(messages per second, points per second) over the recent arrivals."""
        now = time.time() if now is None else now
        with self._lock:
            recent = [(t, n) for t, n in self._arrivals if now - t < 3.0]
        if len(recent) < 2:
            return 0.0, 0.0
        span = recent[-1][0] - recent[0][0]
        if span <= 0:
            return 0.0, 0.0
        return (len(recent) - 1) / span, sum(n for _, n in recent[1:]) / span

    def age_s(self, now=None):
        now = time.time() if now is None else now
        with self._lock:
            return now - self._frames[-1][0] if self._frames else None


class CloudRenderer:
    """Orbit camera looking at a point just above the sensor.

    yaw/pitch are degrees (yaw 0 = looking along the LiDAR's +x from behind
    it); zoom scales the camera distance. The view state is plain attributes
    so the web layer can nudge it from request threads."""

    HOME = (200.0, 24.0, 7.0)    # yaw, pitch, distance

    def __init__(self, size=(960, 720), fov_deg=55.0):
        self.w, self.h = size
        self.f = 0.5 * self.w / np.tan(np.radians(fov_deg) / 2)
        self.yaw, self.pitch, self.dist = self.HOME
        self.target = np.array([2.0, 0.0, 0.2], np.float32)
        self.view_version = 0     # bumped when the camera moves
        self._ground_z = -0.3
        self.bg = np.array([22, 18, 14], np.uint8)   # BGR, near-black blue
        self._canvas = np.empty((self.h, self.w, 4), np.uint8)   # BGRA background, copied per frame
        self._canvas[:] = (*self.bg, 255)
        ang = np.linspace(0, 2 * np.pi, 120, endpoint=False, dtype=np.float32)
        self._rings = np.stack([np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(ang)], axis=1)
                                for r in (1, 2, 4, 8)])         # (4, 120, 3), z set to the ground per frame

    # ---- camera ----
    def _basis(self):
        yaw, pit = np.radians(self.yaw), np.radians(self.pitch)
        eye_dir = np.array([np.cos(pit) * np.cos(yaw), np.cos(pit) * np.sin(yaw), np.sin(pit)], np.float32)
        eye = self.target + eye_dir * self.dist
        fwd = -eye_dir
        right = _cross(fwd, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right) + 1e-9
        up = _cross(right, fwd)
        return eye, right, up, fwd

    def _project(self, pts):
        """(N, 3) world points -> (u, v, depth) with depth along the view axis."""
        eye, right, up, fwd = self._basis()
        # one (N,3)x(3,3) product instead of three; subtracting the eye after
        # projecting avoids allocating pts - eye
        x, y, z = (pts @ np.stack([right, up, fwd], axis=1) - np.stack([right, up, fwd]) @ eye).T
        zs = np.maximum(z, 1e-3)
        u = self.w / 2 + self.f * x / zs
        v = self.h / 2 - self.f * y / zs
        return u, v, z

    def rotate(self, dyaw, dpitch):
        self.view_version += 1
        self.yaw = (self.yaw + dyaw) % 360.0
        self.pitch = float(np.clip(self.pitch + dpitch, -5.0, 85.0))

    def zoom(self, factor):
        self.view_version += 1
        self.dist = float(np.clip(self.dist * factor, 1.5, 30.0))

    def reset(self):
        self.view_version += 1
        self.yaw, self.pitch, self.dist = self.HOME

    # ---- drawing ----
    def _draw_rings(self, img):
        rings = self._rings.copy()
        rings[:, :, 2] = self._ground_z
        u, v, z = self._project(rings.reshape(-1, 3))   # all four rings in one projection
        poly_all = np.stack([u, v], axis=1).reshape(4, 120, 2)
        ok_all = (z > 0.1).reshape(4, 120)
        for poly_r, ok in zip(poly_all, ok_all):
            poly = poly_r[ok].astype(np.int32)
            if len(poly) > 2:
                cv2.polylines(img, [poly.reshape(-1, 1, 2)], len(poly) == 120, (60, 52, 44, 255), 1, cv2.LINE_AA)
        # forward axis tick so "which way is the rover facing" is visible
        tip = np.array([[0, 0, self._ground_z], [1.2, 0, self._ground_z]], np.float32)
        u, v, z = self._project(tip)
        if (z > 0.1).all():
            cv2.arrowedLine(img, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])), (90, 180, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)

    def _marker(self, img):
        """sensor marker at the origin"""
        o = self._project(np.array([[0, 0, 0]], np.float32))
        if o[2][0] > 0.1:
            cv2.circle(img, (int(o[0][0]), int(o[1][0])), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(o[0][0]), int(o[1][0])), 10, (90, 180, 255), 2, cv2.LINE_AA)

    def render(self, pts, age=None, window_s=8.0):
        # BGRA so each point is a single 32-bit write; converted to BGR at the end
        img = self._canvas.copy()
        if len(pts):
            low = float(np.percentile(pts[::8, 2], 2))
            self._ground_z = 0.9 * self._ground_z + 0.1 * low
        self._draw_rings(img)
        if len(pts) == 0:
            return self._finish(img, dilate=False)

        u, v, z = self._project(pts)
        ok = (z > 0.2) & (u >= 0) & (u < self.w - 1) & (v >= 0) & (v < self.h - 1)
        if not ok.any():
            return self._finish(img, dilate=False)
        u, v, z, pz = u[ok].astype(np.int32), v[ok].astype(np.int32), z[ok], pts[ok, 2]
        age = None if age is None else age[ok]

        # colour by height relative to the ground estimate, fade with distance from the camera:
        # both are quantised and looked up together in one table
        h_idx = np.clip((pz - self._ground_z) * (255 / 2.5), 0, 255).astype(np.int32)
        fade = np.clip(1.25 - (z - self.dist) / (self.dist * 2.2 + 4), 0.35, 1.0)
        if age is not None:     # old points dim out gradually instead of vanishing when their frame expires
            fade *= np.clip((window_s - age) * (1.0 / AGE_FADE_S), 0.0, 1.0)
        f_idx = np.clip((fade - FADE_MIN) * ((FADE_LEVELS - 1) / (1.0 - FADE_MIN)), 0, FADE_LEVELS - 1).astype(np.int32)
        pix = COLOUR_TABLE[h_idx * FADE_LEVELS + f_idx]

        # far points first so near points win the pixel
        # (depth quantised to uint16 so numpy uses its radix sort, ~8x faster than a float argsort)
        dq = np.clip(z * (65535.0 / 60.0), 0, 65535).astype(np.uint16)
        order = np.argsort(65535 - dq, kind="stable")
        img.view(np.uint32).reshape(-1)[(v * self.w + u)[order]] = pix[order]
        return self._finish(img, dilate=True)

    def _finish(self, img, dilate):
        if dilate:
            # grow each point to ~2x2 px so the cloud reads as surfaces, not dust
            img = cv2.dilate(img, np.ones((2, 2), np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if dilate:
            self._marker(img)
        return img
