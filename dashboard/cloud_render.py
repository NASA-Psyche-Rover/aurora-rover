"""
Software renderer for the dashboard's LiDAR point-cloud panel.

Takes recent Livox points (LiDAR frame: x forward, y left, z up, metres) and
draws them from an orbiting virtual camera into a BGR image. Pure numpy +
OpenCV, no ROS and no GPU, so it runs the same on the Jetson and in tests.

  CloudAccumulator   rolling window of recent frames -> one (N, 3) array,
                     so a static scene visibly fills in while a moving one
                     shows only its recent past
  CloudRenderer      orbit camera (yaw / pitch / zoom), height colouring,
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
        self._rng = np.random.default_rng(0)
        self._arrivals = deque(maxlen=50)   # (t, n_points) for rate metrics

    def add(self, xyz, t=None):
        t = time.time() if t is None else t
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[np.isfinite(xyz).all(axis=1) & (r > 0.05) & (r < self.max_range_m)]
        with self._lock:
            self._frames.append((t, xyz))
            self._arrivals.append((t, len(xyz)))
            self._expire(t)

    def _expire(self, now):
        while self._frames and now - self._frames[0][0] > self.window_s:
            self._frames.popleft()

    def points(self, now=None):
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            frames = [f for _, f in self._frames]
        if not frames:
            return np.zeros((0, 3), np.float32)
        pts = np.concatenate(frames)
        if len(pts) > self.max_points:
            pts = pts[self._rng.choice(len(pts), self.max_points, replace=False)]
        return pts

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

    def __init__(self, size=(960, 720), fov_deg=55.0):
        self.w, self.h = size
        self.f = 0.5 * self.w / np.tan(np.radians(fov_deg) / 2)
        self.yaw, self.pitch, self.dist = 200.0, 24.0, 7.0
        self.target = np.array([2.0, 0.0, 0.2], np.float32)
        self.auto_orbit = True
        self._ground_z = -0.3
        self.bg = np.array([22, 18, 14], np.uint8)   # BGR, near-black blue

    # ---- camera ----
    def _basis(self):
        yaw, pit = np.radians(self.yaw), np.radians(self.pitch)
        eye_dir = np.array([np.cos(pit) * np.cos(yaw), np.cos(pit) * np.sin(yaw), np.sin(pit)], np.float32)
        eye = self.target + eye_dir * self.dist
        fwd = -eye_dir
        right = np.cross(fwd, [0, 0, 1.0]).astype(np.float32)
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(right, fwd).astype(np.float32)
        return eye, right, up, fwd

    def _project(self, pts):
        """(N, 3) world points -> (u, v, depth) with depth along the view axis."""
        eye, right, up, fwd = self._basis()
        rel = pts - eye
        z = rel @ fwd
        x = rel @ right
        y = rel @ up
        zs = np.maximum(z, 1e-3)
        u = self.w / 2 + self.f * x / zs
        v = self.h / 2 - self.f * y / zs
        return u, v, z

    def step_orbit(self, dt, deg_per_s=9.0):
        if self.auto_orbit:
            self.yaw = (self.yaw + deg_per_s * dt) % 360.0

    def rotate(self, dyaw, dpitch):
        self.auto_orbit = False
        self.yaw = (self.yaw + dyaw) % 360.0
        self.pitch = float(np.clip(self.pitch + dpitch, -5.0, 85.0))

    def zoom(self, factor):
        self.dist = float(np.clip(self.dist * factor, 1.5, 30.0))

    # ---- drawing ----
    def _draw_rings(self, img):
        for r in (1, 2, 4, 8):
            ang = np.linspace(0, 2 * np.pi, 120, endpoint=False, dtype=np.float32)
            ring = np.stack([r * np.cos(ang), r * np.sin(ang), np.full_like(ang, self._ground_z)], axis=1)
            u, v, z = self._project(ring)
            ok = z > 0.1
            poly = np.stack([u, v], axis=1)[ok].astype(np.int32)
            if len(poly) > 2:
                cv2.polylines(img, [poly.reshape(-1, 1, 2)], len(poly) == len(ring), (60, 52, 44), 1, cv2.LINE_AA)
        # forward axis tick so "which way is the rover facing" is visible
        tip = np.array([[0, 0, self._ground_z], [1.2, 0, self._ground_z]], np.float32)
        u, v, z = self._project(tip)
        if (z > 0.1).all():
            cv2.arrowedLine(img, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])), (90, 180, 255), 2, cv2.LINE_AA, tipLength=0.25)

    def render(self, pts):
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = self.bg
        if len(pts):
            low = float(np.percentile(pts[:, 2], 2))
            self._ground_z = 0.9 * self._ground_z + 0.1 * low
        self._draw_rings(img)
        if len(pts) == 0:
            return img

        u, v, z = self._project(pts)
        ok = (z > 0.2) & (u >= 0) & (u < self.w - 1) & (v >= 0) & (v < self.h - 1)
        if not ok.any():
            return img
        u, v, z, p = u[ok].astype(np.int32), v[ok].astype(np.int32), z[ok], pts[ok]

        # colour by height relative to the ground estimate, fade with distance from the camera
        h_norm = np.clip((p[:, 2] - self._ground_z) / 2.5, 0, 1)
        col = HEIGHT_LUT[(h_norm * 255).astype(np.uint8)].astype(np.float32)
        fade = np.clip(1.25 - (z - self.dist) / (self.dist * 2.2 + 4), 0.35, 1.0)[:, None]
        col = (col * fade).astype(np.uint8)

        # far points first so near points win the pixel
        order = np.argsort(-z)
        img[v[order], u[order]] = col[order]
        # grow each point to ~2x2 px so the cloud reads as surfaces, not dust
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))

        # sensor marker at the origin
        o = self._project(np.array([[0, 0, 0]], np.float32))
        if o[2][0] > 0.1:
            cv2.circle(img, (int(o[0][0]), int(o[1][0])), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(o[0][0]), int(o[1][0])), 10, (90, 180, 255), 2, cv2.LINE_AA)
        return img
