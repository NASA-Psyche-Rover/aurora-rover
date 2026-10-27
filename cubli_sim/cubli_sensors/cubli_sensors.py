"""cubli_sensors -- simulated sensor views for the AURORA Cubli rover.

Renders what the cube's sensor head would see inside a MuJoCo scene:

    1  camera       OAK-D Lite style RGB (55 deg vertical FOV, tilt from the XML camera)
    2  stereo depth MuJoCo depth + disparity-domain noise, 0.2-8 m, TURBO, ROI readout
    3  LiDAR 3D     Livox Mid-360 style cloud, orbitable 3D view
    4  LiDAR BEV    same cloud, top-down in the sensor frame
    5  3rd person   orbitable tracking camera that shows the cube
    6  split        3rd person | LiDAR 3D side by side
    0  grid         views 1-4

This module never steps physics. The caller steps (or kinematically sets qpos and
calls mj_forward); sensors only read `data`.

Sensor frame: x forward, y left, z up, origin at `lidar_site`.
  gimbal ON : frame stays level, yawed to `heading` (rad, world yaw of travel).
  gimbal OFF: frame is rigidly attached to the cube (the site's own frame).

LiDAR model (Mid-360): 360 x [-7, +52] deg, 200k points/s, non-repetitive pattern whose
coverage fills in over time, 2 cm range noise, 0.1-40 m. Each lidar() call fires
`lidar_rays_per_call` new rays and returns the newest `lidar_buffer` points (an
integration window, motion-compensated with the sim pose) in the *current* sensor frame.

`lidar()` is the only point source. Assign `lidar_source = callable -> (N,3)` to
feed real Livox points instead; every visualization keeps working.
"""
from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

if sys.platform.startswith("linux"):
    # Headless / Jetson: EGL needs no X server. Must be set before importing mujoco.
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import mujoco

__all__ = ["CubliSensors"]

# ---- sensor specs ---------------------------------------------------------
CAM_VFOV_DEG = 55.0
STEREO_BASELINE_M = 0.075
STEREO_FOCAL_PX = 450.0
DISPARITY_SIGMA_PX = 0.15
DEPTH_MIN_M, DEPTH_MAX_M = 0.2, 8.0
OBSTACLE_M = 0.5
DARK_VALID_FRACTION = 0.02          # passive stereo with the lights off

LIDAR_ELEV_DEG = (-7.0, 52.0)
LIDAR_PTS_PER_S = 200_000
LIDAR_NOISE_M = 0.02
LIDAR_MAX_M = 40.0
LIDAR_MIN_M = 0.10

EARTH_G, PSYCHE_G = 9.81, 0.144
FOOTER_H = 28
TAB_H = 30
TABS = [(1, "1 CAMERA"), (2, "2 DEPTH"), (3, "3 LIDAR 3D"), (4, "4 LIDAR TOP"),
        (5, "5 3RD PERSON"), (6, "6 SPLIT"), (0, "0 GRID")]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TURBO = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1),
                           cv2.COLORMAP_TURBO).reshape(256, 3)      # BGR LUT
# z (mm, 0..8000) -> uint8 TURBO index, log-spaced so near range keeps contrast; near = 255
_zmm = np.maximum(np.arange(8001) / 1000.0, DEPTH_MIN_M)
_DEPTH_IDX = (255.0 * (1.0 - np.log(_zmm / DEPTH_MIN_M) / math.log(DEPTH_MAX_M / DEPTH_MIN_M))
              ).clip(0, 255).astype(np.uint8)
_CUBE = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * 0.075
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _turbo(v: np.ndarray) -> np.ndarray:
    """v in [0,1] -> BGR uint8 (0 = blue, 1 = red)."""
    return _TURBO[(np.clip(v, 0.0, 1.0) * 255.0).astype(np.uint8)]


def _put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    # Outline via offset copies: a thicker stroke changes glyph advance in OpenCV 5
    # and the outline drifts away from the text.
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, text, (x + dx, y + dy), _FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


class _Orbit:
    """Yaw/pitch/distance camera around a target. yaw 0 = behind (looking +x)."""

    def __init__(self, yaw, pitch, dist, dmin, dmax):
        self.yaw, self.pitch, self.dist, self.dmin, self.dmax = yaw, pitch, dist, dmin, dmax

    def drag(self, dx, dy):
        self.yaw -= dx * 0.35
        self.pitch = float(np.clip(self.pitch + dy * 0.35, -5.0, 89.0))

    def zoom(self, f):
        self.dist = float(np.clip(self.dist * f, self.dmin, self.dmax))


class _Zoomer:
    """Right-drag adapter: vertical motion zooms the wrapped orbit."""

    def __init__(self, orb):
        self.orb = orb

    def drag(self, dx, dy):
        self.orb.zoom(1.0 + 0.01 * dy)


# cv2.waitKeyEx arrow codes (macOS Cocoa, Linux GTK/Qt, Windows) -> orbit (dx, dy) in pixels
_ARROWS = {}
for _codes, _d in (((63234, 65361, 2424832), (-18, 0)), ((63235, 65363, 2555904), (18, 0)),
                   ((63232, 65362, 2490368), (0, 12)), ((63233, 65364, 2621440), (0, -12))):
    for _c in _codes:
        _ARROWS[_c] = _d


class CubliSensors:
    def __init__(self, model, data, body="cubli", lidar_site="lidar_site",
                 body_cam="cubli_cam", size=(640, 480), hide_group=3,
                 shadows=True, seed=0, lidar_rays_per_call=6000, lidar_buffer=60000,
                 lidar_threads=None):
        self.model, self.data = model, data
        self.gimbal = True
        self.lights = True
        self.heading = 0.0              # world yaw (rad) of the direction of travel
        self.view = 6
        self.lidar_source = None        # swap-in hook for real Livox points
        self.lidar_rays_per_call = lidar_rays_per_call
        self.bev_range = 8.0            # metres shown from centre to edge
        self.mast = 0.0                 # sensor-head extension (m) along sensor +z, e.g. a pop-up mast
        self.follow = True              # 3rd person: orbit follows heading (True) or stays world-fixed
        self.cube_half = 0.075          # used for the ground-plane estimate under the cube
        self.cam_orbit = _Orbit(yaw=-35.0, pitch=20.0, dist=0.9, dmin=0.3, dmax=12.0)
        self.cloud_orbit = _Orbit(yaw=0.0, pitch=24.0, dist=4.0, dmin=0.8, dmax=40.0)
        self.timing = {}                # EMA ms per stage, for profiling
        self._rng = np.random.default_rng(seed)
        self._panes, self._tabs, self._drag = [], [], None
        self._az = None                 # smoothed 3rd-person azimuth (deg)

        def need(kind, name):
            i = mujoco.mj_name2id(model, kind, name)
            if i < 0:
                raise ValueError(f"CubliSensors: no {kind.name} named {name!r} in model")
            return i
        self._bid = need(mujoco.mjtObj.mjOBJ_BODY, body)
        self._sid = need(mujoco.mjtObj.mjOBJ_SITE, lidar_site)
        self._cid = need(mujoco.mjtObj.mjOBJ_CAMERA, body_cam)

        # Sensor mounting, measured once from the model's reference pose. Everything is
        # stored in the *sensor frame* (= lidar_site frame) so gimbal ON/OFF is just a
        # different choice of sensor-frame orientation.
        d0 = mujoco.MjData(model)
        mujoco.mj_forward(model, d0)
        Rs = d0.site_xmat[self._sid].reshape(3, 3)
        ps = d0.site_xpos[self._sid]
        self._off_lidar = Rs.T @ (ps - d0.xpos[self._bid])             # body origin -> lidar
        self._off_cam = Rs.T @ (d0.cam_xpos[self._cid] - d0.xpos[self._bid])
        self._R_cam = Rs.T @ d0.cam_xmat[self._cid].reshape(3, 3)       # camera in sensor frame
        self._R_site_in_body = d0.xmat[self._bid].reshape(3, 3).T @ Rs   # site in body frame

        # Renderer: one GL context for rgb/depth/world.
        self._w, self._h = size
        g = model.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, self._w), max(g.offheight, self._h)
        model.cam_fovy[self._cid] = CAM_VFOV_DEG
        self._r = mujoco.Renderer(model, self._h, self._w)
        if not shadows:
            self._r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._far = model.vis.map.zfar * model.stat.extent
        self._opt_sensor = mujoco.MjvOption()
        self._opt_sensor.geomgroup[hide_group] = 0          # cube hides from its sensors
        self._opt_world = mujoco.MjvOption()
        self._opt_world.geomgroup[hide_group] = 1           # ...but not from the world view
        self._free_cam = mujoco.MjvCamera()
        self._free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        yy, xx = np.mgrid[0:self._h, 0:self._w].astype(np.float32)
        rr = ((xx - self._w / 2) ** 2 + (yy - self._h / 2) ** 2) / ((self._w / 2) ** 2 + (self._h / 2) ** 2)
        self._vig8 = np.repeat(((1.0 - 0.38 * rr) * 255).astype(np.uint8)[..., None], 3, 2)
        # pre-generated sensor-noise frames, cycled (fresh Gaussian noise every frame is costly)
        self._noise = [self._rng.normal(0, 2.5, (self._h, self._w, 1)).astype(np.int16).repeat(3, 2)
                       for _ in range(6)]
        self._noise_k = 0

        # LiDAR
        self._geomgroup = np.ones(6, np.uint8)
        self._geomgroup[hide_group] = 0
        self._lidar_k = 0
        self._buf = np.zeros((lidar_buffer, 3), np.float32)
        self._buf_n = self._buf_i = 0
        self._last_origin = None
        self._use_multiray = hasattr(mujoco, "mj_multiRay")
        # mj_multiRay releases the GIL, so ray chunks run truly parallel (~4x on 4 cores).
        nt = lidar_threads or max(1, min(4, (os.cpu_count() or 2) - 1))
        self._pool = ThreadPoolExecutor(nt) if nt > 1 else None
        self._nthreads = nt

    # ------------------------------------------------------------------ state
    @property
    def psyche(self) -> bool:
        return abs(self.model.opt.gravity[2]) < 1.0

    @psyche.setter
    def psyche(self, on: bool):
        self.model.opt.gravity[2] = -(PSYCHE_G if on else EARTH_G)

    @property
    def gravity(self) -> float:
        return float(abs(self.model.opt.gravity[2]))

    def _orbit_targets(self):
        return {0: [self.cloud_orbit], 3: [self.cloud_orbit], 5: [self.cam_orbit],
                6: [self.cam_orbit, self.cloud_orbit]}.get(self.view, [])

    def handle_key(self, key: int) -> bool:
        """Feed a cv2.waitKeyEx code. Handles 0-6, G, L, P, F, [ ], - =, arrow keys (orbit).
        Returns True if consumed."""
        if key < 0:
            return False
        arrow = _ARROWS.get(key)
        if arrow is not None:
            for o in self._orbit_targets():
                o.drag(*arrow)
            return True
        ch = chr(key & 0xFF).lower()
        if ch in "0123456":
            self.view = int(ch)
        elif ch == "f":
            self.follow = not self.follow
        elif ch == "g":
            self.gimbal = not self.gimbal
            self.clear_lidar()
        elif ch == "l":
            self.lights = not self.lights
        elif ch == "p":
            self.psyche = not self.psyche
        elif ch == "[":
            self.bev_range = max(2.0, self.bev_range - 1.0)
        elif ch == "]":
            self.bev_range = min(40.0, self.bev_range + 1.0)
        elif ch in "-=+":
            f = 1.15 if ch == "-" else 1 / 1.15
            if self.view in (5, 6):
                self.cam_orbit.zoom(f)
            if self.view in (0, 3, 6):
                self.cloud_orbit.zoom(f)
            if self.view == 4:
                self.bev_range = float(np.clip(self.bev_range * f, 2.0, 40.0))
        else:
            return False
        return True

    def on_mouse(self, event, x, y, flags, *_):
        """Pass straight from cv2.setMouseCallback. Click tabs, drag to orbit, wheel to zoom."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for x0, x1, v in self._tabs:
                if y < TAB_H and x0 <= x < x1:
                    self.view = v
                    return
            for x0, y0, x1, y1, orb in self._panes:
                if x0 <= x < x1 and y0 <= y < y1:
                    self._drag = (orb, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            # Track the button ourselves: macOS' HighGUI doesn't set EVENT_FLAG_LBUTTON on moves.
            orb, px, py = self._drag
            orb.drag(x - px, y - py)
            self._drag = (orb, x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:              # right-drag zooms
            for x0, y0, x1, y1, orb in self._panes:
                if x0 <= x < x1 and y0 <= y < y1:
                    self._drag = (_Zoomer(orb), x, y)
        elif event in (cv2.EVENT_MOUSEWHEEL, getattr(cv2, "EVENT_MOUSEHWHEEL", -99)):
            delta = cv2.getMouseWheelDelta(flags)
            for x0, y0, x1, y1, orb in self._panes:
                if x0 <= x < x1 and y0 <= y < y1:
                    orb.zoom(0.9 if delta > 0 else 1 / 0.9)

    def head_pose(self):
        """(origin xyz of lidar site, 3x3 sensor-frame rotation) in the world."""
        d = self.data
        if self.gimbal:
            R = _rz(self.heading)
        else:
            R = d.xmat[self._bid].reshape(3, 3) @ self._R_site_in_body
        return d.xpos[self._bid] + R @ (self._off_lidar + [0.0, 0.0, self.mast]), R

    def _cam_pose(self):
        _, R = self.head_pose()
        return self.data.xpos[self._bid] + R @ (self._off_cam + [0.0, 0.0, self.mast]), R @ self._R_cam

    def _tic(self):
        return time.perf_counter()

    def _toc(self, key, t0):
        ms = (time.perf_counter() - t0) * 1e3
        self.timing[key] = ms if key not in self.timing else 0.9 * self.timing[key] + 0.1 * ms

    # ------------------------------------------------------------- GL sensors
    def _render_cam(self, depth=False):
        r = self._r
        pos, R = self._cam_pose()
        r.update_scene(self.data, camera=self._cid, scene_option=self._opt_sensor)
        for k in range(2):                       # [1] is the (unused) stereo eye
            c = r.scene.camera[k]
            c.pos[:] = pos
            c.forward[:] = -R[:, 2]
            c.up[:] = R[:, 1]
        if depth:
            r.enable_depth_rendering()
        try:
            return r.render().copy()
        finally:
            if depth:
                r.disable_depth_rendering()

    def rgb(self) -> np.ndarray:
        """HxWx3 uint8 RGB with lens vignetting and sensor noise. Lights off => near-black."""
        t0 = self._tic()
        img = self._render_cam()
        self._noise_k = (self._noise_k + 1) % len(self._noise)
        noise = self._noise[self._noise_k]
        if self.lights:
            img = cv2.multiply(img, self._vig8, scale=1 / 255.0)          # lens vignetting
            out = cv2.add(img.astype(np.int16), noise)
        else:
            out = (img >> 6).astype(np.int16) + 6 + 2 * noise            # ~2% signal + read noise
        self._toc("rgb", t0)
        return np.clip(out, 0, 255).astype(np.uint8)

    def depth(self) -> np.ndarray:
        """HxW float32 metres (z-depth); 0 = invalid. Stereo-realistic."""
        t0 = self._tic()
        z = self._render_cam(depth=True).astype(np.float32)
        fb = STEREO_FOCAL_PX * STEREO_BASELINE_M
        valid = (z >= DEPTH_MIN_M) & (z <= DEPTH_MAX_M) & (z < 0.99 * self._far)
        # Occlusion / steep-slope dropout: stereo cannot match across depth jumps.
        gx = np.abs(np.diff(z, axis=1, prepend=z[:, :1]))
        gy = np.abs(np.diff(z, axis=0, prepend=z[:1]))
        valid &= (gx < 0.30 * z) & (gy < 0.30 * z)
        # Noise lives in disparity (sigma ~0.15 px), so depth error grows with z^2.
        noise = self._rng.standard_normal(z.shape, dtype=np.float32)
        disp = fb / np.maximum(z, 1e-3) + DISPARITY_SIGMA_PX * noise
        valid &= disp > 0.3
        zn = fb / np.maximum(disp, 0.3)
        valid &= (zn >= DEPTH_MIN_M) & (zn <= DEPTH_MAX_M)
        if not self.lights:                      # passive stereo: no texture, no match
            valid &= self._rng.random(z.shape, dtype=np.float32) < DARK_VALID_FRACTION
        out = np.where(valid, zn, 0.0).astype(np.float32)
        self._toc("depth", t0)
        return out

    def world(self) -> np.ndarray:
        """HxWx3 uint8 RGB orbiting 3rd-person camera that shows the cube."""
        t0 = self._tic()
        c, o = self._free_cam, self.cam_orbit
        c.lookat[:] = self.data.xpos[self._bid] + np.array([0.0, 0.0, 0.05])
        c.distance = o.dist
        # only the heading-follow is smoothed; user orbit input applies immediately
        target = math.degrees(self.heading) if self.follow else 0.0
        if self._az is None:
            self._az = target
        self._az += ((target - self._az + 180.0) % 360.0 - 180.0) * 0.25
        c.azimuth = self._az + o.yaw
        c.elevation = -o.pitch
        self._r.update_scene(self.data, camera=c, scene_option=self._opt_world)
        img = self._r.render().copy()
        self._toc("world", t0)
        return img

    # ------------------------------------------------------------------ LiDAR
    def clear_lidar(self):
        self._buf_n = self._buf_i = 0

    def _pattern(self, n):
        """Non-repetitive scan directions (sensor frame): R2 low-discrepancy sequence over
        the Mid-360 FOV. Like the real prism scanner, coverage keeps filling in over time."""
        k = self._lidar_k + np.arange(n, dtype=np.float64)
        self._lidar_k += n
        g = 1.32471795724474602596
        az = 2 * math.pi * ((0.5 + k / g) % 1.0)
        el = np.deg2rad(LIDAR_ELEV_DEG[0] + (LIDAR_ELEV_DEG[1] - LIDAR_ELEV_DEG[0]) * ((0.5 + k / g ** 2) % 1.0))
        ce = np.cos(el)
        return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], -1)

    def lidar(self) -> np.ndarray:
        """(N,3) float32 points in the current sensor frame (x fwd, y left, z up).

        The only point source used by the visualizations. Replace with real Livox
        data by setting `lidar_source` (callable returning (N,3) in the same frame).
        """
        if self.lidar_source is not None:
            return np.asarray(self.lidar_source(), np.float32).reshape(-1, 3)
        t0 = self._tic()
        pos, R = self.head_pose()
        if self._last_origin is not None and np.linalg.norm(pos - self._last_origin) > 1.0:
            self.clear_lidar()                   # teleport / reset: old points are invalid
        self._last_origin = pos.copy()
        n = int(self.lidar_rays_per_call)
        vec = np.ascontiguousarray(self._pattern(n) @ R.T)          # world-frame ray dirs
        dist = np.empty(n)
        gid = np.empty(n, np.int32)

        def cast(a, b):
            mujoco.mj_multiRay(self.model, self.data, pos, vec[a:b].ravel(), self._geomgroup,
                               1, self._bid, gid[a:b], dist[a:b], None, b - a, LIDAR_MAX_M)
        if self._use_multiray:
            try:
                if self._pool is None:
                    cast(0, n)
                else:
                    e = np.linspace(0, n, self._nthreads + 1).astype(int)
                    list(self._pool.map(lambda k: cast(e[k], e[k + 1]), range(self._nthreads)))
            except (TypeError, AttributeError):
                self._use_multiray = False
        if not self._use_multiray:                                  # slow fallback
            gid = np.zeros(1, np.int32)
            for i in range(n):
                dist[i] = mujoco.mj_ray(self.model, self.data, pos, vec[i],
                                        self._geomgroup, 1, self._bid, gid)
        hit = (dist > LIDAR_MIN_M) & (dist < LIDAR_MAX_M)
        r = dist[hit] + self._rng.normal(0.0, LIDAR_NOISE_M, hit.sum())
        self._push(pos + vec[hit] * r[:, None])
        pts = ((self._buf[:self._buf_n] - pos) @ R).astype(np.float32)
        self._toc("lidar", t0)
        return pts

    def _push(self, pw):
        cap = len(self._buf)
        pw = pw[-cap:]
        idx = (self._buf_i + np.arange(len(pw))) % cap
        self._buf[idx] = pw
        self._buf_i = (self._buf_i + len(pw)) % cap
        self._buf_n = min(self._buf_n + len(pw), cap)

    # ---------------------------------------------------------------- drawing
    def _title(self, img, text):
        _put(img, text, (8, 20), 0.55, (255, 255, 255), 1)

    def _rgb_view(self):
        img = cv2.cvtColor(self.rgb(), cv2.COLOR_RGB2BGR)
        self._title(img, "CAMERA  OAK-D Lite RGB  55 deg V-FOV")
        return img

    def _depth_view(self, z=None):
        z = self.depth() if z is None else z
        valid = z > 0
        idx = _DEPTH_IDX[np.minimum((z * 1000.0).astype(np.int32), 8000)]
        img = cv2.applyColorMap(idx, cv2.COLORMAP_TURBO)
        img[~valid] = 0                                           # invalid pixels black
        h, w = z.shape
        bar = _turbo(1.0 - np.linspace(0, 1, h - 80))[:, None, :].repeat(10, 1)
        img[40:h - 40, w - 24:w - 14] = bar
        for m_ in (0.2, 0.5, 1, 2, 4, 8):
            y = int(40 + (h - 80) * math.log(m_ / DEPTH_MIN_M) / math.log(DEPTH_MAX_M / DEPTH_MIN_M))
            _put(img, f"{m_:g}m", (w - 60, y + 4), 0.35)
        x0, x1, y0, y1 = int(w * .42), int(w * .58), int(h * .42), int(h * .58)
        roi = z[y0:y1, x0:x1]
        ok = roi > 0
        # Ground-plane removal: lift ROI pixels to world height using the camera pose, and only
        # count pixels >4 cm above the cube's ground as obstacle (a low camera always sees floor).
        hgt = self._pixel_heights(roi, x0, y0)
        obst = ok & (hgt > 0.04)
        ahead, floor = None, False
        if ok.sum() >= 0.10 * roi.size:          # a few stray matches is not a measurement
            if obst.sum() >= 0.05 * roi.size:
                ahead = float(np.median(roi[obst]))
            else:
                ahead, floor = float(np.median(roi[ok])), True
        hot = ahead is not None and not floor and ahead < OBSTACLE_M
        col = (0, 0, 255) if hot else (255, 255, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        label = ("Ahead: -- (no data)" if ahead is None else
                 f"Ahead: {ahead:.2f} m" + (" (floor)" if floor else ""))
        _put(img, label, (x0 - 14, y0 - 8), 0.6, col, 2)
        if hot:
            _put(img, "OBSTACLE", (x0 + 4, y1 + 24), 0.7, (0, 0, 255), 2)
        self._title(img, "STEREO DEPTH  7.5 cm baseline  0.2-8 m  near = red")
        _put(img, f"valid {100.0 * valid.mean():.0f}%", (8, 40), 0.45)
        return img

    def _pixel_heights(self, zpatch, x0, y0):
        """World height above the cube's ground plane for a patch of depth pixels."""
        h, w = self._h, self._w
        f = (h / 2) / math.tan(math.radians(self.model.cam_fovy[self._cid]) / 2)
        pos, R = self._cam_pose()
        vv, uu = np.mgrid[y0:y0 + zpatch.shape[0], x0:x0 + zpatch.shape[1]].astype(np.float32)
        # MuJoCo camera: x right, y up, looks along -z; zpatch is distance along the optical axis
        dz = R[2, 0] * (uu - w / 2) / f - R[2, 1] * (vv - h / 2) / f - R[2, 2]
        Rb = self.data.xmat[self._bid].reshape(3, 3)
        ground = self.data.xpos[self._bid][2] - self.cube_half * np.abs(Rb[2]).sum()
        return pos[2] + zpatch * dz - ground

    def _cube_in_sensor(self):
        """Cube centre and rotation expressed in the sensor frame."""
        _, R = self.head_pose()
        Rb = self.data.xmat[self._bid].reshape(3, 3)
        return -(self._off_lidar + [0.0, 0.0, self.mast]), R.T @ Rb

    def _cloud_cam(self):
        o = self.cloud_orbit
        tgt = -(self._off_lidar + [0.0, 0.0, self.mast])
        y, p = math.radians(o.yaw), math.radians(o.pitch)
        eye = tgt + o.dist * np.array([-math.cos(p) * math.cos(y), -math.cos(p) * math.sin(y), math.sin(p)])
        f = tgt - eye
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 0, 1.0])
        right /= max(np.linalg.norm(right), 1e-6)
        up = np.cross(right, f)
        foc = (self._w / 2) / math.tan(math.radians(75) / 2)
        return eye, f, right, up, foc

    def _project(self, p, cam):
        eye, f, right, up, foc = cam
        rel = p - eye
        zc = rel @ f
        zs = np.maximum(zc, 1e-3)
        return self._w / 2 + foc * (rel @ right) / zs, self._h / 2 - foc * (rel @ up) / zs, zc

    def _cloud_view(self, pts):
        w, h = self._w, self._h
        img = np.full((h, w, 3), 12, np.uint8)
        cam = self._cloud_cam()
        c, Rc = self._cube_in_sensor()
        # 1 m reference grid on the cube's ground plane (sensor frame)
        gz = c[2] - 0.075
        for k in range(-10, 11):
            for a, b in (((k, -10, gz), (k, 10, gz)), ((-10, k, gz), (10, k, gz))):
                seg = np.linspace(a, b, 40)
                u, v, zc = self._project(seg, cam)
                ok = zc > 0.1
                pts2 = np.stack([u[ok], v[ok]], 1).astype(np.int32)
                if len(pts2) > 1:
                    cv2.polylines(img, [pts2], False, (34, 34, 34), 1, cv2.LINE_AA)
        if len(pts):
            u, v, zc = self._project(pts, cam)
            rng_m = np.linalg.norm(pts, axis=1)
            ok = (zc > 0.1) & (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)
            u, v, zc, rng_m = u[ok].astype(np.int32), v[ok].astype(np.int32), zc[ok], rng_m[ok]
            order = np.argsort(-zc)                                  # far first, near wins
            u, v, zc, rng_m = u[order], v[order], zc[order], rng_m[order]
            col = _turbo(1.0 - np.sqrt(rng_m / 25.0))                 # near = red
            img[v, u] = col
            big = zc < 0.6 * self.cloud_orbit.dist + 2.0
            for dx, dy in ((1, 0), (0, 1), (1, 1)):
                img[v[big] + dy, u[big] + dx] = col[big]
        # cube + heading arrow, in the cube's true attitude relative to the head
        uu, vv, zz = self._project(c + _CUBE @ Rc.T, cam)
        if (zz > 0.05).all():
            for a, b in _EDGES:
                cv2.line(img, (int(uu[a]), int(vv[a])), (int(uu[b]), int(vv[b])), (30, 140, 255), 2, cv2.LINE_AA)
        _, R = self.head_pose()
        rel = self.heading - math.atan2(R[1, 0], R[0, 0])
        tip = np.array([math.cos(rel), math.sin(rel), 0.0]) * 0.6 + np.array([0, 0, c[2]])
        (u0, u1), (v0, v1), zs = self._project(np.array([[0, 0, c[2]], tip]), cam)
        if (zs > 0.05).all():
            cv2.arrowedLine(img, (int(u0), int(v0)), (int(u1), int(v1)), (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)
        self._title(img, "LIDAR 3D  Mid-360 style  colour = range (red near)")
        _put(img, f"{len(pts):,} pts   360 x [{LIDAR_ELEV_DEG[0]:g}, {LIDAR_ELEV_DEG[1]:g}] deg   drag: orbit  wheel/-=: zoom",
             (8, 40), 0.42)
        return img

    def _bev_view(self, pts):
        w, h = self._w, self._h
        img = np.full((h, w, 3), 12, np.uint8)
        cx, cy = w // 2, h // 2
        ppm = (min(w, h) / 2 - 6) / self.bev_range
        step = 1 if self.bev_range <= 10 else 5
        for k in range(step, int(self.bev_range) + 1, step):
            cv2.circle(img, (cx, cy), int(k * ppm), (60, 60, 60), 1, cv2.LINE_AA)
            _put(img, f"{k}m", (cx + int(k * ppm * 0.707) + 2, cy - int(k * ppm * 0.707) - 2), 0.35,
                 (170, 170, 170))
        cv2.line(img, (cx, 0), (cx, h), (45, 45, 45), 1)
        cv2.line(img, (0, cy), (w, cy), (45, 45, 45), 1)
        if len(pts):
            m = np.hypot(pts[:, 0], pts[:, 1]) < self.bev_range * 1.5
            p = pts[m]
            p = p[np.argsort(p[:, 2])]                               # high points drawn last
            u = (cx - p[:, 1] * ppm).astype(np.int32)                # +y (left) -> image left
            v = (cy - p[:, 0] * ppm).astype(np.int32)                # +x (fwd)  -> image up
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            c0, _ = self._cube_in_sensor()
            img[v[ok], u[ok]] = _turbo((p[ok, 2] - c0[2] + 0.6) / 1.6)   # colour = height above cube base
        c, Rc = self._cube_in_sensor()
        xy = (_CUBE @ Rc.T)[:, :2]
        k = max(1.0, 20.0 / (0.15 * ppm))
        poly = np.stack([cx - xy[:, 1] * ppm * k, cy - xy[:, 0] * ppm * k], 1).astype(np.int32)
        hull = cv2.convexHull(poly)
        cv2.fillConvexPoly(img, hull, (30, 140, 255))
        cv2.polylines(img, [hull], True, (255, 255, 255), 1, cv2.LINE_AA)
        _, R = self.head_pose()
        rel = self.heading - math.atan2(R[1, 0], R[0, 0])          # travel dir in sensor frame
        tip = (int(cx - math.sin(rel) * 48), int(cy - math.cos(rel) * 48))
        cv2.arrowedLine(img, (cx, cy), tip, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)
        self._title(img, f"LIDAR TOP-DOWN  colour = height  rings {step} m  arrow = travel")
        _put(img, "[ ] or -/=: zoom", (8, 40), 0.42)
        return img

    def _world_view(self):
        img = cv2.cvtColor(self.world(), cv2.COLOR_RGB2BGR)
        self._title(img, f"3RD PERSON ({'follow' if self.follow else 'free'}, F)   drag/arrows: orbit   wheel/-=: zoom")
        return img

    def _tabbar(self, width, view):
        bar = np.full((TAB_H, width, 3), 22, np.uint8)
        self._tabs, x = [], 4
        for v, name in TABS:
            (tw, _), _b = cv2.getTextSize(name, _FONT, 0.5, 1)
            x1 = x + tw + 20
            active = v == view
            cv2.rectangle(bar, (x, 3), (x1, TAB_H - 1), (200, 120, 20) if active else (55, 55, 55), -1)
            cv2.putText(bar, name, (x + 10, 21), _FONT, 0.5, (255, 255, 255) if active else (190, 190, 190),
                        1, cv2.LINE_AA)
            self._tabs.append((x, x1, v))
            x = x1 + 4
        return bar

    def _footer(self, width):
        bar = np.full((FOOTER_H, width, 3), 28, np.uint8)
        g = f"PSYCHE {self.gravity:.3f}" if self.psyche else f"EARTH {self.gravity:.2f}"
        on = lambda b: "ON" if b else "OFF"
        txt = f"GIMBAL {on(self.gimbal)}   LIGHTS {on(self.lights)}   GRAVITY {g} m/s^2"
        _put(bar, txt, (8, 19), 0.5, (255, 255, 255))
        (tw, _), _b = cv2.getTextSize("SIMULATED", _FONT, 0.55, 2)
        _put(bar, "SIMULATED", (width - tw - 8, 19), 0.55, (0, 170, 255), 2)
        return bar

    # ---------------------------------------------------------------- compose
    def compose(self, view=None, quad_scale=0.75) -> np.ndarray:
        """BGR image for view 0-6 (default: self.view), with tab bar and footer."""
        view = self.view if view is None else int(view)
        t0 = self._tic()
        w, h = self._w, self._h
        panes = []
        if view == 0:
            pts = self.lidar()
            tiles = [self._rgb_view(), self._depth_view(), self._cloud_view(pts), self._bev_view(pts)]
            if quad_scale != 1.0:
                tiles = [cv2.resize(t, None, fx=quad_scale, fy=quad_scale, interpolation=cv2.INTER_AREA)
                         for t in tiles]
            body = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            tw, th = tiles[0].shape[1], tiles[0].shape[0]
            panes.append((0, th, tw, 2 * th, self.cloud_orbit))
        elif view == 1:
            body = self._rgb_view()
        elif view == 2:
            body = self._depth_view()
        elif view == 3:
            body = self._cloud_view(self.lidar())
            panes.append((0, 0, w, h, self.cloud_orbit))
        elif view == 4:
            body = self._bev_view(self.lidar())
        elif view == 5:
            body = self._world_view()
            panes.append((0, 0, w, h, self.cam_orbit))
        elif view == 6:
            body = np.hstack([self._world_view(), self._cloud_view(self.lidar())])
            panes += [(0, 0, w, h, self.cam_orbit), (w, 0, 2 * w, h, self.cloud_orbit)]
        else:
            raise ValueError("view must be 0-6")
        self._panes = [(x0, y0 + TAB_H, x1, y1 + TAB_H, o) for x0, y0, x1, y1, o in panes]
        out = np.vstack([self._tabbar(body.shape[1], view), body, self._footer(body.shape[1])])
        self._toc("compose", t0)
        return out

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
        self._r.close()
