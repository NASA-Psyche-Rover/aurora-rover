"""
Rolling LiDAR window for the dashboard's point-cloud panel.

Takes recent Livox points (LiDAR frame: x forward, y left, z up, metres) and
serves them to the browser, which draws them with WebGL (static/index.html).
Rendering on the client keeps dragging smooth and takes the 3D work off the
Jetson CPU. Pure numpy, no ROS, so it runs the same on the Jetson and in tests.

Wire format of `pack` (little endian), one response per poll:
  u32 latest_seq, u32 n_frames
  then per frame: u32 seq, u32 n_points, u32 age_ms, int16 xyz[n_points * 3]
Coordinates are metres * QSCALE (1/800 m steps, +-40 m range).
"""
import struct
import threading
import time
from collections import deque

import numpy as np

QSCALE = 800.0


class CloudAccumulator:
    """Thread-safe rolling window of point-cloud frames.

    `add` is called from the ROS callback, `pack` from request threads.
    Old frames are dropped by age; if the window still holds more than
    `max_points`, `pack` thins it (bandwidth and GPU load stay bounded)."""

    def __init__(self, window_s=8.0, max_points=150_000, max_range_m=40.0):
        self.window_s, self.max_points, self.max_range_m = window_s, max_points, max_range_m
        self._frames = deque()    # (t, seq, (n, 3) float32)
        self._lock = threading.Lock()
        self._arrivals = deque(maxlen=50)   # (t, n_points) for rate metrics
        self.version = 0          # sequence number of the newest frame

    def add(self, xyz, t=None):
        t = time.time() if t is None else t
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[np.isfinite(xyz).all(axis=1) & (r > 0.05) & (r < self.max_range_m)]
        with self._lock:
            self.version += 1
            self._frames.append((t, self.version, xyz))
            self._arrivals.append((t, len(xyz)))
            self._expire(t)

    def _expire(self, now):
        while self._frames and now - self._frames[0][0] > self.window_s:
            self._frames.popleft()

    def count(self, now=None):
        """Points a client is shown: the window total, thinned to `max_points`."""
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            return min(sum(len(f) for _, _, f in self._frames), self.max_points)

    def pack(self, since=0, now=None):
        """Binary message with every frame newer than `since` (see module doc).

        If the window holds more than `max_points`, every frame is strided by
        the same factor, so the client's total stays bounded and spans the
        whole scene."""
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            frames = list(self._frames)
            latest = self.version
        if since > latest:      # server restarted: resend everything
            since = 0
        total = sum(len(f) for _, _, f in frames)
        k = max(-(-total // self.max_points), 1)
        sel = [fr for fr in frames if fr[1] > since]
        out = [struct.pack("<II", latest, len(sel))]
        for t, seq, xyz in sel:
            q = np.clip(np.rint(xyz[seq % k::k] * QSCALE), -32767, 32767).astype("<i2")
            out.append(struct.pack("<III", seq, len(q), int(max(now - t, 0) * 1000)))
            out.append(q.tobytes())
        return b"".join(out)

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
