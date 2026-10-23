import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud_render import QSCALE, CloudAccumulator  # noqa: E402


def _room(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    floor = np.c_[rng.uniform(-4, 6, n), rng.uniform(-4, 4, n), np.full(n, -0.3)]
    wall = np.c_[np.full(n, 6.0), rng.uniform(-4, 4, n), rng.uniform(-0.3, 2.0, n)]
    return np.vstack([floor, wall]).astype(np.float32)


def unpack(buf):
    """Decode `pack` output -> (latest_seq, [(seq, age_ms, (n, 3) metres)])."""
    latest, n = struct.unpack_from("<II", buf, 0)
    off, frames = 8, []
    for _ in range(n):
        seq, npts, age = struct.unpack_from("<III", buf, off)
        off += 12
        xyz = np.frombuffer(buf, "<i2", 3 * npts, off).reshape(-1, 3) / QSCALE
        off += 6 * npts
        frames.append((seq, age, xyz))
    assert off == len(buf)
    return latest, frames


def test_accumulator_expires_old_frames():
    acc = CloudAccumulator(window_s=2.0)
    acc.add(_room(100), t=100.0)
    acc.add(_room(100), t=103.0)
    assert acc.count(now=103.5) == 200   # only the t=103 frame survives (2 * 100 pts)
    assert acc.count(now=110.0) == 0
    assert unpack(acc.pack(now=110.0))[1] == []


def test_pack_roundtrip_and_incremental():
    acc = CloudAccumulator()
    a, b = _room(100, seed=1), _room(100, seed=2)
    acc.add(a, t=10.0)
    acc.add(b, t=10.5)
    latest, frames = unpack(acc.pack(now=11.0))
    assert latest == 2 and [f[0] for f in frames] == [1, 2]
    assert [f[1] for f in frames] == [1000, 500]                      # age in ms
    assert np.abs(frames[0][2] - a).max() <= 0.5 / QSCALE + 1e-6      # quantisation error only
    latest, frames = unpack(acc.pack(since=1, now=11.0))
    assert [f[0] for f in frames] == [2]
    _, frames = unpack(acc.pack(since=99, now=11.0))                  # client ahead of a restarted server
    assert len(frames) == 2


def test_accumulator_caps_points_and_drops_junk():
    acc = CloudAccumulator(max_points=500)
    xyz = np.vstack([_room(1000), [[np.nan, 0, 0], [0, 0, 0], [500, 0, 0]]])
    acc.add(xyz, t=1.0)
    assert acc.count(now=1.0) == 500
    _, frames = unpack(acc.pack(now=1.0))
    pts = frames[0][2]
    assert len(pts) <= 500
    assert np.isfinite(pts).all() and (np.linalg.norm(pts, axis=1) > 0.05).all()


def test_rates():
    acc = CloudAccumulator()
    for i in range(11):
        acc.add(_room(100), t=10.0 + i * 0.1)
    msgs, pps = acc.rates(now=11.0)
    assert abs(msgs - 10.0) < 0.5
    assert pps > 1000
