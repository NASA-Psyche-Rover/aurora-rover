import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud_render import CloudAccumulator, CloudRenderer  # noqa: E402


def _room(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    floor = np.c_[rng.uniform(-4, 6, n), rng.uniform(-4, 4, n), np.full(n, -0.3)]
    wall = np.c_[np.full(n, 6.0), rng.uniform(-4, 4, n), rng.uniform(-0.3, 2.0, n)]
    return np.vstack([floor, wall]).astype(np.float32)


def test_accumulator_expires_old_frames():
    acc = CloudAccumulator(window_s=2.0)
    acc.add(_room(100), t=100.0)
    acc.add(_room(100), t=103.0)
    assert len(acc.points(now=103.5)) == 200   # only the t=103 frame survives (2 * 100 pts)
    assert len(acc.points(now=110.0)) == 0


def test_accumulator_caps_points_and_drops_junk():
    acc = CloudAccumulator(max_points=500)
    xyz = np.vstack([_room(1000), [[np.nan, 0, 0], [0, 0, 0], [500, 0, 0]]])
    acc.add(xyz, t=1.0)
    pts = acc.points(now=1.0)
    assert len(pts) == 500
    assert np.isfinite(pts).all() and (np.linalg.norm(pts, axis=1) > 0.05).all()


def test_rates():
    acc = CloudAccumulator()
    for i in range(11):
        acc.add(_room(100), t=10.0 + i * 0.1)
    msgs, pps = acc.rates(now=11.0)
    assert abs(msgs - 10.0) < 0.5
    assert pps > 1000


def test_render_draws_points_and_handles_empty():
    r = CloudRenderer(size=(320, 240))
    empty = r.render(np.zeros((0, 3), np.float32))
    assert empty.shape == (240, 320, 3)
    img = r.render(_room())
    bg = np.broadcast_to(r.bg, img.shape)
    lit = (np.abs(img.astype(int) - bg.astype(int)).sum(axis=2) > 60).mean()
    assert lit > 0.02


def test_rotate_disables_orbit_and_clamps_pitch():
    r = CloudRenderer()
    r.rotate(10, 500)
    assert not r.auto_orbit and r.pitch == 85.0
    r.zoom(100)
    assert r.dist == 30.0
