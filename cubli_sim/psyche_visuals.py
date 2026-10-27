"""Realistic look for the live MuJoCo window (and for the simulated cameras, which render the same scene).

Physics is untouched: the heightfield and the boulder ellipsoids still do all the contacts, they are only moved
to geom group 3 (not drawn, not seen by the cube's sensors). On top of them, visual-only geometry:
  - the same terrain as a smooth-shaded mesh with a tiled procedural regolith texture: dark grey, slightly red,
    ~1 mm grains and clods with micro-shadows baked for the sim's sun direction (Psyche visible albedo ~0.16)
  - angular, faceted rocks fitted over each collision ellipsoid, partly buried
  - a few thousand pebbles and cobbles (<= 1.5 cm, visual only, too small to matter for the 15 cm cube)
No surface images of Psyche exist yet (the spacecraft arrives in 2029): this is a physically-plausible analog.
"""
import io
import math

import numpy as np

TILE = 1.6        # m per regolith texture tile


def _fft_noise(n, beta, rng):
    k = np.fft.fftfreq(n)
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    kk[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / kk ** (beta / 2)
    spec[0, 0] = 0
    f = np.fft.ifft2(spec).real
    return ((f - f.mean()) / f.std()).astype(np.float32)


def regolith_png(sun, rng, n=2048):
    """Tileable albedo with grains + baked micro-shading for the fixed sun direction. Returns PNG bytes."""
    import cv2
    alb = 0.42 + 0.04 * _fft_noise(n, 2.4, rng) + 0.03 * cv2.GaussianBlur(
        rng.normal(size=(n, n)).astype(np.float32), (0, 0), 0.8)
    h = 0.5 * _fft_noise(n, 2.2, rng)
    bump = np.zeros((n, n), np.float32)
    cnt = 30000
    xs, ys = rng.integers(0, n, cnt), rng.integers(0, n, cnt)
    rs = rng.choice([1, 1, 1, 1, 2, 2, 2, 3, 4], cnt)
    tones = np.clip(0.42 + rng.normal(0, 0.06, cnt), 0.25, 0.7)
    metal = rng.random(cnt) < 0.04                           # a few bright FeNi grains
    tones[metal] = 0.75
    for x, y, r, t in zip(xs, ys, rs, tones):
        cv2.circle(alb, (int(x), int(y)), int(r), float(t), -1, cv2.LINE_AA)
        cv2.circle(bump, (int(x), int(y)), int(r), 0.35 * float(r), -1, cv2.LINE_AA)
    bump = cv2.GaussianBlur(bump, (0, 0), 1.0)
    for _ in range(2):
        bump = cv2.GaussianBlur(bump, (0, 0), 0.8) + 0.5 * bump
    h += 0.5 * bump
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    gy = -(np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    L = np.asarray(sun, float) / np.linalg.norm(sun)       # direction TO the sun
    shade = (L[2] - 0.35 * (gx * L[0] + gy * L[1])) / max(L[2], 0.2)
    img = np.clip(alb * np.clip(shade, 0.35, 1.6), 0, 1)
    rgb = np.stack([img * 0.90, img * 0.96, img * 1.0], -1)    # BGR: slight red slope
    ok, buf = cv2.imencode(".png", (rgb * 255).clip(0, 255).astype(np.uint8))
    return buf.tobytes()


def rock_png(rng, n=512):
    import cv2
    t = 0.5 + 0.10 * _fft_noise(n, 2.0, rng) + 0.05 * rng.normal(size=(n, n)).astype(np.float32)
    for _ in range(700):
        x, y = rng.integers(0, n, 2)
        cv2.circle(t, (int(x), int(y)), int(rng.integers(1, 4)), float(rng.uniform(0.3, 0.8)), -1)
    t = cv2.GaussianBlur(t, (0, 0), 0.7)
    ok, buf = cv2.imencode(".png", (np.stack([t * 0.93, t * 0.97, t], -1) * 255).clip(0, 255).astype(np.uint8))
    return buf.tobytes()


def _obj(V, F, UV):
    s = io.StringIO()
    np.savetxt(s, V, fmt="v %.5f %.5f %.5f")
    np.savetxt(s, UV, fmt="vt %.5f %.5f")
    np.savetxt(s, np.c_[F[:, 0], F[:, 0], F[:, 1], F[:, 1], F[:, 2], F[:, 2]] + 1, fmt="f %d/%d %d/%d %d/%d")
    return s.getvalue().encode()


def terrain_obj(xs, Z, zmin):
    """The physics heightfield surface as a smooth mesh (same samples, so the cube sits exactly on it)."""
    n = len(xs)
    X, Y = np.meshgrid(xs, xs)
    V = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, 1:].ravel(), i[1:, :-1].ravel()
    F = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    return _obj(V, F, V[:, :2] / TILE)


def _icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    V = [np.array(v, float) / np.linalg.norm(v) for v in
         [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
          (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        cache, F2 = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    return np.array(V), np.array(F)


def _rock(rng, level):
    """Angular rock of unit radius: a sphere projected onto a random convex polytope, plus roughness."""
    V, F = _icosphere(level)
    k = rng.integers(12, 22)
    N = rng.normal(size=(k, 3))
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    off = rng.uniform(0.75, 1.0, k)
    dn = V @ N.T
    t = np.where(dn > 1e-6, off / np.maximum(dn, 1e-6), np.inf).min(1)
    P = V * np.minimum(t, 1.2)[:, None] * (1 + 0.02 * rng.normal(size=(len(V), 1)))
    P /= np.abs(P).max(0)                                   # unit half-extent on each axis
    return P, F


def _rot(yaw, tilt_x=0.0, tilt_y=0.0):
    cz, sz, cx, sx, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(tilt_x), math.sin(tilt_x), math.cos(tilt_y), math.sin(tilt_y)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rz @ Ry @ Rx


def rocks_obj(boulders, xs, Z, height, rng, n_pebbles=3000, patch=16.0):
    """One merged mesh (one draw call, one BVH): a faceted rock over every collision ellipsoid + pebbles."""
    shapes = [_rock(rng, 3) for _ in range(8)] + [_rock(rng, 1) for _ in range(6)]
    parts = []
    for b in boulders:                                      # x, y, z, sx, sy, sz, yaw  (the collision ellipsoid)
        x, y, z, sx, sy, sz, yaw = b
        P, F = shapes[rng.integers(8)]
        R = _rot(yaw)
        parts.append(((P * np.array([sx, sy, sz]) * 1.04) @ R.T + [x, y, z], F))
    for _ in range(n_pebbles):                              # visual-only pebbles, mostly 3-15 mm
        r = 0.003 * (1 - rng.random()) ** -0.7
        if r > 0.015:
            continue
        x, y = rng.uniform(-patch / 2 + 0.3, patch / 2 - 0.3, 2)
        if math.hypot(x, y) < 0.25:                          # keep the spawn spot clean
            continue
        P, F = shapes[8 + rng.integers(6)]
        R = _rot(rng.uniform(0, 2 * math.pi), rng.normal(0, 0.3), rng.normal(0, 0.3))
        s = r * np.array([1.0, rng.uniform(0.6, 1.0), rng.uniform(0.4, 0.7)])
        parts.append(((P * s) @ R.T + [x, y, height(x, y) + 0.2 * s[2]], F))
    V = np.concatenate([p for p, _ in parts])
    offs = np.cumsum([0] + [len(p) for p, _ in parts[:-1]])
    F = np.concatenate([f + o for (_, f), o in zip(parts, offs)])
    UV = np.c_[2.0 * (V[:, 0] + 0.5 * V[:, 2]), 2.0 * (V[:, 1] + 0.5 * V[:, 2])]
    return _obj(V, F, UV)


def _cached(path, make):
    import os
    if os.path.exists(path):
        return open(path, "rb").read()
    data = make()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def build(xs, Z, zmin, boulders, sun, height, seed, cache_dir):
    """-> (asset xml, world geom xml, {asset name: bytes}). Textures are cached (they take a few seconds)."""
    import os
    rng = np.random.default_rng(seed + 11)
    tag = f"{seed}_{int(round(math.degrees(math.atan2(sun[1], sun[0]))))}_{int(round(math.degrees(math.asin(sun[2]))))}"
    assets = {"vis_terrain.obj": terrain_obj(xs, Z, zmin), "vis_rocks.obj": rocks_obj(boulders, xs, Z, height, rng),
              "regolith.png": _cached(os.path.join(cache_dir, f"regolith_{tag}.png"),
                                      lambda: regolith_png(sun, np.random.default_rng(seed + 12))),
              "rock.png": _cached(os.path.join(cache_dir, f"rock_{seed}.png"),
                                  lambda: rock_png(np.random.default_rng(seed + 13)))}
    asset_xml = """
    <texture name="regtex" type="2d" file="regolith.png"/>
    <texture name="rocktex" type="2d" file="rock.png"/>
    <material name="vis_regolith" texture="regtex" specular="0.06" shininess="0.08"/>
    <material name="vis_rock" texture="rocktex" rgba="0.72 0.70 0.67 1" specular="0.25" shininess="0.25"/>
    <mesh name="vis_terrain" file="vis_terrain.obj" inertia="shell"/>
    <mesh name="vis_rocks" file="vis_rocks.obj" inertia="shell"/>"""
    geom_xml = """
    <geom name="vis_terrain" type="mesh" mesh="vis_terrain" material="vis_regolith" contype="0" conaffinity="0" group="0"/>
    <geom name="vis_rocks" type="mesh" mesh="vis_rocks" material="vis_rock" contype="0" conaffinity="0" group="0"/>"""
    return asset_xml, geom_xml, assets
