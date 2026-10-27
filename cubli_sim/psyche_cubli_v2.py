"""
PSYCHE CUBLI v2  -  the AURORA cube (from the team CAD) on the surface of 16 Psyche.

    mjpython psyche_cubli_v2.py                 # 3D window (Mac needs mjpython for the window)
    mjpython psyche_cubli_v2.py --site north_pole
    mjpython psyche_cubli_v2.py --slope 15      # add a 15 deg regional slope (real Psyche reaches ~21 deg)
    python   psyche_cubli_v2.py sites           # gravity + slope at each landing site
    python   psyche_cubli_v2.py report          # headless: plots + renders into psyche_report_v2/
    python   psyche_cubli_v2.py test            # headless: scripted rolls, prints what happened

CONTROLS (in the 3D window, directions are relative to the camera)
    Arrow keys   roll one face that way (spin wheel up, brake, tip over the edge)
    Enter        jump up and BALANCE on the edge in front (LQR, like the 1-axis rig)
    Space        all-out hop forward (full wheel speed + hard brake)
    T            send the cube to where the camera is looking (press F first to free the camera)
    F            camera: follow the cube / free (drag to look around, right-drag to pan)
    G            next gravity: the Psyche sites, then Moon, then Earth
    P            pause          - / =  slower / faster       R  reset        H  status text on/off
    S            sensor head deploy / stow: the mast pushes the Mid-360 + OAK-D out of whatever face is on top
                 (fake gimbal, always level) and both switch on when it is out. L does the same.
    O / C        LiDAR on / off, camera on / off (while deployed)   J / ;  camera pan       I / K  camera tilt
    B            photoreal live view: Blender (EEVEE) follows the sim, same look as the film
    Dashboard    a second window (dashboard.py) shows camera, depth, LiDAR 3D / top, 3rd person and the team's
                 detection display; it opens by itself (CUBLI_NO_DASHBOARD=1 to skip) and forwards keys here
    Right panel  Control sliders for the mast, pan and tilt
    Rolls, hops and letting go of a balance stow the sensors first (shown in the sidebar).

What is real and what is a guess is marked on every number below
(CAD = measured from the assembly STL, DATASHEET, PAPER, YOUR NUMBER, PLACEHOLDER).
Needs the assets/ folder next to this file (make it with build_assets.py).
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# =====================================================================
# 16 PSYCHE  (Shepard et al. 2021, Planet. Sci. J. 2:125, radar + adaptive optics model)
# =====================================================================
PSYCHE_AXES_KM = (278.0, 238.0, 171.0)   # PAPER: 278 x 238 x 171 km triaxial dimensions
PSYCHE_MASS = 2.287e19                   # PAPER: 22.87e18 kg
PSYCHE_PERIOD_H = 4.195948               # PAPER: spin period
G_CONST = 6.674e-11
G_MOON, G_EARTH = 1.62, 9.81

# Landing sites (planetocentric lat, lon on the ellipsoid; spin axis = short axis).
# Feature names/places follow Shepard 2021; their longitudes are approximate on this ellipsoid.
SITES = collections.OrderedDict([
    ("equator_long_end", dict(lat=0.0, lon=0.0, note="tip of the long axis: weakest gravity, most spin")),
    ("equator_broad", dict(lat=0.0, lon=90.0, note="flat 'Charlie' side of the equator")),
    ("panthia", dict(lat=40.0, lon=300.0, note="shallow 80-90 km crater, radar-bright")),
    ("golf_south", dict(lat=-45.0, lon=30.0, note="southern mid-latitudes: steepest effective slopes")),
    ("eros_crater", dict(lat=-65.0, lon=260.0, note="50-75 km wide, ~4 km deep crater")),
    ("north_pole", dict(lat=89.0, lon=0.0, note="'Foxtrot' crater, strongest gravity, no spin effect")),
])
DEFAULT_SITE = "equator_broad"

# =====================================================================
# THE CUBE  (geometry + where the mass sits come from your CAD; masses below)
# =====================================================================
TOTAL_MASS = 0.900        # kg   YOUR NUMBER ("roughly 900 g": 3 motors, ESP32, breadboard, wires, batteries, printed sides)
MOTOR_MASS = 0.072        # kg   DATASHEET: iFlight GM3506 72 g (64 g bare; 80 g with AS5048A encoder on RobotShop)
ROTOR_MASS = 0.025        # kg   PLACEHOLDER: part of each motor that spins (bell + magnets)
WHEEL_MASS = 0.033        # kg   CAD 26.6 cc x PLA 1.24 g/cc = 100% infill. WEIGH ONE and put the real number here
PAYLOAD_MASS = 0.250      # kg   PLACEHOLDER: battery + ESP32 + breadboard + wires (sits in the CAD's electronics bay)
# everything else (printed panels, frame, mounts, screws) = TOTAL - the above

# =====================================================================
# SENSOR HEAD ON A FAKE GIMBAL: one head (Mid-360 with the OAK-D Lite on a pan/tilt turret on top) rides a slide
# mast on a gimbal at the cube centre. The gimbal is held level and turned to the travel direction whatever face the
# cube is on, so the head always comes out of the TOP face. FAKE: the CAD has no room for it (the head passes
# through a reaction wheel when a wheel face is up) and the levelling is kinematic (no gimbal motors are modelled).
# =====================================================================
LIDAR_MASS = 0.265        # kg   DATASHEET: Livox Mid-360, 65 x 65 x 60 mm
LIDAR_SIZE = (0.060, 0.065, 0.065)   # m  DATASHEET: height (along the mast) x width x depth
MAST_MASS = 0.030         # kg   PLACEHOLDER: slide rail + carriage + lead screw (+ the gimbal ring)
MAST_STROKE = 0.150       # m    PLACEHOLDER: full travel. The mast goes only as high as the pose needs: Mid-360
WINDOW_CLEAR = 0.019      # m    window this far above the cube's highest point, so its lowest beam (-7 deg) clears
                          #      the top face corners (on a face: 115 mm of mast; balancing on an edge: 146 mm)
MAST_SPEED = 0.03         # m/s  YOUR NUMBER: ~3 cm/s linear actuator
MAST_FORCE = 6.0          # N    PLACEHOLDER: actuator force limit
LIDAR_STOW_Z = -0.032     # m    PLACEHOLDER: stowed Mid-360 centre below the cube centre (whole head inside the cube)
GIMBAL_YAW_RATE = math.radians(90)   # rad/s PLACEHOLDER: how fast the gimbal turns the head to a new heading
CAM_MASS = 0.061          # kg   DATASHEET: Luxonis OAK-D Lite, 91 x 28 x 17.5 mm
CAM_SIZE = (0.028, 0.091, 0.0175)    # m  DATASHEET: height x width x depth
PANTILT_MASS = 0.010      # kg   PLACEHOLDER: pan/tilt bracket (servo masses inside this number)
CAM_FWD = 0.020           # m    PLACEHOLDER: camera ahead of the mast axis on the turret
CAM_RISE = 0.022          # m    PLACEHOLDER: post under the camera. With it the OAK-D sits 78 mm above the top face, so
                          #      even a cube corner stays below the bottom of the image (15 deg tilt, 55 deg V-FOV)
SERVO_SPEED = math.radians(60) / 0.15   # rad/s YOUR NUMBER: ~0.15 s per 60 deg (hobby servo class)
SERVO_TORQUE = 0.2        # N*m  PLACEHOLDER: micro-servo stall torque class
CAM_TILT_DEPLOYED = math.radians(-15)  # YOUR NUMBER: 15 deg down when deployed
SENSORS_IN_TOTAL_MASS = False   # PLACEHOLDER flag: does TOTAL_MASS (900 g) already include LiDAR, camera and
                                # their mechanisms? False = they add on top (900 g was "motors, ESP32, breadboard,
                                # wires, batteries, printed sides"). True = they come out of the "everything else" share.
SENSOR_MASS = LIDAR_MASS + MAST_MASS + CAM_MASS + PANTILT_MASS

# =====================================================================
# MOTOR: iPower GM3506 gimbal motor, driven with FOC (SimpleFOC-style)
# =====================================================================
V_BUS = 14.8              # V    PLACEHOLDER (needs your decision): 4S LiPo. Was YOUR NUMBER 11.1 V (3S). With the
                          #      Mid-360 + OAK-D Lite + mechanisms on board (1.29 kg) the 3S cube cannot roll anywhere on
                          #      Psyche (needs 1452-1730 rpm, motor max 1379 rpm at 11.1 V). The GM3506 KV is rated at a
                          #      16 V test, so 4S is inside the datasheet. Set back to 11.1 to see the 3S result.
R_PHASE = 5.6             # ohm  DATASHEET 5.6 +-5%
KV = 2262.0 / 16.0        # rpm/V DATASHEET: 2149-2375 rpm no-load at 16 V (midpoint) -> about 1,570 rpm at 11.1 V
I_LIMIT = 2.0             # A    PLACEHOLDER: driver current limit (5.6 ohm already caps it near 2 A at 11.1 V)
I_NOLOAD = 0.17           # A    DATASHEET: no-load current -> bearing/iron losses
MECH_BRAKE = 0.0          # N*m  0 = no mechanical brake (braking is done electrically by the motor)

# =====================================================================
# CONTROLLER (what would run on the ESP32)
# =====================================================================
CTRL_HZ = 200             # Hz   ESP32 control loop rate
IMU_NOISE_DEG = 0.05      # deg  PLACEHOLDER: BNO055-class angle noise
GYRO_NOISE_DPS = 0.3      # deg/s PLACEHOLDER
SPIN_FRACTION = 0.35      #      spin-up torque as a fraction of the torque that would tip the cube backwards
ROLL_MARGIN = 1.10        #      wheel speed = this x the physics minimum to reach the edge (auto-raised on a failed roll)

# =====================================================================
# SURFACE
# =====================================================================
PATCH = 16.0              # m    local terrain patch simulated in full detail
GRID = 512                #      -> 3.1 cm cells (the cube is 15.2 cm)
HORIZON = 1500.0          # m    curved far-field that shows the real horizon (about 130 m away at cube height)
SEED = 7
N_CRATERS = 70
N_BOULDERS = 160
NEAR_FRACTION = 0.6       #      share of the random boulders put 1.2-7 m from the start (what the sensors can see)
# ARTISTIC: showcase features in view of the start pose, so the LiDAR and the camera have craters and rocks of every
# size to show (the camera starts looking along +x). No Psyche surface images exist yet; sizes follow the random ones.
SHOWCASE_CRATERS = [(3.6, 0.4, 2.0), (1.9, -1.4, 0.9), (-1.9, 2.4, 1.4), (-5.0, -3.5, 4.0), (5.5, -3.8, 1.6)]
SHOWCASE_BOULDERS = [(2.3, 1.0, 0.22), (1.5, -0.4, 0.09), (1.75, -0.2, 0.06), (2.9, -0.9, 0.14), (4.7, 1.9, 0.40),
                     (-1.6, -1.4, 0.18), (0.4, 2.2, 0.12), (-2.8, 0.6, 0.30), (1.3, 0.55, 0.05), (2.6, 2.6, 0.10)]
REG_FRICTION = 0.7        #      PLACEHOLDER: regolith friction (Psyche regolith is unmeasured; lunar-like value)
REG_SOLREF = "0.03 2"     #      PLACEHOLDER: contact time-constant / damping ratio. Porous regolith soaks up impacts;
REG_SOLIMP = "0.9 0.99 0.003"   #   was "0.01 1": that spring hands back ~13% of the impact speed, which at Psyche
                          #      gravity floats the cube for tenths of a second. Measured on 4 rolls (tools/contact_sweep.py):
                          #      airborne phases 180 -> 21, micro-hops 1532 -> 591, rolls/balance/release still pass.
FINE_RMS = 0.0005         # m    PLACEHOLDER: mm-scale roughness in the PHYSICS heightfield (was 3 mm: on 3.1 cm cells it
                          #      flips the contact normal under the corner spheres = chatter). The visible detail is in
                          #      the visual terrain mesh / textures instead.
EXTRA_SLOPE_DEG = 0.0     # deg  extra regional slope on top of the ellipsoid's own tilt. The smooth ellipsoid only
                          #      gives 0-4 deg; Psyche's real topography gives up to ~21 deg (84% of it < 10 deg)
                          #      (Ferrais et al. 2020, Dynamical environment of 16 Psyche). Try 5, 10, 20.
SUN_ELEV_DEG = 22.0       #      sun height above the horizon (low sun = long shadows)
SUN_AZ_DEG = 140.0
DT = 0.001                # s    physics step
# =====================================================================

KT = 60.0 / (2 * math.pi * KV)          # N*m/A  (= back-EMF constant V*s/rad)
W_NOLOAD = V_BUS / KT                   # rad/s
FRIC_COULOMB = KT * I_NOLOAD * 0.35     # N*m   bearing friction (35% of the no-load loss)
FRIC_VISC = KT * I_NOLOAD * 0.65 / (16.0 / KT)   # N*m/(rad/s) rest of the no-load loss, at the 16 V test speed
GLFW = dict(UP=265, DOWN=264, LEFT=263, RIGHT=262, ENTER=257, SPACE=32, T=84, F=70, G=71, P=80,
            R=82, H=72, MINUS=45, EQUAL=61, V=86, L=76, C=67, I=73, K=75, J=74, SEMICOLON=59, O=79, N=78, B=66,
            S=83)


def rpm(w):
    return w * 60.0 / (2 * math.pi)


# ---------------------------------------------------------------------
# PSYCHE GRAVITY: exact field of a uniform triaxial ellipsoid + spin
# ---------------------------------------------------------------------
def psyche_axes():
    return tuple(1e3 * k / 2.0 for k in PSYCHE_AXES_KM)


def psyche_density():
    a, b, c = psyche_axes()
    return PSYCHE_MASS / (4.0 / 3.0 * math.pi * a * b * c)


def gravity_at(P):
    """Acceleration (gravity + centrifugal) at a point on/inside the ellipsoid, body frame, m/s^2."""
    from scipy.special import elliprd
    a, b, c = psyche_axes()
    k = -4.0 / 3.0 * math.pi * G_CONST * psyche_density() * a * b * c
    g = np.array([k * P[0] * elliprd(b * b, c * c, a * a),
                  k * P[1] * elliprd(a * a, c * c, b * b),
                  k * P[2] * elliprd(a * a, b * b, c * c)])
    w = 2 * math.pi / (PSYCHE_PERIOD_H * 3600.0)
    return g + w * w * np.array([P[0], P[1], 0.0])


def site_frame(lat, lon):
    """Surface point, local axes (east, north, up=surface normal) and effective gravity there."""
    a, b, c = psyche_axes()
    la, lo = math.radians(lat), math.radians(lon)
    u = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    P = u / math.sqrt((u[0] / a) ** 2 + (u[1] / b) ** 2 + (u[2] / c) ** 2)
    n = np.array([P[0] / a ** 2, P[1] / b ** 2, P[2] / c ** 2])
    n /= np.linalg.norm(n)
    east = np.cross([0.0, 0.0, 1.0], n)
    if np.linalg.norm(east) < 1e-6:
        east = np.array([0.0, 1.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(n, east)
    g = gravity_at(P)
    g_local = np.array([g @ east, g @ north, g @ n])
    slope = math.degrees(math.acos(np.clip(-g_local[2] / np.linalg.norm(g_local), -1, 1)))
    return dict(P=P, up=n, east=east, north=north, g=g, g_local=g_local,
                g_mag=float(np.linalg.norm(g)), slope_deg=slope, radius_km=float(np.linalg.norm(P) / 1e3))


def surface_drop(fr, x, y):
    """Exact height of the ellipsoid surface above the local tangent plane at local (x, y), metres (<0)."""
    a, b, c = psyche_axes()
    inv = np.array([1 / a ** 2, 1 / b ** 2, 1 / c ** 2])
    Q = fr["P"][None, :] + np.asarray(x).reshape(-1, 1) * fr["east"] + np.asarray(y).reshape(-1, 1) * fr["north"]
    n = fr["up"]
    A = (n * n * inv).sum()
    B = 2 * (Q * n * inv).sum(axis=1)
    C = (Q * Q * inv).sum(axis=1) - 1.0
    t = (-B + np.sqrt(np.maximum(B * B - 4 * A * C, 0))) / (2 * A)
    return t.reshape(np.shape(x))


def gravity_presets(site):
    out = []
    for name, s in SITES.items():
        fr = site_frame(s["lat"], s["lon"])
        g = fr["g_local"]
        if EXTRA_SLOPE_DEG:
            a = math.radians(EXTRA_SLOPE_DEG)          # tip "down" toward -x: the ground slopes downhill along -x
            g = np.array([g[0] * math.cos(a) - g[2] * math.sin(a) * -1, g[1], g[2] * math.cos(a) + g[0] * math.sin(a)])
        slope = math.degrees(math.acos(np.clip(-g[2] / np.linalg.norm(g), -1, 1)))
        out.append((f"Psyche: {name}", g, slope))
    out.append(("Moon", np.array([0, 0, -G_MOON]), 0.0))
    out.append(("Earth", np.array([0, 0, -G_EARTH]), 0.0))
    names = [o[0] for o in out]
    return out, names.index(f"Psyche: {site}")


# ---------------------------------------------------------------------
# MASS MODEL from the CAD split
# ---------------------------------------------------------------------
def load_mass_model():
    with open(os.path.join(ASSETS, "mass_parts.json")) as f:
        J = json.load(f)
    cls = J["classes"]
    struct = TOTAL_MASS - 3 * MOTOR_MASS - 3 * WHEEL_MASS - PAYLOAD_MASS
    if SENSORS_IN_TOTAL_MASS:
        struct -= SENSOR_MASS                  # the sensors are separate bodies; their mass leaves this share
    if struct <= 0:
        raise ValueError(f"TOTAL_MASS is smaller than motors + wheels + payload"
                         f"{' + sensors' if SENSORS_IN_TOTAL_MASS else ''} (struct would be {struct*1000:.0f} g)")
    want = {"struct": struct, "payload": PAYLOAD_MASS}
    for k in range(3):
        want[f"motor{k}"] = MOTOR_MASS - ROTOR_MASS

    def combine(names):
        M, S1, S2 = 0.0, np.zeros(3), np.zeros((3, 3))
        for nm in names:
            c = cls[nm]
            rho = masses[nm] / c["V"]
            M += masses[nm]
            S1 += rho * np.array(c["S1"])
            S2 += rho * np.array(c["S2"])
        com = S1 / M
        I = S2 - M * (com @ com * np.eye(3) - np.outer(com, com))   # move to the centre of mass
        return M, com, I

    masses = dict(want)
    for k in range(3):
        masses[f"wheel{k}"] = WHEEL_MASS
        masses[f"rotor{k}"] = ROTOR_MASS
    body = combine(["struct", "payload", "motor0", "motor1", "motor2"])
    wheels = []
    for w in J["wheel_axes"]:
        ax, sgn = w["axis"], w["sign"]
        M, com, I = combine([f"wheel{ax}", f"rotor{ax}"])
        e = np.zeros(3)
        e[ax] = sgn
        other = [i for i in range(3) if i != ax]
        wheels.append(dict(axis=ax, sign=sgn, e=e, pos=np.array(w["centre"]), mass=M, com=com,
                           I_axial=float(I[ax, ax]), I_trans=float((I[other[0], other[0]] + I[other[1], other[1]]) / 2)))
    return dict(edge=J["edge"], body=body, wheels=wheels, masses=masses)


# ---------------------------------------------------------------------
# TERRAIN
# ---------------------------------------------------------------------
def make_terrain(fr, seed=SEED, n=GRID, size=PATCH):
    """Cratered, porous metal-rich regolith on the real curvature of Psyche. Statistical, NOT a map."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, xs)
    R0 = np.hypot(X, Y)
    k = np.fft.fftfreq(n, d=size / n)
    KX, KY = np.meshgrid(k, k)
    K = np.hypot(KX, KY)
    K[0, 0] = 1.0

    def noise(power, rms):
        s = K ** -power
        s[0, 0] = 0
        z = np.fft.ifft2(s * np.exp(2j * np.pi * rng.random((n, n)))).real
        return z / z.std() * rms

    calm = np.clip((R0 - 0.8) / 2.0, 0.25, 1.0)
    Z = noise(1.6, 0.05) * calm ** 2 + noise(1.1, FINE_RMS)
    diams = []
    while len(diams) < N_CRATERS:
        d = 0.4 * (1 - rng.random()) ** -0.5          # N(>D) ~ D^-2
        if d < 6.0:
            diams.append(d)
    craters = []
    placed = [(d, None) for d in sorted(diams, reverse=True)] + [(d, (x, y)) for x, y, d in SHOWCASE_CRATERS]
    for d, at in placed:
        if at is not None:
            cx, cy = at
        else:
            for _ in range(50):
                cx, cy = rng.uniform(-size / 2, size / 2, 2)
                if math.hypot(cx, cy) > d / 2 + 1.2:
                    break
        r = np.hypot(X - cx, Y - cy) / (d / 2)
        depth = d * (0.12 if at is not None else rng.uniform(0.08, 0.17))
        bowl = np.where(r < 1, depth * (r ** 2 - 1), 0.0)
        rim = 0.18 * depth * np.exp(-((r - 1.0) / 0.25) ** 2)
        Z += bowl + rim
        craters.append((cx, cy, d))
    Z += surface_drop(fr, X.ravel(), Y.ravel()).reshape(Z.shape)
    return xs, Z, craters


def terrain_h(xs, Z, x, y):
    n = len(xs)
    step = xs[1] - xs[0]
    fx = np.clip((x - xs[0]) / step, 0, n - 1.001)
    fy = np.clip((y - xs[0]) / step, 0, n - 1.001)
    i, j = int(fy), int(fx)
    tx, ty = fx - j, fy - i
    return float((1 - ty) * ((1 - tx) * Z[i, j] + tx * Z[i, j + 1]) + ty * ((1 - tx) * Z[i + 1, j] + tx * Z[i + 1, j + 1]))


def make_boulders(xs, Z, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    out = []

    def add(x, y, r):
        sx, sy, sz = r * rng.uniform(0.8, 1.25), r * rng.uniform(0.7, 1.1), r * rng.uniform(0.5, 0.8)
        z = terrain_h(xs, Z, x, y) + sz * 0.35
        out.append((x, y, z, sx, sy, sz, rng.uniform(0, math.pi)))
    for x, y, r in SHOWCASE_BOULDERS:
        add(x, y, r)
    while len(out) < N_BOULDERS + len(SHOWCASE_BOULDERS):
        r = 0.04 * (1 - rng.random()) ** -0.6
        if r > 0.45:
            continue
        if rng.random() < NEAR_FRACTION:
            a, rad = rng.uniform(0, 2 * math.pi), rng.uniform(1.2, 7.0)
            x, y = rad * math.cos(a), rad * math.sin(a)
        else:
            x, y = rng.uniform(-PATCH / 2 + 0.5, PATCH / 2 - 0.5, 2)
        if math.hypot(x, y) < 1.0 + r or any(math.hypot(x - b[0], y - b[1]) < r + max(b[3], b[4]) for b in out):
            continue
        add(x, y, r)
    return out


def stl_bytes(V, F):
    V = np.asarray(V, np.float32)
    tri = V[np.asarray(F)]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    return b"\0" * 80 + np.uint32(len(F)).tobytes() + rec.tobytes()


def horizon_mesh(fr, inner, outer, seed=SEED, arrays=False):
    """Ring of terrain from the patch edge out to the horizon, following the real curvature."""
    rng = np.random.default_rng(seed + 2)
    radii = np.geomspace(inner, outer, 70)
    angs = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    R, A = np.meshgrid(radii, angs, indexing="ij")
    X, Y = R * np.cos(A), R * np.sin(A)
    Zc = surface_drop(fr, X.ravel(), Y.ravel()).reshape(X.shape)
    bumps = sum(rng.uniform(0.2, 1.0) * np.sin(f * A + rng.uniform(0, 6.3)) for f in (3, 7, 13, 29))
    Z = Zc + 0.004 * R * bumps / 3.0 - 0.03
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    na = len(angs)
    F = []
    for i in range(len(radii) - 1):
        for j in range(na):
            a, b = i * na + j, i * na + (j + 1) % na
            c, d = a + na, b + na
            F += [(a, c, b), (b, c, d)]
    if arrays:
        return V, np.asarray(F)
    return stl_bytes(V, F)


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
EDGE_RADIUS = 0.005       # m    rounded cube edges used for ground contact (printed edges are never sharp)


def rounded_hull(h, r=EDGE_RADIUS):
    """Ground-contact hull: 8 corner spheres + 12 edge capsules (stable on a heightfield, unlike a box)."""
    c = h - r
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                out.append(f'      <geom type="sphere" size="{r}" pos="{sx*c:.4f} {sy*c:.4f} {sz*c:.4f}" '
                           f'contype="1" conaffinity="1" group="3" rgba="0 0 0 0" mass="0"/>')
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                p0, p1 = [0.0] * 3, [0.0] * 3
                p0[ax], p1[ax] = -c, c
                p0[o[0]] = p1[o[0]] = s1 * c
                p0[o[1]] = p1[o[1]] = s2 * c
                out.append(f'      <geom type="capsule" size="{r}" fromto="{p0[0]:.4f} {p0[1]:.4f} {p0[2]:.4f} '
                           f'{p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f}" contype="1" conaffinity="1" group="3" rgba="0 0 0 0" mass="0"/>')
    return "\n".join(out)


WHEEL_QUAT = {(2, 1): "1 0 0 0", (1, 1): "0.7071068 -0.7071068 0 0", (0, -1): "0.7071068 0 -0.7071068 0"}
START_QUAT = "0.7071068 0 -0.7071068 0"   # PLACEHOLDER pose: start with the wheel-free +X face up

# --- sensor head layout, gimbal frame (x = travel direction, z = up), m
LIDAR_WINDOW = 0.012             # optical centre above the housing centre (window band)
TURRET_GAP = 0.003               # pan bracket between the Mid-360 top and the camera
SENSOR_JOINTS = ("gimbal", "mast", "cam_pan", "cam_tilt")     # recorded in this order (gimbal = 4-value quaternion)
SENSOR_ACTUATORS = ("mast", "cam_pan", "cam_tilt")


def sensor_xml():
    """Sensor head on the fake gimbal: ball joint at the cube centre (levelled every step by SensorRig), slide mast
    along the gimbal's up axis, Mid-360 on the mast, OAK-D Lite on a pan/tilt turret on top of the Mid-360. Real
    bodies with mass, so the centre of mass moves when the head deploys. Group 1 = visible to the cube's own
    sensors (and to everything else)."""
    lh, lw = LIDAR_SIZE[0] / 2, LIDAR_SIZE[1] / 2
    ch, cw, cd = (s / 2 for s in CAM_SIZE)
    vis = 'contype="0" conaffinity="0" group="1" mass="0"'
    s = lambda m, x, y, z: f"{m * (y * y + z * z) / 3:.3e} {m * (x * x + z * z) / 3:.3e} {m * (x * x + y * y) / 3:.3e}"
    dmp = lambda kp, inert: 2 * 0.8 * math.sqrt(kp * inert)
    rail_lo, rail_hi = LIDAR_STOW_Z - lh, 0.068
    return f"""
      <body name="gimbal" pos="0 0 0">
        <joint name="gimbal" type="ball" damping="0" armature="0"/>
        <inertial pos="0 0 0" mass="{MAST_MASS}" diaginertia="2e-5 2e-5 1e-5"/>
        <geom name="gimbal_ring" type="cylinder" size="0.012 0.004" material="alu" {vis}/>
        <geom name="mast_rail" type="cylinder" size="0.004 {(rail_hi - rail_lo) / 2:.4f}" pos="0 0 {(rail_hi + rail_lo) / 2:.4f}" material="alu" {vis}/>
        <body name="lidar" pos="0 0 {LIDAR_STOW_Z}">
          <joint name="mast" type="slide" axis="0 0 1" range="0 {MAST_STROKE}" damping="{dmp(200, LIDAR_MASS):.2f}"/>
          <inertial pos="0 0 0" mass="{LIDAR_MASS}" diaginertia="{s(LIDAR_MASS, lw, lw, lh)}"/>
          <geom name="lidar_body" type="box" size="{lw} {lw} {lh}" material="lidar" group="1" contype="0" conaffinity="0"/>
          <geom name="lidar_window" type="box" size="{lw + 0.0004} {lw + 0.0004} 0.008" pos="0 0 {LIDAR_WINDOW}" material="glass" {vis}/>
          <site name="lidar_site" pos="0 0 {LIDAR_WINDOW}" size="0.004"/>
          <camera name="lidar_cam" pos="{lw + 0.001} 0 0" xyaxes="0 -1 0 0 0 1" fovy="55"/>
          <body name="cam_pan" pos="0 0 {lh + TURRET_GAP}">
            <joint name="cam_pan" type="hinge" axis="0 0 1" range="-1.5708 1.5708" damping="{dmp(0.03, 4e-5):.4f}" armature="1e-5"/>
            <inertial pos="0 0 0" mass="{PANTILT_MASS}" diaginertia="1e-6 1e-6 1e-6"/>
            <geom type="cylinder" size="0.010 {TURRET_GAP / 2}" pos="0 0 {-TURRET_GAP / 2}" material="alu" {vis}/>
            <geom type="box" size="{CAM_FWD / 2 + 0.003:.4f} 0.006 0.0015" pos="{CAM_FWD / 2:.4f} 0 0.0015" material="alu" {vis}/>
            <geom type="box" size="0.004 0.004 {CAM_RISE / 2}" pos="{CAM_FWD} 0 {0.002 + CAM_RISE / 2}" material="alu" {vis}/>
            <body name="cam_head" pos="{CAM_FWD} 0 {ch + 0.002 + CAM_RISE}">
              <joint name="cam_tilt" type="hinge" axis="0 -1 0" range="-0.8 0.6" damping="{dmp(0.03, 4e-5):.4f}" armature="1e-5"/>
              <inertial pos="0 0 0" mass="{CAM_MASS}" diaginertia="{s(CAM_MASS, cd, cw, ch)}"/>
              <geom name="oakd" type="box" size="{cd} {cw} {ch}" material="oakd" contype="0" conaffinity="0" group="1"/>
              <geom type="cylinder" size="0.0045 0.0006" pos="{cd} -0.0375 0" euler="0 1.5708 0" material="glass" {vis}/>
              <geom type="cylinder" size="0.0060 0.0006" pos="{cd} 0 0" euler="0 1.5708 0" material="glass" {vis}/>
              <geom type="cylinder" size="0.0045 0.0006" pos="{cd} 0.0375 0" euler="0 1.5708 0" material="glass" {vis}/>
              <camera name="cubli_cam" pos="{cd + 0.0012} 0 0" xyaxes="0 -1 0 0 0 1" fovy="55"/>
              <site name="cam_site" pos="{cd + 0.0012} 0 0" size="0.003"/>
            </body>
          </body>
        </body>
      </body>"""


def sensor_actuator_xml():
    # kp gains are PLACEHOLDER servo/actuator stiffnesses (tuned to settle without overshoot at 1 kHz);
    # force limits come from MAST_FORCE / SERVO_TORQUE above
    return f"""
    <position name="mast" joint="mast" kp="200" ctrlrange="0 {MAST_STROKE}" forcerange="{-MAST_FORCE} {MAST_FORCE}" group="1"/>
    <position name="cam_pan" joint="cam_pan" kp="0.03" ctrlrange="-1.5708 1.5708" forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}" group="1"/>
    <position name="cam_tilt" joint="cam_tilt" kp="0.03" ctrlrange="-0.8 0.6" forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}" group="1"/>"""


def build_world(site=DEFAULT_SITE, seed=SEED):
    s = SITES[site]
    fr = site_frame(s["lat"], s["lon"])
    mm = load_mass_model()
    xs, Z, craters = make_terrain(fr, seed)
    boulders = make_boulders(xs, Z, seed)
    zmin, zmax = float(Z.min()), float(Z.max())
    zr = max(zmax - zmin, 1e-3)
    h = mm["edge"] / 2
    M, com, I = mm["body"]
    g = fr["g_local"]
    sun = np.array([math.cos(math.radians(SUN_ELEV_DEG)) * math.cos(math.radians(SUN_AZ_DEG)),
                    math.cos(math.radians(SUN_ELEV_DEG)) * math.sin(math.radians(SUN_AZ_DEG)),
                    math.sin(math.radians(SUN_ELEV_DEG))])
    q0 = np.array([float(v) for v in START_QUAT.split()])
    R0 = np.zeros(9)
    mujoco.mju_quat2Mat(R0, q0)
    R0 = R0.reshape(3, 3)
    sun_l = R0.T @ sun                   # the light is in the cube body: give its direction in the body frame
    sp = 3.0 * sun_l
    jl_damp, jl_fric = FRIC_VISC, FRIC_COULOMB
    wheels_xml = []
    for k, w in enumerate(mm["wheels"]):
        R_lw = np.zeros(3)
        loc_com = w["com"] - w["pos"]
        R_lw[2] = loc_com @ w["e"]
        wheels_xml.append(f"""
      <body name="wheel{k}" pos="{w['pos'][0]:.6f} {w['pos'][1]:.6f} {w['pos'][2]:.6f}" quat="{WHEEL_QUAT[(w['axis'], w['sign'])]}">
        <joint name="w{k}" type="hinge" axis="0 0 1" damping="{jl_damp:.3e}" frictionloss="{jl_fric:.4f}" armature="0"/>
        <inertial pos="0 0 {R_lw[2]:.5f}" mass="{w['mass']:.5f}" diaginertia="{w['I_trans']:.4e} {w['I_trans']:.4e} {w['I_axial']:.4e}"/>
        <geom type="mesh" mesh="wheel" material="wheel" contype="0" conaffinity="0" group="1"/>
      </body>""")
    boulder_xml = "\n".join(
        f'    <geom type="ellipsoid" size="{b[3]:.3f} {b[4]:.3f} {b[5]:.3f}" pos="{b[0]:.3f} {b[1]:.3f} {b[2]:.3f}" '
        f'euler="0 0 {b[6]:.2f}" material="rock" contype="2" conaffinity="2" group="3"/>' for b in boulders)
    import psyche_visuals                       # drawn rocks/terrain over the (hidden) collision shapes
    vis_asset_xml, vis_geom_xml, vis_assets = psyche_visuals.build(
        xs, Z, zmin, boulders, sun, lambda x, y: terrain_h(xs, Z, x, y), seed, os.path.join(ASSETS, "visual_cache"))
    z0 = terrain_h(xs, Z, 0, 0) + h + 0.004
    hull_xml = rounded_hull(h)
    xml = f"""
<mujoco model="psyche_cubli_v2">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="10"
          gravity="{g[0]:.6f} {g[1]:.6f} {g[2]:.6f}"/>
  <statistic extent="2" center="0 0 0"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="45" azimuth="120" elevation="-12"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.002" zfar="900" shadowclip="1.2" shadowscale="0.6" haze="0"/>
    <headlight ambient="0.025 0.025 0.03" diffuse="0.06 0.06 0.06" specular="0 0 0"/>
    <rgba haze="0 0 0 1"/>
  </visual>
  <default>
    <geom friction="{REG_FRICTION} 0.01 0.0005" condim="4" solref="{REG_SOLREF}" solimp="{REG_SOLIMP}"/>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" mark="random"
             random="0.0012" markrgb="0.85 0.85 0.9" width="2048" height="2048"/>
    <texture name="reg" type="2d" builtin="flat" rgb1="0.43 0.41 0.39" mark="random" random="0.18"
             markrgb="0.33 0.315 0.30" width="1024" height="1024"/>
    <material name="regolith" texture="reg" texrepeat="40 40" specular="0.12" shininess="0.15" reflectance="0"/>
    <material name="far" rgba="0.34 0.33 0.31 1" specular="0.05"/>
    <material name="rock" rgba="0.50 0.49 0.47 1" specular="0.45" shininess="0.5"/>
    <material name="pla" rgba="0.80 0.80 0.78 1" specular="0.25" shininess="0.35"/>
    <material name="wheel" rgba="0.85 0.30 0.05 1" specular="0.3" shininess="0.4"/>
    <material name="bay" rgba="0.20 0.20 0.22 1" specular="0.2"/>
    <material name="hatch" rgba="0.80 0.80 0.78 1" specular="0.25" shininess="0.35"/>
    <material name="lidar" rgba="0.12 0.12 0.13 1" specular="0.4" shininess="0.4"/>
    <material name="glass" rgba="0.03 0.03 0.05 1" specular="1" shininess="1"/>
    <material name="oakd" rgba="0.18 0.18 0.20 1" specular="0.3"/>
    <material name="alu" rgba="0.70 0.71 0.74 1" specular="0.7" shininess="0.6"/>
    <mesh name="frame" file="frame.stl" inertia="shell"/>
    <mesh name="wheel" file="wheel.stl" inertia="shell"/>
    <mesh name="horizon" file="horizon.stl" inertia="shell"/>
    <hfield name="ground" nrow="{len(xs)}" ncol="{len(xs)}" size="{PATCH/2} {PATCH/2} {zr:.4f} 0.5"/>
{vis_asset_xml}
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" hfield="ground" pos="0 0 {zmin:.4f}" material="regolith" contype="1" conaffinity="1" group="3"/>
{vis_geom_xml}
    <geom name="far" type="mesh" mesh="horizon" material="far" contype="0" conaffinity="0"/>
{boulder_xml}
    <body name="cube" pos="0 0 {z0:.4f}" quat="{START_QUAT}">
      <freejoint name="cube"/>
      <light name="sun" mode="track" directional="true" castshadow="true" pos="{sp[0]:.3f} {sp[1]:.3f} {sp[2]:.3f}"
             dir="{-sun_l[0]:.4f} {-sun_l[1]:.4f} {-sun_l[2]:.4f}" diffuse="1.0 0.97 0.92" specular="0.3 0.3 0.3"/>
      <inertial pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}" mass="{M:.5f}"
                fullinertia="{I[0,0]:.5e} {I[1,1]:.5e} {I[2,2]:.5e} {I[0,1]:.5e} {I[0,2]:.5e} {I[1,2]:.5e}"/>
      <geom name="hull" type="box" size="{h:.4f} {h:.4f} {h:.4f}" rgba="0 0 0 0" group="3" contype="2" conaffinity="2"/>
{hull_xml}
      <geom type="mesh" mesh="frame" material="pla" contype="0" conaffinity="0" group="1"/>
      <site name="imu" pos="0 0 0" size="0.004"/>
{''.join(wheels_xml)}
{sensor_xml()}
    </body>
  </worldbody>
  <actuator>
    <motor name="m0" joint="w0" group="3"/> <motor name="m1" joint="w1" group="3"/> <motor name="m2" joint="w2" group="3"/>
{sensor_actuator_xml()}
  </actuator>
</mujoco>"""
    assets = {}
    for nm in ("frame.stl", "wheel.stl"):
        with open(os.path.join(ASSETS, nm), "rb") as f:
            assets[nm] = f.read()
    assets["horizon.stl"] = horizon_mesh(fr, PATCH * 0.47, HORIZON, seed)
    assets.update(vis_assets)
    m = mujoco.MjModel.from_xml_string(xml, assets)
    m.hfield_data[:] = ((Z - zmin) / zr).ravel()
    world = World(m, fr, site, mm, xs, Z)
    world.boulders, world.seed = boulders, seed
    return world


class SensorRig:
    """Deploy / stow / aim the sensor head. Targets are rate-limited (actuator speed) and sent to MuJoCo position
    actuators. Deploy: mast rises out of the top face, then the camera tilts to its look-down angle. Stow: camera
    back to level, then the mast comes down. The LiDAR and the camera switch on when the head is fully out and off
    when it starts to stow (O / C still switch each one by hand while deployed).

    Fake gimbal: every physics step the gimbal's ball joint is set so the head stays level and faces `heading`
    (world yaw, turned at GIMBAL_YAW_RATE), with the joint velocity that cancels the cube's own rotation. Nothing
    models the gimbal motors or their reaction torque."""
    TOL = dict(mast=0.002)

    def __init__(self, m, d, half):
        self.m, self.d, self.half = m, d, half
        self.mast_face = half + WINDOW_CLEAR - LIDAR_WINDOW - LIDAR_STOW_Z     # mast needed when resting on a face
        self.cube_id = m.body("cube").id
        self.act = {n: m.actuator(n).id for n in SENSOR_ACTUATORS}
        self.qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in SENSOR_JOINTS}
        self.gdof = m.jnt_dofadr[m.joint("gimbal").id]
        self.rate = dict(mast=MAST_SPEED, cam_pan=SERVO_SPEED, cam_tilt=SERVO_SPEED)
        self.reset()

    def reset(self):
        self.want_up = False
        self.lidar_on = self.cam_on = False
        self.lidar_epoch = getattr(self, "lidar_epoch", 0) + 1   # +1 on every LiDAR power-up: old points go
        self.target = {n: 0.0 for n in SENSOR_ACTUATORS}
        self.cmd = dict(self.target)
        self.written = dict(self.target)
        self.heading = self.yaw = 0.0
        self._was_deployed = False
        self.level(0.0)

    def q(self, n):
        return float(self.d.qpos[self.qadr[n]])

    # --- commands -------------------------------------------------------
    def deploy(self):
        self.want_up = True

    def stow(self):
        self.want_up = False

    def toggle(self):
        (self.stow if self.want_up else self.deploy)()

    deploy_lidar = deploy_cam = deploy               # one head: either name deploys both sensors
    stow_lidar = stow_cam = stow_all = stow

    def aim(self, dpan=0.0, dtilt=0.0):
        """Pan/tilt by a step (rad). Only while the head is out."""
        if not self.deployed:
            return False
        for n, dv in (("cam_pan", dpan), ("cam_tilt", dtilt)):
            lo, hi = self.m.actuator_ctrlrange[self.act[n]]
            self.target[n] = float(np.clip(self.target[n] + dv, lo, hi))
        return True

    def face(self, direction):
        """Turn the gimbal (head) toward a world direction (only its horizontal part counts)."""
        if math.hypot(direction[0], direction[1]) > 1e-6:
            self.heading = math.atan2(direction[1], direction[0])

    # --- state -----------------------------------------------------------
    @property
    def want_lidar(self):                            # kept for the dashboard state mirror
        return self.want_up

    def mast_goal(self):
        """Mast extension that puts the Mid-360 window WINDOW_CLEAR above the cube's highest corner right now."""
        R = self.d.xmat[self.cube_id].reshape(3, 3)
        top = self.half * np.abs(R[2]).sum()                 # highest corner above the cube centre
        return float(np.clip(top + WINDOW_CLEAR - LIDAR_WINDOW - LIDAR_STOW_Z, 0.0, MAST_STROKE))

    @property
    def deployed(self):
        return self.q("mast") > self.mast_face - self.TOL["mast"]

    lidar_deployed = cam_deployed = deployed

    @property
    def stowed(self):
        return not self.want_up and self.q("mast") < self.TOL["mast"]

    def state(self, which):
        frac = min(max(self.q("mast") / self.mast_face, 0.0), 1.0)
        if self.want_up:
            if frac <= 0.96:
                return f"DEPLOYING {100 * frac:3.0f}%"
            return "ON" if (self.lidar_on if which == "lidar" else self.cam_on) else "DEPLOYED (off)"
        return "STOWED" if frac < 0.04 else f"STOWING {100 * frac:3.0f}%"

    # --- per physics step -------------------------------------------------
    def level(self, dt):
        """Fake gimbal: head level, facing self.yaw (rate-limited toward self.heading)."""
        d = self.d
        err = (self.heading - self.yaw + math.pi) % (2 * math.pi) - math.pi
        step = GIMBAL_YAW_RATE * dt
        yaw_rate = float(np.clip(err, -step, step)) / dt if dt > 0 else 0.0
        self.yaw = self.heading if dt == 0 else self.yaw + yaw_rate * dt
        qt = np.array([math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)])
        qc_inv = np.zeros(4)
        mujoco.mju_negQuat(qc_inv, d.qpos[3:7] / np.linalg.norm(d.qpos[3:7]))
        q_rel = np.zeros(4)
        mujoco.mju_mulQuat(q_rel, qc_inv, qt)
        a = self.qadr["gimbal"]
        d.qpos[a:a + 4] = q_rel
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, d.qpos[3:7] / np.linalg.norm(d.qpos[3:7]))
        w_cube = Rc.reshape(3, 3) @ d.qvel[3:6]          # freejoint angular velocity is in the body frame
        Rt = np.zeros(9)
        mujoco.mju_quat2Mat(Rt, qt)
        d.qvel[self.gdof:self.gdof + 3] = Rt.reshape(3, 3).T @ (np.array([0.0, 0.0, yaw_rate]) - w_cube)

    def update(self, dt):
        for n, i in self.act.items():               # a slider drag in the viewer's Control panel
            if abs(self.d.ctrl[i] - self.written[n]) > 1e-9:
                v = float(self.d.ctrl[i])
                if n == "mast":
                    self.want_up = v > MAST_STROKE / 2
                else:
                    self.target[n] = v                # the command still moves at actuator speed
        if self.want_up:                             # sequence: mast up, then the camera looks down
            self.target["mast"] = self.mast_goal()
            if self.deployed and not self._was_deployed:
                self.target["cam_tilt"] = CAM_TILT_DEPLOYED
                self.lidar_on = self.cam_on = True   # both sensors start when the head is out
                self.lidar_epoch += 1
        else:
            self.lidar_on = self.cam_on = False
            self.target.update(cam_pan=0.0, cam_tilt=0.0)
            level = abs(self.q("cam_pan")) < math.radians(5) and abs(self.q("cam_tilt")) < math.radians(5)
            self.target["mast"] = 0.0 if level or self.q("mast") < self.mast_face - 0.01 else self.q("mast")
        self._was_deployed = self.deployed
        for n, i in self.act.items():
            step = self.rate[n] * dt
            self.cmd[n] += float(np.clip(self.target[n] - self.cmd[n], -step, step))
            self.d.ctrl[i] = self.cmd[n]
            self.written[n] = self.cmd[n]
        self.level(dt)


class World:
    def __init__(self, m, fr, site, mm, xs, Z):
        self.m, self.d = m, mujoco.MjData(m)
        self.fr, self.site, self.mm, self.xs, self.Z = fr, site, mm, xs, Z
        self.cube = m.body("cube").id
        self.hull = m.geom("hull").id
        self.half = mm["edge"] / 2
        self.qadr = [m.jnt_qposadr[m.joint(f"w{k}").id] for k in range(3)]
        self.vadr = [m.jnt_dofadr[m.joint(f"w{k}").id] for k in range(3)]
        self.wheel_e = [w["e"] for w in mm["wheels"]]
        self.I_w = [w["I_axial"] for w in mm["wheels"]]
        self.volts = np.zeros(3)
        self.brake = np.zeros(3)
        self.recorder = None
        self.presets, self.gi = gravity_presets(site)
        self.rig = SensorRig(m, self.d, mm["edge"] / 2)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.volts[:] = 0
        self.brake[:] = 0
        self.rig.reset()
        mujoco.mj_forward(self.m, self.d)

    @property
    def g(self):
        return np.array(self.m.opt.gravity)

    def set_gravity(self, i):
        self.gi = i % len(self.presets)
        self.m.opt.gravity[:] = self.presets[self.gi][1]

    def wheel_speed(self, k):
        return float(self.d.qvel[self.vadr[k]])

    def step(self):
        for k in range(3):
            w = self.wheel_speed(k)
            i = np.clip((self.volts[k] - KT * w) / R_PHASE, -I_LIMIT, I_LIMIT)
            self.d.ctrl[k] = KT * i - self.brake[k] * MECH_BRAKE * np.sign(w)
        self.rig.update(DT)
        mujoco.mj_step(self.m, self.d)
        if self.recorder is not None:
            self.recorder.tick()

    # --- geometry helpers -----------------------------------------
    def R(self):
        return self.d.xmat[self.cube].reshape(3, 3)

    def com(self):
        return self.d.subtree_com[self.cube].copy()

    def corners(self):
        h = self.half
        loc = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)])
        return self.d.xpos[self.cube] + loc @ self.R().T, loc

    def inertia_about(self, p):
        """Inertia tensor of cube + wheels about world point p (wheels counted as rigid)."""
        m, d = self.m, self.d
        I = np.zeros((3, 3))
        for b in range(self.cube, m.nbody):
            Rb = d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
            r = d.xipos[b] - p
            I += Ib + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
        return I

    def total_mass(self):
        return float(self.m.body_subtreemass[self.cube])

    def ang_vel(self):
        return self.d.qvel[3:6] @ self.R().T          # freejoint angular velocity is in the body frame

    def lin_vel(self):
        return self.d.qvel[0:3].copy()

    def height_at(self, x, y):
        return terrain_h(self.xs, self.Z, x, y)


# ---------------------------------------------------------------------
# CONTROLLER  (what would run on the ESP32)
# ---------------------------------------------------------------------
class Cubli:
    """Roll / balance / hop by spinning a wheel up and braking it, plus an LQR edge balancer."""

    def __init__(self, w):
        self.w = w
        self.rng = np.random.default_rng(3)
        self.queue = collections.deque()
        self.action = None
        self.goal = None
        self.margin = {}
        self.log = collections.deque(maxlen=12000)
        self.status = "ready"
        self.last_result = ""
        self.fails = 0
        self.next_ctrl = 0.0

    # --- public -----------------------------------------------------
    def roll(self, d):
        self.queue.append(("roll", np.asarray(d, float)))

    def hop(self, d):
        self.queue.append(("hop", np.asarray(d, float)))

    def balance(self, d):
        self.queue.append(("balance", np.asarray(d, float)))

    def go_to(self, xy):
        self.goal = np.asarray(xy, float)
        self.fails = 0

    def cancel(self):
        self.queue.clear()
        self.goal = None
        if self.action:
            self.action = dict(kind="stop", phase="brake", k=None, t0=self.w.d.time)
        self.status = "stopped"

    # --- planning -----------------------------------------------------
    def up(self):
        return -self.w.g / np.linalg.norm(self.w.g)

    def plan(self, kind, d):
        w = self.w
        up = self.up()
        R = w.R()
        dirs = []
        for ax in range(3):
            for s in (1, -1):
                a = s * R[:, ax]
                ah = a - (a @ up) * up
                if np.linalg.norm(ah) > 0.5:
                    dirs.append(ah / np.linalg.norm(ah))
        d = d - (d @ up) * up
        d /= np.linalg.norm(d) + 1e-12
        dvec = max(dirs, key=lambda v: v @ d)
        hax = np.cross(up, dvec)
        hax /= np.linalg.norm(hax)
        cw, loc = w.corners()
        hgt = cw @ up
        bottom = np.argsort(hgt)[:4]
        front = sorted(bottom, key=lambda i: -(cw[i] - w.com()) @ dvec)[:2]
        rear = sorted(bottom, key=lambda i: (cw[i] - w.com()) @ dvec)[:2]
        k = int(np.argmax([abs((R @ e) @ hax) for e in w.wheel_e]))
        s = float(np.sign((R @ w.wheel_e[k]) @ hax))
        pivot = cw[front].mean(0)
        need = self.needed_speed(pivot, hax, k)
        rear_arm = abs((w.com() - cw[rear].mean(0)) @ dvec)
        tau_tip = w.total_mass() * np.linalg.norm(w.g) * rear_arm
        w.rig.face(dvec)                              # the sensor head looks where the cube is going
        return dict(kind=kind, dvec=dvec, hax=hax, k=k, s=s, edge_loc=loc[front], need=need,
                    tau_spin=max(SPIN_FRACTION * tau_tip, 2e-4), start=w.com(), t0=w.d.time)

    def needed_speed(self, pivot, hax, k):
        """Minimum wheel speed so that braking it gives the cube enough momentum to reach the edge."""
        w = self.w
        g = w.g
        r = w.com() - pivot
        g_perp = g - (g @ hax) * hax
        r_perp = r - (r @ hax) * hax
        dU = w.total_mass() * (np.linalg.norm(g_perp) * np.linalg.norm(r_perp) + g_perp @ r_perp)
        I_h = hax @ w.inertia_about(pivot) @ hax
        omega = math.sqrt(2 * max(dU, 0) / I_h)
        return I_h * omega / w.I_w[k]

    def tilt(self, a):
        """Angle (rad) of the centre of mass past the balance point over the action's pivot edge."""
        w = self.w
        cw = w.d.xpos[w.cube] + a["edge_loc"] @ w.R().T
        pivot = cw.mean(0)
        hax = a["hax"]
        r = w.com() - pivot
        r -= (r @ hax) * hax
        g = w.g - (w.g @ hax) * hax
        u = -g / np.linalg.norm(g)
        return math.atan2(np.cross(u, r) @ hax, u @ r), pivot

    def lqr(self, a):
        from scipy.linalg import solve_continuous_are
        w = self.w
        _, pivot = self.tilt(a)
        hax = a["hax"]
        I_h = hax @ w.inertia_about(pivot) @ hax
        Iw = w.I_w[a["k"]]
        Ib = I_h - Iw
        g = w.g - (w.g @ hax) * hax
        r = w.com() - pivot
        r -= (r @ hax) * hax
        mgl = w.total_mass() * np.linalg.norm(g) * np.linalg.norm(r)
        A = np.array([[0, 1, 0], [mgl / Ib, 0, 0], [-mgl / Ib, 0, 0]])
        B = np.array([[0], [-1 / Ib], [1 / Iw + 1 / Ib]])
        Q = np.diag([1 / math.radians(1) ** 2, 1 / math.radians(10) ** 2, 1 / 200.0 ** 2])
        Rr = np.array([[1 / 0.05 ** 2]])
        P = solve_continuous_are(A, B, Q, Rr)
        return (np.linalg.solve(Rr, B.T @ P)).ravel()

    # --- control loop -------------------------------------------------
    def update(self):
        w = self.w
        t = w.d.time
        if t < self.next_ctrl:
            return
        self.next_ctrl = t + 1.0 / CTRL_HZ
        if self.action is None:
            self.next_action()
        a = self.action
        volts = np.zeros(3)
        w.brake[:] = 0
        hold = [k for k in range(3) if a is None or k != a.get("k")]
        for k in hold:                                    # idle wheels: gently hold zero speed
            volts[k] = self.torque_volts(k, -0.002 * w.wheel_speed(k), limit=0.004)
        if a is not None:
            volts[a["k"] if a.get("k") is not None else 0] = self.run_action(a, volts)
        w.volts[:] = volts
        self.log.append((t, *[rpm(w.wheel_speed(k)) for k in range(3)],
                         math.degrees(self.tilt(a)[0]) if a and a.get("k") is not None else float("nan"),
                         *w.com()))

    def friction_ff(self, k, direction):
        """Torque that just cancels the motor's own bearing friction (internal: it does not tip the cube)."""
        w = self.w.wheel_speed(k)
        return float(np.sign(direction)) * FRIC_COULOMB * 1.05 + FRIC_VISC * w

    def torque_volts(self, k, tau, limit=None):
        if limit is not None:
            tau = float(np.clip(tau, -limit, limit))
        v = R_PHASE * tau / KT + KT * self.w.wheel_speed(k)
        return float(np.clip(v, -V_BUS, V_BUS))

    def next_action(self):
        w = self.w
        if not self.queue and self.goal is not None:
            off = self.goal - w.com()[:2]
            if np.linalg.norm(off) < 0.13:
                self.status = "arrived at target"
                self.goal = None
            elif self.fails >= 4:
                self.status = "gave up on target: too steep / blocked here"
                self.goal = None
            else:
                self.queue.append(("roll", np.array([off[0], off[1], 0.0])))
        if self.queue and self.queue[0][0] in ("roll", "hop") and not w.rig.stowed:
            w.rig.stow_all()                       # rolling/hopping with a sensor out would smash it: stow first
            self.status = (f"{self.queue[0][0]} waits: stowing sensors first (LiDAR {w.rig.state('lidar')}, "
                           f"camera {w.rig.state('cam')}). Rolls and hops are blocked while deployed")
            return
        if self.queue:
            kind, d = self.queue.popleft()
            a = self.plan(kind, d)
            key = (w.gi, kind)
            mult = {"roll": self.margin.get(key, ROLL_MARGIN), "balance": self.margin.get(key, 1.03),
                    "hop": 99.0}[kind]
            top = 0.97 * W_NOLOAD - 15.0
            a["target"] = min(a["need"] * mult, top)
            a["phase"] = "spinup"
            a["short"] = a["need"] * (1.0 if kind != "roll" else 1.02) > top
            self.action = a
            self.status = (f"{kind}: spinning wheel {a['k']} to {rpm(a['target']):.0f} rpm "
                           f"(needs {rpm(a['need']):.0f}, motor max {rpm(top):.0f})")
            if a["short"] and kind != "hop":
                self.status += "  -- NOT ENOUGH WHEEL SPEED at this gravity"

    def run_action(self, a, volts):
        w = self.w
        t = w.d.time
        if a["kind"] == "stop":
            for j in range(3):
                volts[j] = self.torque_volts(j, -0.002 * w.wheel_speed(j), limit=0.003)
            if max(abs(w.wheel_speed(j)) for j in range(3)) < 5:
                self.action = None
                self.status = "stopped"
            return volts[0]
        k, s = a["k"], a["s"]
        om = s * w.wheel_speed(k)
        if a["phase"] == "spinup":
            err = a["target"] - om
            if err < 0.01 * a["target"] or t - a["t0"] > 30:
                a["phase"], a["t1"] = "brake", t
            acc = s * np.clip(2e-3 * err, -a["tau_spin"], a["tau_spin"])
            return self.torque_volts(k, acc + self.friction_ff(k, s))
        if a["phase"] == "brake":
            w.brake[k] = 1
            # done when nearly stopped, or when the speed crossed zero since the last tick: full-voltage braking
            # changes it by ~8 rad/s per 5 ms tick, so it can jump the +-4 window and chatter around zero forever
            crossed = a.get("om_prev") is not None and om * a["om_prev"] < 0
            a["om_prev"] = om
            if abs(om) < 4.0 or crossed:
                a["phase"], a["t2"] = ("catch" if a["kind"] == "balance" else "settle"), t
                a["calm_since"] = None
            return -s * np.sign(om) * V_BUS if abs(om) > 4.0 else 0.0
        if a["phase"] == "catch":
            th, _ = self.tilt(a)
            if abs(th) < math.radians(8):
                a["K"] = self.lqr(a)
                a["phase"] = "balance"
                self.status = "balancing on the edge (LQR). Press Enter again to let go"
            elif t - a["t2"] > 6 or th < -math.radians(60):
                self.finish(a, False, "did not reach the edge")
            return self.torque_volts(k, -0.004 * w.wheel_speed(k), limit=0.01)
        if a["phase"] == "balance":
            th, _ = self.tilt(a)
            if abs(th) > math.radians(25):
                self.finish(a, False, "fell off the edge")
                return 0.0
            th_m = th + math.radians(self.rng.normal(0, IMU_NOISE_DEG))
            dth = w.ang_vel() @ a["hax"] + math.radians(self.rng.normal(0, GYRO_NOISE_DPS))
            tau = -a["K"] @ np.array([th_m, dth, om])
            if a.get("release_when_stowed") and w.rig.stowed:
                a["release"] = True
                a.pop("release_when_stowed")
            if a.get("release"):
                tau = -0.02
                if th > math.radians(20):
                    a["phase"], a["t2"], a["calm_since"] = "settle", t, None
            return self.torque_volts(k, s * tau + self.friction_ff(k, s * tau) * 0.9)
        if a["phase"] == "settle":
            v = np.linalg.norm(w.lin_vel())
            om_c = np.linalg.norm(w.ang_vel())
            if v < 0.004 and om_c < 0.03:
                a["calm_since"] = a["calm_since"] or t
                if t - a["calm_since"] > 0.4:
                    moved = (w.com() - a["start"]) @ a["dvec"]
                    ok = moved > {"roll": 0.08, "balance": 0.04}.get(a["kind"], -1.0)
                    self.finish(a, ok, f"moved {moved*100:.0f} cm in {t - a['t0']:.1f} s")
            else:
                a["calm_since"] = None
            if t - a["t2"] > 40:
                self.finish(a, True, "timeout")
            return self.torque_volts(k, -0.004 * w.wheel_speed(k), limit=0.01)
        return 0.0

    def finish(self, a, ok, msg):
        key = (self.w.gi, a["kind"])
        if not ok and a["kind"] in ("roll", "balance"):
            self.margin[key] = min(self.margin.get(key, ROLL_MARGIN if a["kind"] == "roll" else 1.03) * 1.07, 3.0)
            self.fails += 1
            msg += f"; retry with {self.margin[key]:.2f}x"
        elif ok:
            self.fails = 0
        self.last_result = f"{a['kind']} {'OK' if ok else 'FAILED'}: {msg}"
        self.status = self.last_result
        self.action = None

    def release_balance(self):
        if self.action and self.action.get("phase") == "balance":
            if self.w.rig.stowed:
                self.action["release"] = True
            else:                                  # it falls onto a face: bring the sensors in first (LQR keeps
                self.w.rig.stow_all()              # balancing while the mass moves)
                self.action["release_when_stowed"] = True
                self.status = "release waits: stowing sensors first (the cube falls onto a face)"
            return True
        return False


# ---------------------------------------------------------------------
# RECORDING (for the cinematic Blender render: render_blender.py)
# ---------------------------------------------------------------------
REC_HZ = 120              # Hz   poses saved per SIM second (the renderer interpolates between them)


def rig_spec(m):
    """Geometry + joint layout of the deployable sensor bodies, so render_blender.py can rebuild and animate them."""
    out = []
    for name in ("gimbal", "lidar", "cam_pan", "cam_head"):
        b = m.body(name).id
        jid = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        jnt = None
        if jid:
            j = jid[0]
            kind = {int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
                    int(mujoco.mjtJoint.mjJNT_BALL): "ball"}.get(int(m.jnt_type[j]), "hinge")
            jnt = dict(name=m.joint(j).name, type=kind, axis=m.jnt_axis[j].tolist(), pos=m.jnt_pos[j].tolist())
        geoms = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            geoms.append(dict(type=int(m.geom_type[g]), size=m.geom_size[g].tolist(), pos=m.geom_pos[g].tolist(),
                              quat=m.geom_quat[g].tolist(), rgba=rgba.tolist(),
                              mat=m.material(m.geom_matid[g]).name if m.geom_matid[g] >= 0 else ""))
        out.append(dict(name=name, parent=m.body(m.body_parentid[b]).name, pos=m.body_pos[b].tolist(),
                        quat=m.body_quat[b].tolist(), joint=jnt, geoms=geoms))
    return out


def sensor_qpos(m, d):
    """Flat joint values of SENSOR_JOINTS (the ball-jointed gimbal gives 4: its quaternion)."""
    out = []
    for n in SENSOR_JOINTS:
        j = m.joint(n).id
        a = m.jnt_qposadr[j]
        out += list(d.qpos[a:a + (4 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1)])
    return out


class Recorder:
    """Saves the exact simulated motion + every ground impact, so Blender can re-render it photoreal."""

    def __init__(self, w):
        self.w = w
        self.t, self.pos, self.quat, self.wheel, self.sens = [], [], [], [], []
        self.events = []
        self.next = w.d.time
        self.last_event = -1.0
        self.prev_force = 0.0
        self.prev_v = 0.0
        self.hull = {g for g in range(w.m.ngeom) if w.m.geom_bodyid[g] == w.cube and w.m.geom_contype[g]}
        self.f6 = np.zeros(6)

    def tick(self):
        w, d = self.w, self.w.d
        force, pts, nrm = 0.0, [], np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in self.hull or c.geom2 in self.hull:
                mujoco.mj_contactForce(w.m, d, i, self.f6)
                force += abs(self.f6[0])
                pts.append(c.pos.copy())
                nrm += c.frame[:3] * (1 if c.geom2 in self.hull else -1)
        weight = w.total_mass() * np.linalg.norm(w.g)
        if force > 8 * weight and self.prev_force < 2 * weight and d.time - self.last_event > 0.3 and pts:
            n = nrm / (np.linalg.norm(nrm) + 1e-9)
            if n @ w.g > 0:
                n = -n
            self.events.append([d.time, *np.mean(pts, 0), *n, self.prev_v])
            self.last_event = d.time
        self.prev_force = force
        self.prev_v = float(np.linalg.norm(w.lin_vel()) + 0.076 * np.linalg.norm(w.ang_vel()))
        if d.time >= self.next:
            self.next += 1.0 / REC_HZ
            self.t.append(d.time)
            self.pos.append(d.qpos[0:3].copy())
            self.quat.append(d.qpos[3:7].copy())
            self.wheel.append([d.qpos[a] for a in w.qadr])
            self.sens.append(sensor_qpos(w.m, d))

    def save(self, path):
        w, m = self.w, self.w.m
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        V, F = horizon_mesh(w.fr, PATCH * 0.47, HORIZON, w.seed, arrays=True)
        wb = [m.body(f"wheel{k}").id for k in range(3)]
        sun = np.array([math.cos(math.radians(SUN_ELEV_DEG)) * math.cos(math.radians(SUN_AZ_DEG)),
                        math.cos(math.radians(SUN_ELEV_DEG)) * math.sin(math.radians(SUN_AZ_DEG)),
                        math.sin(math.radians(SUN_ELEV_DEG))])
        np.savez_compressed(
            path, t=np.array(self.t), pos=np.array(self.pos), quat=np.array(self.quat), wheel=np.array(self.wheel),
            events=np.array(self.events).reshape(-1, 8), xs=w.xs, Z=w.Z, boulders=np.array(w.boulders),
            horizon_v=V, horizon_f=F, sun=sun, gravity=np.array(w.g), site=np.array(w.site),
            gravity_name=np.array(w.presets[w.gi][0]), edge=np.array(w.mm["edge"]),
            wheel_pos=np.array([m.body_pos[b] for b in wb]), wheel_quat=np.array([m.body_quat[b] for b in wb]),
            psyche_axes_km=np.array(PSYCHE_AXES_KM), site_latlon=np.array([SITES[w.site]["lat"], SITES[w.site]["lon"]]),
            sensor_q=np.array(self.sens), sensor_joints=np.array(SENSOR_JOINTS), sensor_rig=np.array(json.dumps(rig_spec(m))))
        print(f"saved {len(self.t)} poses ({self.t[-1] - self.t[0]:.1f} s sim), {len(self.events)} impacts -> {path}")


DEMO = [("roll", [1, 0, 0]), ("roll", [1, 0, 0]), ("roll", [0, 1, 0]), ("balance", [1, 0, 0]),
        ("wait", 8.0), ("release", None), ("roll", [0, -1, 0]), ("hop", [1, 0, 0])]
DEMO_DEPLOY = [("wait", 1.0), ("deploy", None), ("wait", 5.0), ("aim", (0.6, 0.0)),
               ("wait", 2.0), ("aim", (-1.2, 0.15)), ("wait", 2.5), ("aim", (0.6, -0.15)), ("wait", 1.5),
               ("stow", None), ("wait", 4.0), ("roll", [1, 0, 0])]   # the roll would auto-stow anyway
MISSIONS = {"demo": DEMO, "deploy": DEMO_DEPLOY}


def record(path, site=DEFAULT_SITE, mission="demo", goto=None):
    """Run a scripted mission headless and save it for render_blender.py."""
    w = build_world(site)
    for _ in range(int(1.0 / DT)):
        w.step()
    c = Cubli(w)
    w.recorder = Recorder(w)
    for _ in range(int(1.5 / DT)):                         # a beat of stillness to open the shot
        w.step()
    steps = MISSIONS[mission] if goto is None else [("goto", goto)]
    for kind, arg in steps:
        if kind in ("deploy", "deploy_lidar", "deploy_cam", "stow", "aim"):
            {"deploy": w.rig.deploy, "deploy_lidar": w.rig.deploy, "deploy_cam": w.rig.deploy, "stow": w.rig.stow,
             "aim": lambda: w.rig.aim(*arg)}[kind]()
            c.status = f"sensors: LiDAR {w.rig.state('lidar')}, camera {w.rig.state('cam')}"
        elif kind == "wait":
            t_end = w.d.time + arg
            while w.d.time < t_end:
                c.update()
                w.step()
        elif kind == "release":
            c.release_balance()
            run_script(w, c, [], t_max=40)
        elif kind == "goto":
            c.go_to(np.array(arg, float))
            run_script(w, c, [], t_max=900)
        else:
            if kind == "balance":
                c.balance(np.array(arg, float))
                t0 = w.d.time
                while (c.action is None or c.action.get("phase") != "balance") and w.d.time - t0 < 25 and \
                        (c.action or c.queue):
                    c.update()
                    w.step()
            else:
                run_script(w, c, [(kind, arg)])
                if "FAILED" in c.last_result:                 # the controller learned a bigger margin: retry once
                    run_script(w, c, [(kind, arg)])
        print(f"  {kind:8s} -> {c.status}")
    for _ in range(int(2.0 / DT)):
        c.update()
        w.step()
    w.recorder.save(path)
    return path


# ---------------------------------------------------------------------
# INTERACTIVE WINDOW
# ---------------------------------------------------------------------
PAN_STEP, TILT_STEP = math.radians(10), math.radians(5)


def hud(w, c, ui):
    """Top-left: what the controller is doing (H hides it; the sidebar on the right always stays)."""
    name, gvec, slope = w.presets[w.gi]
    v = np.linalg.norm(w.lin_vel())
    return (f"{name}   down tilted {slope:.1f} deg from the surface normal   cube speed {v * 100:.1f} cm/s\n"
            f"> {c.status}")


def sidebar(w, c, ui):
    """Right-hand control panel: every action with its key, grouped, plus live status."""
    name, gvec, _ = w.presets[w.gi]
    r = w.rig
    rp = [rpm(w.wheel_speed(k)) for k in range(3)]
    rec = "off"
    if w.recorder is not None:
        rec = f"REC {w.d.time - w.recorder.t[0] if w.recorder.t else 0:5.1f} s"
    deployed = not r.stowed
    pan, tilt = math.degrees(r.q("cam_pan")), math.degrees(r.q("cam_tilt"))
    rows = [
        ("DRIVE", "rolls/hops auto-stow sensors" if deployed else ""),
        ("  roll one face  [arrows]", ""),
        ("  hop forward  [Space]", ""),
        ("  go to camera target  [T]", "target set" if c.goal is not None else ""),
        ("BALANCE", "balancing" if (c.action or {}).get("phase") == "balance" else ""),
        ("  balance on edge / let go  [Enter]", ""),
        ("SENSORS", ""),
        ("  deploy / stow both  [S]", "out of the top face" if r.want_up else ""),
        ("  LiDAR on / off  [O]", r.state("lidar")),
        ("  camera on / off  [C]", r.state("cam")),
        ("  gimbal heading", f"{math.degrees(r.yaw):+4.0f} deg (level)"),
        ("  camera pan  [J] / [;]", f"{pan:+4.0f} deg"),
        ("  camera tilt  [I] / [K]", f"{tilt:+4.0f} deg"),
        ("  (sliders: right panel > Control)", ""),
        ("  sensor dashboard window", ui.get("dash", "off")),
        ("  photoreal Blender view  [B]", ui["blender"].status if ui.get("blender") else "off"),
        ("CAMERA", ""),
        ("  follow / free view  [F]", "follow" if ui["follow"] else "free"),
        ("  orbit / pan / zoom  [mouse]", ""),
        ("SIM", ""),
        ("  gravity  [G]", f"{name.replace('Psyche: ', '')}  {np.linalg.norm(gvec):.4f}"),
        ("  pause  [P]", "PAUSED" if ui["paused"] else "running"),
        ("  slower / faster  [-] / [=]", f"x{ui['speed']:g}" + ("  REAL TIME" if ui["speed"] == 1 else "")),
        ("  reset  [R]    status text  [H]", ""),
        ("  wheel rpm", " ".join(f"{x:5.0f}" for x in rp)),
        ("  mass / CoM height", f"{w.total_mass()*1000:.0f} g / {(w.com()[2] - w.height_at(*w.com()[:2]))*1000:.0f} mm"),
        ("RECORD", ""),
        ("  start / stop take  [V]", rec),
    ]
    return "\n".join(a for a, _ in rows), "\n".join(b for _, b in rows)


def _dash_python():
    """A Python that can run dashboard.py: needs mujoco + opencv + scipy; detection also wants torch +
    ultralytics + onnxruntime. Override with CUBLI_DASH_PYTHON."""
    import shutil
    import subprocess
    cands = [os.environ.get("CUBLI_DASH_PYTHON"), sys.executable, "/opt/anaconda3/bin/python3",
             shutil.which("python3"), shutil.which("python")]
    check = ("import importlib.util as u,sys; m=sys.argv[1].split(','); "
             "sys.exit(0 if all(u.find_spec(x) for x in m) else 1)")
    best = None
    for py in [p for p in cands if p and os.path.exists(p)]:
        if subprocess.run([py, "-c", check, "mujoco,cv2,scipy"], capture_output=True).returncode:
            continue
        if not subprocess.run([py, "-c", check, "torch,ultralytics,onnxruntime"], capture_output=True).returncode:
            return py
        best = best or py
    return best


class DashboardLink:
    """Starts dashboard.py (a plain-python OpenCV window) and streams this sim's state to it; keys typed in the
    dashboard come back. Sensor rendering happens in that process, so this window never stutters for it."""

    def __init__(self, site, extra_args=()):
        import subprocess
        import threading
        from multiprocessing.connection import Listener
        self.keys, self.latest, self.sent, self.conn = collections.deque(), None, None, None
        py = _dash_python()
        if py is None:
            raise RuntimeError("no python with mujoco + opencv + scipy for the dashboard")
        key = os.urandom(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=key)
        self.proc = subprocess.Popen([py, "-B", os.path.join(HERE, "dashboard.py"), "--port",
                                      str(self.listener.address[1]), "--site", site, *extra_args],
                                     env=dict(os.environ, CUBLI_DASH_KEY=key.hex()))
        self.status = f"starting ({os.path.basename(os.path.dirname(os.path.dirname(py)))} python)"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        try:
            self.conn = self.listener.accept()
            self.status = "connected"
            while True:
                st = self.latest
                if st is not None and st is not self.sent:
                    self.conn.send(st)
                    self.sent = st
                while self.conn.poll():
                    kind, val = self.conn.recv()
                    if kind == "key":
                        self.keys.append(val)
                time.sleep(1 / 60)
        except (EOFError, OSError):
            self.status = "closed"

    def publish(self, w, c, ui):
        r = w.rig
        self.latest = dict(qpos=w.d.qpos.copy(), time=float(w.d.time), gravity=np.array(w.m.opt.gravity),
                           gravity_name=w.presets[w.gi][0], lidar_on=r.lidar_on, cam_on=r.cam_on, want_up=r.want_up,
                           lidar_epoch=r.lidar_epoch,
                           targets=dict(r.target), lidar_state=r.state("lidar"), cam_state=r.state("cam"),
                           status=c.status, speed=ui["speed"], paused=ui["paused"],
                           recording=w.recorder is not None, resets=ui.get("resets", 0))

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


def _find_blender():
    """CUBLI_BLENDER, then `blender` on the PATH, then the usual install folders (macOS, Windows, Linux)."""
    import glob
    import shutil
    cands = [os.environ.get("CUBLI_BLENDER"), shutil.which("blender"), "/Applications/Blender.app/Contents/MacOS/Blender"]
    cands += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender*\blender.exe"), reverse=True)
    cands += ["/usr/bin/blender", "/snap/bin/blender"]
    return next((c for c in cands if c and os.path.exists(c)), "/Applications/Blender.app/Contents/MacOS/Blender")


BLENDER = _find_blender()


class BlenderLink:
    """Photoreal live view: Blender (EEVEE) runs blender_live.py and copies this sim's poses every frame over UDP.
    MuJoCo stays the only source of motion."""
    PACKET = f"<d3d4d3d{len(SENSOR_JOINTS) + 3}d"     # time, pos, quat, wheels, sensor joints (gimbal = 4)

    def __init__(self, w, background_test=None):
        import socket
        import struct
        import subprocess
        self.struct = struct
        path = os.path.join(HERE, "recordings", "live_scene.npz")
        rec = Recorder(w)                       # same scene description the film renderer uses
        rec.tick()
        rec.save(path)
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cmd = [BLENDER] + (["-b"] if background_test else []) + [
            "--python", os.path.join(HERE, "blender_live.py"), "--", "--scene", path, "--port", str(self.port)]
        if background_test:
            cmd += ["--test", background_test]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE if background_test else None,
                                     stderr=subprocess.STDOUT if background_test else None, text=True)

    @property
    def status(self):
        return "running" if self.proc.poll() is None else "closed"

    def send(self, w):
        d = w.d
        pkt = self.struct.pack(self.PACKET, float(d.time), *d.qpos[0:3], *d.qpos[3:7],
                               *[d.qpos[a] for a in w.qadr], *sensor_qpos(w.m, d))
        try:
            self.sock.sendto(pkt, ("127.0.0.1", self.port))
        except OSError:
            pass

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


def view(site=DEFAULT_SITE, speed=1.0):
    import mujoco.viewer
    w = build_world(site)
    c = Cubli(w)
    keys = collections.deque()
    ui = dict(speed=speed, paused=False, follow=True, help=True, resets=0)
    for _ in range(int(1.0 / DT)):
        w.step()
    print(__doc__.split("CONTROLS")[1].split("What is real")[0])
    link = None
    if not os.environ.get("CUBLI_NO_DASHBOARD"):
        try:
            link = DashboardLink(site)
        except Exception as e:                              # noqa: BLE001
            print(f"sensor dashboard off: {e}")
            ui["dash"] = "off (see terminal)"
    try:
        _run_viewer(w, c, keys, ui, link)
    finally:
        if w.recorder is not None and len(w.recorder.t) > 10:   # closing the window keeps the take
            w.recorder.save(os.path.join(HERE, "recordings", time.strftime("take_%Y%m%d_%H%M%S.npz")))
        if link is not None:
            link.close()
        if ui.get("blender") is not None:
            ui["blender"].close()


def _run_viewer(w, c, keys, ui, link=None):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(w.m, w.d, key_callback=keys.append,
                                      show_left_ui=False,
                                      show_right_ui=not os.environ.get("CUBLI_NO_RIGHT_UI")) as v:
        with v.lock():
            v.cam.lookat[:] = w.com()
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.1, 125.0, -14.0
            flags0 = v.opt.flags.copy()
            groups0 = v.opt.geomgroup.copy()
            v.opt.actuatorgroup[:] = 0
            v.opt.actuatorgroup[1] = 1                     # Control panel: sensor sliders only (wheels = group 3)
            act0 = v.opt.actuatorgroup.copy()
            rnd0 = v.user_scn.flags.copy() if v.user_scn is not None else None
        last = time.perf_counter()
        frame_i, sim_debt = 0, 0.0
        selftest = os.environ.get("CUBLI_SELFTEST")      # CUBLI_SELFTEST=1: scripted key presses + log, 15 s
        script = [(1.0, "S"), (6.0, "J"), (6.5, "I")]
        t_start, next_log = time.perf_counter(), 0.0
        prof = dict(phys=0.0, lock=0.0, txt=0.0, sync=0.0)
        tic = time.perf_counter
        while v.is_running():
            now = time.perf_counter()
            frame = min(now - last, 0.1)
            last = now
            if link is not None:
                ui["dash"] = link.status
                while link.keys:
                    keys.append(link.keys.popleft())
            if selftest:
                el = now - t_start
                while script and el >= script[0][0]:
                    keys.append(GLFW[script.pop(0)[1]])
                if el >= next_log:
                    next_log += 1.0
                    vp = v.viewport
                    print(f"[selftest {el:4.1f}s] frame {frame_i}  viewport {vp.width}x{vp.height}  "
                          f"LiDAR {w.rig.state('lidar')}  camera {w.rig.state('cam')}  "
                          f"pan {math.degrees(w.rig.q('cam_pan')):+.0f}  tilt {math.degrees(w.rig.q('cam_tilt')):+.0f}  "
                          f"dashboard {ui.get('dash')}  sim t {w.d.time:6.2f}  ms: physics {prof['phys']:.1f} "
                          f"lock {prof['lock']:.1f} texts {prof['txt']:.1f} sync {prof['sync']:.1f}", flush=True)
                if el > 15:
                    print("[selftest] sidebar shown:\n" + "\n".join(
                        f"   {a:40s} {b}" for a, b in zip(*(t.split("\n") for t in sidebar(w, c, ui)))), flush=True)
                    v.close()
                    break
            keys_seen = bool(keys)
            while keys:
                handle_key(keys.popleft(), w, c, ui, v)
            t0 = tic()
            if not ui["paused"]:                           # fixed 1 ms steps; the remainder carries to the next
                sim_debt += frame * ui["speed"]            # frame so sim time tracks wall time exactly (no drift,
                n = min(int(sim_debt / DT), 12000)         # no rounding jitter)
                sim_debt -= n * DT
                for _ in range(n):
                    c.update()
                    w.step()
            t1 = tic()
            if link is not None and frame_i % 2 == 0:     # ~30 Hz to the sensor dashboard
                link.publish(w, c, ui)
            if ui.get("blender") is not None:              # every frame to the Blender live view
                ui["blender"].send(w)
            with v.lock():
                v.opt.flags[:] = flags0                    # single letters toggle MuJoCo's own visual flags,
                v.opt.geomgroup[:] = groups0               # geom groups and scene render flags (L, K, R, G, S...)
                v.opt.actuatorgroup[:] = act0              # -> put them back every frame
                if rnd0 is not None and (v.user_scn.flags != rnd0).any():
                    v.user_scn.flags[:] = rnd0
                if ui["follow"]:
                    a = 1 - math.exp(-frame * 3.0)
                    v.cam.lookat[:] += a * (w.com() + 0.03 * c.up() - v.cam.lookat)
            t2 = tic()
            if now - ui.get("t_overlay", 0.0) > 0.2 or keys_seen:   # overlays at ~5 Hz: set_texts waits for
                ui["t_overlay"] = now                                # the viewer's render lock
                left, right = sidebar(w, c, ui)            # texts are set outside v.lock()
                texts = [(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT, left, right)]
                if ui["help"]:
                    texts.append((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  hud(w, c, ui), ""))
                v.set_texts(texts)
            t3 = tic()
            v.sync()
            t4 = tic()
            for k_, a_, b_ in (("phys", t0, t1), ("lock", t1, t2), ("txt", t2, t3), ("sync", t3, t4)):
                prof[k_] = 0.8 * prof[k_] + 0.2 * (b_ - a_) * 1000
            frame_i += 1
            if w.recorder is not None and not v.is_running():
                break
            spare = 1 / 60 - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)


def handle_key(k, w, c, ui, v):
    az = math.radians(v.cam.azimuth)
    fwd = np.array([math.cos(az), math.sin(az), 0.0])
    right = np.array([fwd[1], -fwd[0], 0.0])
    arrows = {GLFW["UP"]: fwd, GLFW["DOWN"]: -fwd, GLFW["RIGHT"]: right, GLFW["LEFT"]: -right}
    r = w.rig
    if k in arrows:
        c.goal = None
        c.roll(arrows[k])
    elif k == GLFW["ENTER"]:
        if not c.release_balance():
            c.goal = None
            c.balance(fwd)
    elif k == GLFW["SPACE"]:
        c.hop(fwd)
    elif k == GLFW["T"]:
        tgt = np.array(v.cam.lookat[:2])
        c.go_to(tgt)
        c.status = f"heading to ({tgt[0]:.1f}, {tgt[1]:.1f}) m"
    elif k == GLFW["F"]:
        ui["follow"] = not ui["follow"]
    elif k == GLFW["G"]:
        c.cancel()
        w.set_gravity(w.gi + 1)
        c.status = f"gravity: {w.presets[w.gi][0]}"
    elif k == GLFW["P"]:
        ui["paused"] = not ui["paused"]
    elif k == GLFW["MINUS"]:
        ui["speed"] = max(ui["speed"] / 1.5, 0.25)
    elif k == GLFW["EQUAL"]:
        ui["speed"] = min(ui["speed"] * 1.5, 20)
    elif k == GLFW["R"]:
        c.__init__(w)
        gi = w.gi
        w.reset()
        w.set_gravity(gi)
        for _ in range(int(0.5 / DT)):
            w.step()
        c.status = "reset"
        ui["resets"] = ui.get("resets", 0) + 1
    elif k == GLFW["H"]:
        ui["help"] = not ui["help"]
    elif k in (GLFW["S"], GLFW["L"]):
        if r.want_up:
            r.stow()
            c.status = "sensors: stowing (camera level, then the mast comes down)"
        else:
            r.deploy()
            c.status = "sensors: deploying out of the top face; LiDAR + camera switch on when the head is out"
    elif k == GLFW["C"]:
        if not r.deployed:
            c.status = "camera: deploy the sensor head first (S)"
        else:
            r.cam_on = not r.cam_on
            c.status = f"camera {'on' if r.cam_on else 'off'}"
    elif k in (GLFW["J"], GLFW["SEMICOLON"], GLFW["I"], GLFW["K"]):
        d = {GLFW["J"]: (PAN_STEP, 0), GLFW["SEMICOLON"]: (-PAN_STEP, 0),
             GLFW["I"]: (0, TILT_STEP), GLFW["K"]: (0, -TILT_STEP)}[k]
        if not r.aim(*d):
            c.status = "pan/tilt: deploy the sensor head first (S)"
    elif k == GLFW["O"]:
        if not r.deployed:
            c.status = "LiDAR: deploy the sensor head first (S)"
        else:
            r.lidar_on = not r.lidar_on
            r.lidar_epoch += r.lidar_on
            c.status = f"LiDAR {'on' if r.lidar_on else 'off'}"
    elif k == GLFW["B"]:
        if ui.get("blender") is None or ui["blender"].status == "closed":
            if os.path.exists(BLENDER):
                ui["blender"] = BlenderLink(w)
                c.status = "opening the Blender live view (EEVEE). It takes ~10-20 s to build the scene"
            else:
                c.status = f"Blender not found at {BLENDER} (install it, or set CUBLI_BLENDER to blender's path)"
        else:
            c.status = "Blender live view already running"
    elif k == GLFW["V"]:
        if w.recorder is None:
            w.recorder = Recorder(w)
            c.status = "RECORDING (press V again to save)"
        else:
            path = os.path.join(HERE, "recordings", time.strftime("take_%Y%m%d_%H%M%S.npz"))
            if len(w.recorder.t) > 10:
                w.recorder.save(path)
                c.status = f"saved {os.path.relpath(path, HERE)}  -> render it with render_blender.py"
            w.recorder = None


# ---------------------------------------------------------------------
# HEADLESS: sites table, scripted test, report
# ---------------------------------------------------------------------
def print_sites():
    a, b, c = psyche_axes()
    print(f"Psyche ellipsoid {PSYCHE_AXES_KM} km, mass {PSYCHE_MASS:.3e} kg, density {psyche_density():.0f} kg/m^3, "
          f"spin {PSYCHE_PERIOD_H} h")
    print(f"{'site':18s} {'lat':>5s} {'lon':>5s} {'|g| m/s^2':>10s} {'geoid tilt':>10s} {'radius km':>9s}  note")
    for name, s in SITES.items():
        fr = site_frame(s["lat"], s["lon"])
        print(f"{name:18s} {s['lat']:5.0f} {s['lon']:5.0f} {fr['g_mag']:10.4f} {fr['slope_deg']:9.1f}d "
              f"{fr['radius_km']:9.1f}  {s['note']}")


def capability(w):
    c = Cubli(w)
    out = []
    gi0 = w.gi
    for i, (name, g, slope) in enumerate(w.presets):
        w.reset()
        w.set_gravity(i)
        for _ in range(300):
            w.step()
        a = c.plan("roll", np.array([1.0, 0, 0]))
        out.append((name, np.linalg.norm(g), rpm(a["need"]), rpm(a["need"] * ROLL_MARGIN)))
    w.reset()
    w.set_gravity(gi0)
    return out


def mass_summary(mm):
    M, com, I = mm["body"]
    tot = M + sum(x["mass"] for x in mm["wheels"])
    lines = [f"cube without sensors {tot*1000:.0f} g | body {M*1000:.0f} g, centre of mass offset {np.round(com*1000,1)} mm "
             f"from cube centre",
             f"+ sensors {SENSOR_MASS*1000:.0f} g as separate bodies (LiDAR {LIDAR_MASS*1000:.0f} + mast {MAST_MASS*1000:.0f}"
             f" + OAK-D {CAM_MASS*1000:.0f} + pan/tilt "
             f"{PANTILT_MASS*1000:.0f}); SENSORS_IN_TOTAL_MASS={SENSORS_IN_TOTAL_MASS} -> whole cube "
             f"{(tot + SENSOR_MASS) * 1000:.0f} g"]
    for k, x in enumerate(mm["wheels"]):
        lines.append(f"wheel {k} ({'+' if x['sign'] > 0 else '-'}{'XYZ'[x['axis']]} face): "
                     f"{x['mass']*1000:.0f} g spinning, I_axial {x['I_axial']:.2e} kg m^2")
    lines.append(f"GM3506 at {V_BUS} V: no-load {rpm(W_NOLOAD):.0f} rpm, stall {KT*min(V_BUS/R_PHASE, I_LIMIT):.3f} N*m, "
                 f"Kt {KT:.4f} N*m/A")
    return "\n".join(lines)


def run_script(w, c, script, t_max=120.0):
    for item in script:
        getattr(c, item[0])(np.array(item[1], float))
    t0 = w.d.time
    while (c.action or c.queue or c.goal is not None) and w.d.time - t0 < t_max:
        c.update()
        w.step()
    return w.d.time - t0


def test(site=DEFAULT_SITE):
    w = build_world(site)
    print(mass_summary(w.mm))
    for i in range(int(1.0 / DT)):
        w.step()
    for name, gm, need, use in capability(w):
        top = rpm(0.97 * W_NOLOAD - 15)
        print(f"  {name:28s} |g|={gm:7.4f}  roll needs {need:6.0f} rpm (uses {use:6.0f})  "
              f"{'OK' if use < top else 'TOO SLOW'} (max {top:.0f})")
    w.reset()
    for _ in range(int(1.0 / DT)):
        w.step()
    c = Cubli(w)
    for d in ([1, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]):
        p0 = w.com().copy()
        t = run_script(w, c, [("roll", d)])
        print(f"  roll {d}: {c.last_result}  ({t:.1f} s sim, moved {np.round((w.com()-p0)[:2]*100,0)} cm)")
    t = run_script(w, c, [("balance", [0, -1, 0])], t_max=20)
    print(f"  balance: {c.status}  ({t:.1f} s)")
    if c.release_balance():
        run_script(w, c, [], t_max=30)
        print(f"  released: {c.last_result}")
    return w, c


def report(outdir="psyche_report_v2", site=DEFAULT_SITE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    # 1. gravity + geoid tilt over the whole asteroid
    lats = np.linspace(-89, 89, 90)
    lons = np.linspace(0, 360, 181)
    G = np.zeros((len(lats), len(lons)))
    S = np.zeros_like(G)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            fr = site_frame(la, lo)
            G[i, j], S[i, j] = fr["g_mag"], fr["slope_deg"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for a_, D, ttl, cm in ((ax[0], G, "surface |g| incl. spin (m/s^2)", "viridis"),
                           (ax[1], S, "geoid tilt: angle between 'down' and the surface normal (deg)", "magma")):
        im = a_.pcolormesh(lons, lats, D, cmap=cm, shading="auto")
        fig.colorbar(im, ax=a_)
        for n, s in SITES.items():
            a_.plot(s["lon"], s["lat"], "w^", ms=8, mec="k")
            a_.annotate(n, (s["lon"], s["lat"]), xytext=(4, 4), textcoords="offset points", color="w", fontsize=8)
        a_.set_title(ttl)
        a_.set_xlabel("longitude (deg)")
        a_.set_ylabel("latitude (deg)")
    fig.suptitle(f"16 Psyche as a {PSYCHE_AXES_KM} km ellipsoid (Shepard et al. 2021), mass {PSYCHE_MASS:.3g} kg")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "psyche_gravity_map.png"), dpi=120)
    plt.close(fig)
    # 2. capability + scripted rolls
    w, c = test(site)
    cap = capability(w)
    top = rpm(0.97 * W_NOLOAD - 15)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = [x[0].replace("Psyche: ", "") for x in cap]
    ax.bar(names, [x[3] for x in cap], color=["#e67e22" if x[3] < top else "#c0392b" for x in cap])
    ax.axhline(top, color="k", ls="--", label=f"GM3506 max at {V_BUS} V ({top:.0f} rpm)")
    ax.set_yscale("log")
    ax.set_ylabel("wheel speed needed to roll (rpm)")
    ax.set_title(f"Can the {w.mm['edge']*1000:.0f} mm, {TOTAL_MASS*1000:.0f} g cube roll? (wheel {WHEEL_MASS*1000:.0f} g each)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roll_capability.png"), dpi=120)
    plt.close(fig)
    L = np.array(c.log)
    if len(L):
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for k in range(3):
            ax[0].plot(L[:, 0], L[:, 1 + k], label=f"wheel {k}")
        ax[0].set_ylabel("rpm")
        ax[0].legend()
        ax[1].plot(L[:, 0], L[:, 4], "k")
        ax[1].set_ylabel("tilt past the edge (deg)")
        ax[1].set_xlabel("sim time (s)")
        fig.suptitle(f"scripted rolls + edge balance at {site}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "roll_timeline.png"), dpi=120)
        plt.close(fig)
    # 3. renders
    r = mujoco.Renderer(w.m, 720, 1280)
    for nm, (dist, az, el) in {"hero.png": (0.9, 130, -10), "horizon.png": (2.5, 60, -3)}.items():
        cam = mujoco.MjvCamera()
        cam.lookat[:] = w.com()
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(w.d, cam)
        import imageio
        imageio.imwrite(os.path.join(outdir, nm), r.render())
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(mass_summary(w.mm) + "\n\n")
        for name, gm, need, use in cap:
            f.write(f"{name:28s} |g|={gm:.4f}  roll needs {need:.0f} rpm, uses {use:.0f}, max {top:.0f}\n")
    print(f"saved report to {outdir}/")


def frames_to_mp4(folder, out, fps=30):
    """Stitch the PNG frames from render_blender.py into an H.264 mp4 (uses imageio-ffmpeg from the venv)."""
    import imageio.v2 as imageio
    files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    if not files:
        sys.exit(f"no PNG frames in {folder}")
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=9, macro_block_size=8,
                            pixelformat="yuv420p", ffmpeg_params=["-preset", "veryfast"]) as wtr:
        for f in files:
            wtr.append_data(imageio.imread(os.path.join(folder, f))[..., :3])
    print(f"wrote {out} ({len(files)} frames, {len(files) / fps:.1f} s)")


def main():
    ap = argparse.ArgumentParser(description="Psyche Cubli v2")
    ap.add_argument("mode", nargs="?", default="view", choices=["view", "sites", "test", "report", "record", "video"])
    ap.add_argument("--out", default=None, help="record: .npz path | video: .mp4 path")
    ap.add_argument("--mission", default="demo", choices=list(MISSIONS), help="record: scripted mission")
    ap.add_argument("--goto", nargs=2, type=float, default=None, metavar=("X", "Y"),
                    help="record: drive to this point instead of the demo mission")
    ap.add_argument("--frames", default=None, help="video: folder of rendered PNG frames")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--site", default=DEFAULT_SITE, choices=list(SITES))
    ap.add_argument("--speed", type=float, default=1.0, help="sim speed multiplier in the window (1 = real time; "
                    "x4 made Psyche look like Moon gravity: time x4 looks like gravity x16)")
    ap.add_argument("--slope", type=float, default=None, help="extra regional slope in degrees (downhill = -x)")
    args = ap.parse_args()
    if args.slope is not None:
        global EXTRA_SLOPE_DEG
        EXTRA_SLOPE_DEG = args.slope
    if args.mode == "sites":
        print_sites()
    elif args.mode == "test":
        test(args.site)
    elif args.mode == "report":
        report(site=args.site)
    elif args.mode == "record":
        record(args.out or os.path.join(HERE, "recordings", f"{args.mission}.npz"), args.site, args.mission,
               goto=args.goto)
    elif args.mode == "video":
        frames_to_mp4(args.frames or os.path.join(HERE, "renders", "demo", "frames"),
                      args.out or os.path.join(HERE, "renders", "demo.mp4"), args.fps)
    else:
        if sys.platform == "darwin" and "mjpython" not in os.path.basename(sys.executable) and \
                not os.environ.get("MJPYTHON_BIN"):
            print("On a Mac the 3D window needs:  mjpython psyche_cubli_v2.py")
        view(args.site, args.speed)


if __name__ == "__main__":
    main()
