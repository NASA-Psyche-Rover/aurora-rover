#!/usr/bin/env python3
"""STEP 1 evidence: run the same movement tests on the existing sim (MuJoCo Sim/psyche_cubli.py) and on v2.

    "NASA Psyche/venv/bin/python" tools/movement_compare.py      # -> movement_results.json + printed table

Both sims are imported read-only and run headless with their own default site/gravity.
Tests: idle drift 5 s, 4 rolls (+x +y -x -y), edge balance + 5 s hold + release, hop, real-time factor.
A roll counts as OK if the cube moved >= half an edge along the commanded direction, ends flat on a face
(a body axis within 3 deg of vertical) and at rest.
"""
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(os.path.expanduser("~"), "Desktop", "Aayou", "ASU", "EPICS", "NASA Psyche", "Stage 3",
                   "MuJoCo Sim", "psyche_cubli.py")
DIRS = {"+x": (1, 0), "+y": (0, 1), "-x": (-1, 0), "-y": (0, -1)}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flat(R):
    return float(np.max(np.abs(R[2]))) > math.cos(math.radians(3))


class Clock:
    def __init__(self, d):
        self.d, self.sim0, self.wall0 = d, d.time, time.perf_counter()

    def factor(self):
        return (self.d.time - self.sim0) / max(time.perf_counter() - self.wall0, 1e-9)


# ------------------------------------------------------------------ existing sim
def test_old():
    P = load(OLD, "psyche_cubli_old")
    out = dict(name="existing (psyche_cubli.py)")
    w = P.World(g=P.G_PSYCHE)
    out["edge_m"], out["mass_kg"] = P.L, P.CUBE_MASS
    out["mass_source"] = "PLACEHOLDER 1.0 kg; uniform-cube inertia (I = m L^2/6), wheels point masses on a 4.5 cm ring"
    out["contact"] = "single box geom on the heightfield"
    out["gravity"] = f"{P.G_PSYCHE} m/s^2 hard-coded, straight down"
    out["motor"] = (f"linear torque-speed line: stall {P.MOTOR_STALL_TORQUE} N*m (PLACEHOLDER), no-load "
                    f"{P.MOTOR_NOLOAD_RPM} rpm (used-part listing); no current/back-EMF/resistance model; "
                    f"separate mechanical brake {P.BRAKE_TORQUE} N*m (PLACEHOLDER); no bearing friction")
    # idle drift
    w.reset()
    for _ in range(int(1.0 / P.DT)):
        mujoco.mj_step(w.m, w.d)
    p0, R0 = w.pos(), w.rot()
    path = []
    clk = Clock(w.d)
    for _ in range(int(5.0 / P.DT)):
        mujoco.mj_step(w.m, w.d)
        path.append(w.pos())
    path = np.array(path)
    out["idle_drift_mm"] = float(np.linalg.norm(w.pos() - p0) * 1000)
    out["idle_max_excursion_mm"] = float(np.max(np.linalg.norm(path - p0, axis=1)) * 1000)
    out["idle_rot_deg"] = float(math.degrees(math.acos(np.clip((np.trace(R0.T @ w.rot()) - 1) / 2, -1, 1))))
    out["idle_rtf"] = clk.factor()
    # rolls
    c = P.Cubli(w)
    rolls = {}
    clk = Clock(w.d)
    for name, dxy in DIRS.items():
        p0 = w.pos()
        c.start_roll(dxy)
        t = 0.0
        while c.mode == "roll" and t < 40:
            c.step()
            t += P.DT
        for _ in range(int(1.5 / P.DT)):          # let it come to rest
            c.step()
        mv = (w.pos() - p0)[:2] @ np.array(dxy, float)
        ok = mv > 0.5 * P.L and flat(w.rot()) and np.linalg.norm(w.lin_vel()) < 0.02
        rolls[name] = dict(ok=bool(ok), moved_cm=round(float(mv) * 100, 1), sim_s=round(t, 1), flat=flat(w.rot()))
        c.mode = "idle"
    out["rolls"] = rolls
    out["roll_rtf"] = clk.factor()
    out["balance"] = dict(ok=False, note="no edge-balance controller in this sim")
    out["release"] = dict(ok=False, note="n/a (no balance)")
    clk = Clock(w.d)
    h = P.execute_hop(w, c, heading=0.0, rpm=2500)
    out["hop"] = dict(ok=bool(h["lifted"] and not h["unstable"]), moved_m=round(h["moved"], 2),
                      air_s=round(h["air_s"], 1), apex_cm=round(h["apex_m"] * 100, 1))
    out["hop_rtf"] = clk.factor()
    return out


# ------------------------------------------------------------------ v2
def test_v2():
    sys.path.insert(0, HERE)
    V = load(os.path.join(HERE, "psyche_cubli_v2.py"), "psyche_cubli_v2_cmp")
    out = dict(name="v2 (psyche_cubli_v2.py)")
    w = V.build_world(V.DEFAULT_SITE)
    out["edge_m"], out["mass_kg"] = w.mm["edge"], w.total_mass()
    M, com, _ = w.mm["body"]
    out["mass_source"] = (f"CAD edge {w.mm['edge']*1000:.1f} mm; part volumes/centroids/inertias from the STL split, "
                          f"masses YOUR NUMBER 900 g total + DATASHEET motors; CoM offset {np.round(com*1000,1).tolist()} mm")
    out["contact"] = "rounded hull: 8 corner spheres + 12 edge capsules (r 5 mm) on the heightfield; box only vs boulders"
    out["gravity"] = (f"{np.linalg.norm(w.g):.4f} m/s^2 at {w.site}, computed from the Shepard 2021 ellipsoid + spin, "
                      f"tilted {w.presets[w.gi][2]:.1f} deg from the surface normal")
    out["motor"] = (f"GM3506: R {V.R_PHASE} ohm, KV {V.KV:.0f} rpm/V, Kt {V.KT:.4f} N*m/A, I limit {V.I_LIMIT} A, "
                    f"{V.V_BUS} V bus, back-EMF current model, electrical braking, Coulomb+viscous bearing friction "
                    f"with friction feed-forward, 200 Hz control with IMU noise")
    for _ in range(int(1.0 / V.DT)):
        w.step()
    p0, R0 = w.com(), w.R().copy()
    path = []
    clk = Clock(w.d)
    for _ in range(int(5.0 / V.DT)):
        w.step()
        path.append(w.com())
    path = np.array(path)
    out["idle_drift_mm"] = float(np.linalg.norm(w.com() - p0) * 1000)
    out["idle_max_excursion_mm"] = float(np.max(np.linalg.norm(path - p0, axis=1)) * 1000)
    out["idle_rot_deg"] = float(math.degrees(math.acos(np.clip((np.trace(R0.T @ w.R()) - 1) / 2, -1, 1))))
    out["idle_rtf"] = clk.factor()
    c = V.Cubli(w)
    rolls = {}
    clk = Clock(w.d)
    for name, dxy in DIRS.items():
        p0 = w.com()
        t = V.run_script(w, c, [("roll", [dxy[0], dxy[1], 0])])
        if "FAILED" in c.last_result:            # the controller raises its margin and retries once (by design)
            t += V.run_script(w, c, [("roll", [dxy[0], dxy[1], 0])])
        mv = (w.com() - p0)[:2] @ np.array(dxy, float)
        ok = mv > 0.5 * w.mm["edge"] and flat(w.R()) and np.linalg.norm(w.lin_vel()) < 0.02
        rolls[name] = dict(ok=bool(ok), moved_cm=round(float(mv) * 100, 1), sim_s=round(t, 1), flat=flat(w.R()),
                           result=c.last_result)
    out["rolls"] = rolls
    out["roll_rtf"] = clk.factor()
    clk = Clock(w.d)
    V.run_script(w, c, [("balance", [1, 0, 0])], t_max=25)
    t_hold = w.d.time + 5.0
    held = c.action is not None and c.action.get("phase") == "balance"
    while held and w.d.time < t_hold:
        c.update()
        w.step()
        held = c.action is not None and c.action.get("phase") == "balance"
    out["balance"] = dict(ok=bool(held), note=f"held on the edge 5 s: {held}; {c.status}")
    rel = c.release_balance()
    V.run_script(w, c, [], t_max=40)
    out["release"] = dict(ok=bool(rel and "OK" in c.last_result), note=c.last_result)
    p0 = w.com()
    V.run_script(w, c, [("hop", [0, 1, 0])], t_max=60)
    mv = np.linalg.norm((w.com() - p0)[:2])
    out["hop"] = dict(ok=bool(mv > 0.1), moved_m=round(float(mv), 2), note=c.last_result)
    out["balance_hop_rtf"] = clk.factor()
    return out


def main():
    res = [test_old(), test_v2()]
    path = os.path.join(HERE, "movement_results.json")
    json.dump(res, open(path, "w"), indent=1)
    for r in res:
        print(f"\n=== {r['name']}")
        for k, v in r.items():
            if k != "name":
                print(f"  {k:22s} {v}")
    print("saved", path)


if __name__ == "__main__":
    main()
