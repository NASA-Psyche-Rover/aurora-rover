#!/usr/bin/env python3
"""Measures how 'glitchy' the motion is, headless: bounces after each landing, contact chatter, and jitter
at rest, for a roll, a hop and a balance at Psyche gravity. Also hop height with / without a brake.

    ./vpy tools/motion_diag.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psyche_cubli_v2 as V  # noqa: E402


def hull_contacts(w, hull):
    return sum(1 for i in range(w.d.ncon) if w.d.contact[i].geom1 in hull or w.d.contact[i].geom2 in hull)


def watch(w, c, start, t_max=80.0):
    """Run one manoeuvre; log contact count, CoM height, speeds."""
    hull = {g for g in range(w.m.ngeom) if w.m.geom_bodyid[g] == w.cube and w.m.geom_contype[g]}
    start()
    log = []
    t0 = w.d.time
    while (c.action or c.queue) and w.d.time - t0 < t_max:
        c.update()
        w.step()
        log.append((w.d.time, hull_contacts(w, hull), w.com()[2], np.linalg.norm(w.lin_vel()),
                    np.linalg.norm(w.ang_vel())))
    for _ in range(int(3 / V.DT)):                # 3 s after the controller says it is done
        c.update()
        w.step()
        log.append((w.d.time, hull_contacts(w, hull), w.com()[2], np.linalg.norm(w.lin_vel()),
                    np.linalg.norm(w.ang_vel())))
    L = np.array(log)
    air = L[:, 1] == 0
    edges = np.flatnonzero(np.diff(air.astype(int)) == 1)            # contact -> airborne transitions
    lens = []
    for e in edges:
        f = np.flatnonzero(~air[e + 1:])
        lens.append((f[0] if len(f) else len(air) - e) * V.DT)
    lens = np.array(lens)
    rest = L[-int(2 / V.DT):]                      # last 2 s: should be still
    return dict(result=c.last_result, airborne_phases=int((lens > 0.02).sum()),
                longest_air_s=float(lens.max()) if len(lens) else 0.0,
                micro_hops=int(((lens > 0) & (lens <= 0.02)).sum()),
                contact_changes_per_s=float(np.abs(np.diff(L[:, 1])).sum() / (L[-1, 0] - L[0, 0])),
                rest_speed_mm_s=float(rest[:, 3].max() * 1000), rest_spin_deg_s=float(math.degrees(rest[:, 4].max())),
                apex_mm=float((L[:, 2].max() - L[0, 2]) * 1000))


def main():
    for brake in (0.0, 0.3):
        V.MECH_BRAKE = brake
        w = V.build_world()
        c = V.Cubli(w)
        for _ in range(int(2 / V.DT)):
            w.step()
        print(f"\n=== Psyche {np.linalg.norm(w.g):.4f} m/s^2, MECH_BRAKE {brake} N*m ===")
        tests = [("roll +x", lambda: c.roll(np.array([1.0, 0, 0]))),
                 ("hop +y", lambda: c.hop(np.array([0, 1.0, 0])))]
        if brake == 0.0:
            tests.append(("balance (Enter) + release", lambda: c.balance(np.array([-1.0, 0, 0]))))
        for name, fn in tests:
            r = watch(w, c, fn)
            if name.startswith("balance"):
                c.release_balance()
                r2 = watch(w, c, lambda: None)
                r["result"] += " | " + r2["result"]
            print(f"{name:26s} {r['result']}")
            print(f"   airborne phases >20 ms: {r['airborne_phases']} (longest {r['longest_air_s']:.2f} s), "
                  f"micro-hops <=20 ms: {r['micro_hops']}, contact changes {r['contact_changes_per_s']:.1f}/s, "
                  f"CoM apex +{r['apex_mm']:.0f} mm, after: max speed {r['rest_speed_mm_s']:.2f} mm/s, "
                  f"max spin {r['rest_spin_deg_s']:.2f} deg/s")


if __name__ == "__main__":
    main()
