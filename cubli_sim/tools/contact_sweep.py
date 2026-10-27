#!/usr/bin/env python3
"""Which regolith contact settings stop the chatter without breaking rolls / balance? Headless sweep.

    ./vpy tools/contact_sweep.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import motion_diag as MD  # noqa: E402
import psyche_cubli_v2 as V  # noqa: E402

CONFIGS = [  # (label, solref, solimp, fine roughness m)
    ("baseline v2", "0.01 1", "0.95 0.99 0.001", 0.003),
    ("smooth physics ground", "0.01 1", "0.95 0.99 0.001", 0.0005),
    ("overdamped 0.01/2", "0.01 2", "0.95 0.99 0.001", 0.003),
    ("softer overdamped 0.02/2", "0.02 2", "0.9 0.99 0.002", 0.003),
    ("smooth + overdamped 0.01/2", "0.01 2", "0.95 0.99 0.001", 0.0005),
    ("smooth + softer 0.02/2", "0.02 2", "0.9 0.99 0.002", 0.0005),
    ("smooth + softer 0.02/3", "0.02 3", "0.9 0.99 0.002", 0.0005),
    ("smooth + softer 0.03/2", "0.03 2", "0.9 0.99 0.003", 0.0005),
]
if len(sys.argv) > 1:                       # e.g. "6,7": run only those rows
    CONFIGS = [CONFIGS[int(i)] for i in sys.argv[1].split(",")]


def run(label, solref, solimp, fine):
    V.REG_SOLREF, V.REG_SOLIMP, V.FINE_RMS = solref, solimp, fine
    w = V.build_world()
    c = V.Cubli(w)
    for _ in range(int(2 / V.DT)):
        w.step()
    rows, ok = [], 0
    for d in ([1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]):
        r = MD.watch(w, c, lambda d=d: c.roll(np.array(d, float)))
        if "FAILED" in r["result"]:
            r = MD.watch(w, c, lambda d=d: c.roll(np.array(d, float)))
        ok += "OK" in r["result"]
        rows.append(r)
    b = MD.watch(w, c, lambda: c.balance(np.array([1.0, 0, 0])))
    if c.action is None or c.action.get("phase") != "balance":
        b = MD.watch(w, c, lambda: c.balance(np.array([1.0, 0, 0])))
    bal = "balancing" in c.status
    c.release_balance()
    rel = MD.watch(w, c, lambda: None)
    h = MD.watch(w, c, lambda: c.hop(np.array([0, 1.0, 0])))
    chg = np.mean([r["contact_changes_per_s"] for r in rows])
    air = sum(r["airborne_phases"] for r in rows)
    micro = sum(r["micro_hops"] for r in rows)
    w2 = V.build_world()                     # idle: sink depth + drift from rest
    z_spawn = w2.com()[2]
    for _ in range(int(8 / V.DT)):
        w2.step()
    p0, worst = w2.com().copy(), 0.0
    for _ in range(int(5 / V.DT)):
        w2.step()
        worst = max(worst, float(np.linalg.norm(w2.com() - p0)))
    roll_s = np.mean([float(r["result"].split(" in ")[1].split(" s")[0]) for r in rows if " in " in r["result"]])
    print(f"{label:28s} rolls {ok}/4 ({roll_s:.1f} s avg)  balance {'OK' if bal else 'FAIL'}  "
          f"release {'OK' if 'OK' in rel['result'] else 'FAIL'}"
          f"  | rolls: contact changes {chg:6.1f}/s, airborne {air:3d}, micro-hops {micro:4d}"
          f" | hop apex +{h['apex_mm']:.0f} mm, longest air {h['longest_air_s']:.2f} s"
          f" | idle {worst*1000:.3f} mm, settled {(z_spawn - p0[2])*1000:.1f} mm", flush=True)


if __name__ == "__main__":
    for cfg in CONFIGS:
        run(*cfg)
