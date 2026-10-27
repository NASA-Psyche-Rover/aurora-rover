#!/usr/bin/env python3
"""DONE check 2: movement tests on the chosen sim (v2 + stowed sensors): idle drift < 1 mm, 4 rolls, balance,
release (+ hop). Same procedure as tools/movement_compare.py. Exit code 0 only if all pass.

    ./vpy tools/movement_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import movement_compare as MC  # noqa: E402


def idle_from_rest():
    """Idle jitter = movement while it should be still. Spawned cubes first settle into the soft regolith (a one-off
    ~1 mm sink for the 1.29 kg cube); so wait until it is at rest (< 0.5 mm/s for 0.5 s), then watch 5 s."""
    import numpy as np
    V = MC.load(os.path.join(MC.HERE, "psyche_cubli_v2.py"), "v2_idle_rest")
    w = V.build_world(V.DEFAULT_SITE)
    calm, prev = 0.0, w.com().copy()
    while calm < 0.5 and w.d.time < 15.0:
        w.step()
        v = np.linalg.norm(w.com() - prev) / V.DT
        prev = w.com().copy()
        calm = calm + V.DT if v < 0.0005 else 0.0
    t_rest, p0, worst = w.d.time, w.com().copy(), 0.0
    for _ in range(int(5.0 / V.DT)):
        w.step()
        worst = max(worst, float(np.linalg.norm(w.com() - p0)))
    return t_rest, float(np.linalg.norm(w.com() - p0)) * 1000, worst * 1000


def main():
    r = MC.test_v2()
    print(f"cube {r['mass_kg']*1000:.0f} g (sensors stowed), {r['gravity']}")
    t_rest, end, worst = idle_from_rest()
    print(f"(fixed 1 s settle, as in MOVEMENT_DECISION.md: end {r['idle_drift_mm']:.3f} mm, worst "
          f"{r['idle_max_excursion_mm']:.3f} mm -- that window still contains the spawn settling)")
    checks = [("idle drift 5 s < 1 mm (from rest)", worst < 1.0,
               f"at rest after {t_rest:.1f} s; then end {end:.3f} mm, worst {worst:.3f} mm")]
    for d, x in r["rolls"].items():
        checks.append((f"roll {d}", x["ok"], f"{x['moved_cm']} cm in {x['sim_s']} s, {x['result']}"))
    checks += [("balance on edge (5 s hold)", r["balance"]["ok"], r["balance"]["note"]),
               ("release", r["release"]["ok"], r["release"]["note"]),
               ("hop", r["hop"]["ok"], r["hop"]["note"]),
               ("faster than real time", min(r["idle_rtf"], r["roll_rtf"], r["balance_hop_rtf"]) > 1,
                f"idle {r['idle_rtf']:.0f}x, rolls {r['roll_rtf']:.0f}x, balance/hop {r['balance_hop_rtf']:.0f}x")]
    for name, ok, det in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {det}")
    n = sum(ok for _, ok, _ in checks)
    print(f"{n}/{len(checks)} movement checks passed")
    return n == len(checks)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
