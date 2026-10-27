#!/usr/bin/env python3
"""STEP 1c evidence: idle drift for 5 s at the SAME gravity values in both sims, plus an ablation of v2 with
its rounded hull replaced by the box hull (what the existing sim uses). -> movement_idle_same_gravity.json"""
import json
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import movement_compare as MC  # noqa: E402

HERE = MC.HERE
P = MC.load(MC.OLD, "old_idle")
V = MC.load(os.path.join(HERE, "psyche_cubli_v2.py"), "v2_idle")


def idle(step, pos, n):
    for _ in range(n):                     # 1 s settle
        step()
    p0, ex = pos(), 0.0
    for _ in range(5 * n):                 # 5 s idle
        step()
        ex = max(ex, float(np.linalg.norm(pos() - p0)))
    return float(np.linalg.norm(pos() - p0)) * 1000, ex * 1000


def main():
    rows = []
    for g in (0.0885, 0.0966, 0.1234, 0.144, 1.62):
        w = P.World(g=g)
        w.reset()
        d1, e1 = idle(lambda: mujoco.mj_step(w.m, w.d), w.pos, int(1 / P.DT))
        vw = V.build_world(V.DEFAULT_SITE)
        vw.m.opt.gravity[:] = (0, 0, -g)
        d2, e2 = idle(vw.step, vw.com, int(1 / V.DT))
        vb = V.build_world(V.DEFAULT_SITE)          # ablation: box hull on the heightfield, rounded hull off
        vb.m.opt.gravity[:] = (0, 0, -g)
        for gi in range(vb.m.ngeom):
            if vb.m.geom_bodyid[gi] == vb.cube and vb.m.geom_contype[gi] == 1:
                vb.m.geom_contype[gi] = vb.m.geom_conaffinity[gi] = 0
        vb.m.geom_contype[vb.hull] = vb.m.geom_conaffinity[vb.hull] = 3
        d3, e3 = idle(vb.step, vb.com, int(1 / V.DT))
        rows.append(dict(g=g, existing_box=[round(d1, 3), round(e1, 3)], v2_rounded=[round(d2, 3), round(e2, 3)],
                         v2_box_ablation=[round(d3, 3), round(e3, 3)]))
        print(f"g={g:7.4f}  existing(box): drift {d1:7.3f} mm, max {e1:7.3f} | v2 rounded: {d2:6.3f} / {e2:6.3f}"
              f" | v2 with box hull: {d3:7.3f} / {e3:7.3f}")
    json.dump(rows, open(os.path.join(HERE, "movement_idle_same_gravity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
