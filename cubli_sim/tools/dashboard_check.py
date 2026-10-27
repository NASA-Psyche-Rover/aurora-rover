#!/usr/bin/env python3
"""Headless check of the sim -> dashboard link: runs the sim (sensors deployed), streams its state to dashboard.py
started in --snapshot mode, which writes one PNG per tab to graphics/dash_<tab>.png.

    ./vpy tools/dashboard_check.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import psyche_cubli_v2 as V  # noqa: E402


def main():
    w = V.build_world()
    c = V.Cubli(w)
    ui = dict(speed=1.0, paused=False, resets=0)
    for _ in range(int(1.5 / V.DT)):
        w.step()
    w.rig.deploy_lidar()
    w.rig.deploy_cam()
    for _ in range(int(5 / V.DT)):
        c.update()
        w.step()
    link = V.DashboardLink(V.DEFAULT_SITE, ["--snapshot", os.path.join(HERE, "graphics", "dash")])
    t0 = time.time()
    while link.proc.poll() is None and time.time() - t0 < 240:
        for _ in range(20):                       # keep the sim alive and streaming while the dashboard renders
            c.update()
            w.step()
        link.publish(w, c, ui)
        while link.keys:
            print("key from dashboard:", link.keys.popleft())
        time.sleep(0.02)
    print("dashboard exit code:", link.proc.poll(), " link:", link.status)


if __name__ == "__main__":
    main()
