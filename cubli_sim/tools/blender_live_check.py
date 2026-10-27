#!/usr/bin/env python3
"""Headless check of the Blender live link: the sim streams real poses (sensors deploying) over UDP to
blender_live.py started with -b --test, which renders one EEVEE frame -> graphics/blender_live_test.png

    ./vpy tools/blender_live_check.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import psyche_cubli_v2 as V  # noqa: E402


def main():
    w = V.build_world()
    c = V.Cubli(w)
    for _ in range(int(1.5 / V.DT)):
        w.step()
    out = os.path.join(HERE, "graphics", "blender_live_test.png")
    bl = V.BlenderLink(w, background_test=out)
    w.rig.deploy_lidar()
    w.rig.deploy_cam()
    t0, sent = time.time(), 0
    while bl.proc.poll() is None and time.time() - t0 < 600:
        for _ in range(33):                      # ~real time: 33 ms of sim per 33 ms of wall time
            c.update()
            w.step()
        bl.send(w)
        sent += 1
        time.sleep(1 / 30)
    print(bl.proc.stdout.read()[-1500:])
    print(f"sent {sent} poses; blender exit {bl.proc.poll()}; mast {w.rig.q('mast')*1000:.0f} mm")


if __name__ == "__main__":
    main()
