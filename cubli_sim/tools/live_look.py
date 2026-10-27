#!/usr/bin/env python3
"""Offscreen renders of what the live MuJoCo window shows (same scene, same renderer) -> graphics/live_*.png"""
import os
import sys

import imageio.v2 as imageio
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import psyche_cubli_v2 as V  # noqa: E402


def main():
    w = V.build_world()
    c = V.Cubli(w)
    for _ in range(int(1.5 / V.DT)):
        w.step()
    w.rig.deploy_lidar()
    w.rig.deploy_cam()
    for _ in range(int(5 / V.DT)):
        c.update()
        w.step()
    r = mujoco.Renderer(w.m, 720, 1280)
    shots = {"live_hero.png": (0.72, 35.0, 10.0, 27.0), "live_wide.png": (2.6, 120.0, 8.0, 45.0)}
    for name, (dist, az, el, fovy) in shots.items():
        w.m.vis.global_.fovy = fovy
        cam = mujoco.MjvCamera()
        cam.lookat[:] = w.com() + np.array([0, 0, 0.035])
        cam.distance, cam.azimuth, cam.elevation = dist, az + 180.0, -el
        r.update_scene(w.d, cam)
        imageio.imwrite(os.path.join(HERE, "graphics", name), r.render())
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    r.update_scene(w.d, "cubli_cam", opt)            # what the deployed OAK-D sees
    imageio.imwrite(os.path.join(HERE, "graphics", "live_oakd.png"), r.render())
    print("wrote graphics/live_hero.png, live_wide.png, live_oakd.png")


if __name__ == "__main__":
    main()
