#!/usr/bin/env python3
"""DONE check 3: deploy / stow the sensor head in the sim and run the FROZEN perception code on it.

    python3 -B tools/sensor_check.py          # system python3: has mujoco, scipy, opencv, ultralytics, onnxruntime
    (with the sim venv it still runs; object detection is skipped there because ultralytics/onnxruntime are missing)

Checks: stowed -> LiDAR and camera off (no points, black frame, no depth); S deploys -> head out of the TOP face,
level (fake gimbal), both sensors switch on, the camera does not see the cube, point cloud + camera image + stereo
depth + detection/terrain-risk pipeline; balance holds with sensors deployed (deploy-then-balance and
balance-then-deploy); release and rolls auto-stow first; after rolling onto another face the head again comes out of
the top. Saves the views to sensor_check/.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import cv2  # noqa: E402
import psyche_cubli_v2 as V  # noqa: E402
from sensor_bridge import ObjDetec, SensorBridge, SimOakDLite  # noqa: E402

OUT = os.path.join(HERE, "sensor_check")
RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def run(w, c, secs):
    t1 = w.d.time + secs
    while w.d.time < t1:
        c.update()
        w.step()


def head_out_of_top(w):
    """(Mid-360 window height above the cube's highest corner in m, gimbal up-axis . world up)."""
    m, d = w.m, w.d
    window = d.site_xpos[m.site("lidar_site").id][2]
    top = w.corners()[0][:, 2].max()
    return float(window - top), float(d.xmat[m.body("gimbal").id].reshape(3, 3)[2, 2])


def cube_pixels(w, br):
    """Fraction of the OAK-D image covered by the cube itself (frame + wheels), from a segmentation render."""
    import mujoco
    m, d = w.m, w.d
    r = mujoco.Renderer(m, br.cam._h, br.cam._w)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    r.update_scene(d, camera="cubli_cam", scene_option=opt)
    r.enable_segmentation_rendering()
    seg = r.render()[..., 0]
    r.close()
    own = {g for g in range(m.ngeom) if m.geom_bodyid[g] == w.cube or m.body(m.geom_bodyid[g]).name.startswith("wheel")}
    return float(np.isin(seg, list(own)).mean())


def look(c, br, tag, det=None):
    """Read every sensor through the team's code; save the views; return the numbers."""
    for _ in range(14):                       # fill the LiDAR integration window (0.5 s of sim time)
        run(br.w, c, 0.036)
        pts = br.points()
    oak = SimOakDLite(br)
    frame, depth_mm = oak.read()
    valid = float((depth_mm > 0).mean())
    for name, img in (("camera", br.cam_view(1)), ("depth", br.cam_view(2)),
                      ("lidar3d", br.lidar_view(3)), ("lidar_bev", br.lidar_view(4))):
        cv2.imwrite(os.path.join(OUT, f"{tag}_{name}.png"), img)
    res = det.run(frame, depth_mm) if det else None
    rng = np.linalg.norm(pts, axis=1) if len(pts) else np.zeros(1)
    return dict(points=len(pts), max_range=float(rng.max()), depth_valid=valid, rgb_mean=float(frame.mean()),
                objdetec=res)


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        det = ObjDetec()
    except Exception as e:                    # noqa: BLE001  (venv without ultralytics / onnxruntime)
        print(f"(object detection skipped: {e})")
        det = None
    w = V.build_world()
    c = V.Cubli(w)
    br = SensorBridge(w)
    run(w, c, 1.0)
    r = w.rig
    print(f"cube {w.total_mass()*1000:.0f} g (sensors {V.SENSOR_MASS*1000:.0f} g, "
          f"SENSORS_IN_TOTAL_MASS={V.SENSORS_IN_TOTAL_MASS}), V_BUS {V.V_BUS} V")

    s = look(c, br, "1_stowed", None)
    check("stowed: LiDAR off", s["points"] == 0, f"{s['points']} points")
    check("stowed: camera off (black, no depth)", s["depth_valid"] == 0 and s["rgb_mean"] == 0,
          f"depth valid {100*s['depth_valid']:.1f}%, mean brightness {s['rgb_mean']:.0f}/255")

    com0 = w.com().copy()
    r.deploy()
    run(w, c, 5.0)
    above, level = head_out_of_top(w)
    check("deploy: head out of the top face", r.deployed and above > 0.015,
          f"mast {r.q('mast')*1000:.1f} mm, Mid-360 window {above*1000:.0f} mm above the cube's top")
    check("deploy: gimbal level", level > 0.9995, f"head up-axis . world up = {level:.5f}")
    check("deploy: both sensors switched on by themselves", r.lidar_on and r.cam_on,
          f"LiDAR {r.state('lidar')}, camera {r.state('cam')}, tilt {math.degrees(r.q('cam_tilt')):.0f} deg")
    f = cube_pixels(w, br)
    check("deploy: the camera does not see the cube", f < 0.002, f"cube covers {100*f:.2f}% of the image")
    check("deploy: centre of mass moved (sensors are real bodies)", np.linalg.norm(w.com() - com0) > 0.005,
          f"CoM shift {np.round((w.com() - com0)*1000, 1).tolist()} mm")
    s = look(c, br, "2_deployed", det)
    check("deployed: LiDAR point cloud", s["points"] > 2000, f"{s['points']} points, max range {s['max_range']:.1f} m")
    check("deployed: camera image + stereo depth", s["depth_valid"] > 0.3 and s["rgb_mean"] > 10,
          f"depth valid {100*s['depth_valid']:.0f}%, mean brightness {s['rgb_mean']:.0f}/255")
    if det:
        o = s["objdetec"]
        check("deployed: team detection + terrain-risk pipeline ran on the sim frame", o["decision"] in
              ("STOP", "CAUTION", "PROCEED"), f"labels {o['labels']}, terrain {o['terrain']} (risk {o['risk']:.2f}), "
              f"decision {o['decision']}, nearest {o['nearest_m']}")
    r.aim(dpan=0.5, dtilt=0.1)
    run(w, c, 1.0)
    check("pan/tilt", abs(r.q("cam_pan") - 0.5) < 0.05,
          f"pan {math.degrees(r.q('cam_pan')):.0f} deg, tilt {math.degrees(r.q('cam_tilt')):.0f} deg")
    r.aim(dpan=-0.5, dtilt=-0.1)
    run(w, c, 1.0)

    # balance with the sensors already deployed
    for attempt in range(2):
        c.balance(np.array([0.0, 1.0, 0.0]))
        t0 = w.d.time
        while (c.action is None or c.action.get("phase") != "balance") and w.d.time - t0 < 40 and \
                (c.action or c.queue):
            c.update()
            w.step()
        if c.action is not None and c.action.get("phase") == "balance":
            break
        print("    retry:", c.last_result)
        run(w, c, 1.0)
    run(w, c, 5.0)
    held = c.action is not None and c.action.get("phase") == "balance"
    check("balance holds with LiDAR + camera deployed (5 s)", held, c.status)
    s = look(c, br, "3_balancing_deployed", det)
    # stow while balancing (mass moves under the LQR), then release
    c.release_balance()
    print("   ", c.status)
    V.run_script(w, c, [], t_max=60)
    check("release auto-stows first, then lands on a face", r.stowed and "OK" in c.last_result,
          f"stowed={r.stowed}; {c.last_result}")

    # balance first, deploy while balancing (v2 retries once with a learned bigger margin if a jump falls short)
    run(w, c, 1.0)
    for attempt in range(2):
        c.balance(np.array([0.0, -1.0, 0.0]))
        t0 = w.d.time
        while (c.action is None or c.action.get("phase") != "balance") and w.d.time - t0 < 40 and \
                (c.action or c.queue):
            c.update()
            w.step()
        if c.action is not None and c.action.get("phase") == "balance":
            break
        print("    retry:", c.last_result)
        run(w, c, 1.0)
    r.deploy()
    run(w, c, 6.0)
    held = c.action is not None and c.action.get("phase") == "balance"
    above, level = head_out_of_top(w)
    check("balance holds while the sensors deploy (CoM moving)", held and r.deployed,
          f"{c.status}; mast {r.q('mast')*1000:.0f} mm")
    check("balancing on an edge: head still level and above the cube", level > 0.9995 and above > 0.015,
          f"up . up = {level:.5f}, window {above*1000:.0f} mm above the highest corner")
    s = look(c, br, "3b_balancing_deployed", None)
    check("balancing on an edge: full LiDAR scan + camera", s["points"] > 5000 and s["depth_valid"] > 0.3,
          f"{s['points']} points, depth valid {100*s['depth_valid']:.0f}%")
    c.release_balance()
    V.run_script(w, c, [], t_max=60)

    # a roll requested while deployed must stow first
    run(w, c, 1.0)
    r.deploy()
    run(w, c, 5.0)
    c.roll(np.array([1.0, 0.0, 0.0]))
    msgs = set()
    t0 = w.d.time
    while (c.action or c.queue) and w.d.time - t0 < 120:
        c.update()
        w.step()
        if "waits" in c.status:
            msgs.add(c.status.split(" (")[0])
    if "FAILED" in c.last_result:                 # v2 retries once with its learned margin (as record() does)
        print("    retry:", c.last_result)
        V.run_script(w, c, [("roll", [1.0, 0.0, 0.0])])
    check("roll while deployed: auto-stow, then roll", "OK" in c.last_result and bool(msgs),
          f"HUD said '{sorted(msgs)[0] if msgs else '-'}...'; then {c.last_result}")
    s = look(c, br, "4_stowed_again", det)
    check("stowed again: LiDAR + camera off", s["points"] == 0 and s["depth_valid"] == 0,
          f"{s['points']} points, depth valid {100*s['depth_valid']:.1f}%")

    # the point of the gimbal: on every face the cube lands on, the head comes out of the top
    for d in ([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]):
        V.run_script(w, c, [("roll", d)])
        up_face = max(range(6), key=lambda i: ((-1) ** i) * w.R()[2, i // 2])
        r.deploy()
        run(w, c, 5.0)
        above, level = head_out_of_top(w)
        s = look(c, br, f"5_face_{'xyz'[up_face // 2]}{'+-'[up_face % 2]}", None)
        f = cube_pixels(w, br)
        check(f"after a roll ({c.last_result.split(':')[0]}), {'+-'[up_face % 2]}{'XYZ'[up_face // 2]} face up: "
              f"head out of the top, level, sensing", above > 0.015 and level > 0.9995 and s["points"] > 5000
              and s["depth_valid"] > 0.3 and f < 0.002,
              f"window {above*1000:.0f} mm above, level {level:.5f}, {s['points']} pts, depth {100*s['depth_valid']:.0f}%,"
              f" cube in image {100*f:.2f}%")
        r.stow()
        run(w, c, 5.0)
    br.close()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed. Views saved in {OUT}/")
    return n_ok == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
