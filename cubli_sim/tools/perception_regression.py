#!/usr/bin/env python3
"""Runs the EXISTING perception code, unmodified, and records its results so before/after can be compared.

    python3 -B tools/perception_regression.py run before      # -> perception_freeze/runs/before.json
    python3 -B tools/perception_regression.py run after
    python3 -B tools/perception_regression.py compare before after

What it runs (headless; the repo scripts themselves need a webcam / window):
  objdetec  : Detector.detect_objects, TerrainAnalyzer.get_risk_assessment, compute_cluster_density,
              estimate_object_proximity, should_proceed, estimate_object_distances_mm on every image in
              data/sample_images (the inputs main.py uses when there is no webcam).
  cubli_sensors: demo_sensors.py --snapshot (17 PNGs) + CubliSensors rgb()/depth()/lidar() on the demo scene.
Needs a Python with ultralytics, onnxruntime, opencv, mujoco (the system python3 here has all four).
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "perception_freeze", "runs")
HOME = os.path.expanduser("~")
OBJDETEC = os.path.join(HOME, "Desktop", "Aayou", "ASU", "EPICS", "NASA Psyche", "Stage 3", "MuJoCo Sim", "psyche-objdetec")
SENSORS = os.path.join(HOME, "cubli_sensors")


def h(a, dec=None):
    a = np.asarray(a)
    if dec is not None:
        a = np.round(a.astype(np.float64), dec)
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def run_objdetec():
    cwd = os.getcwd()
    os.chdir(OBJDETEC)                       # the code uses repo-relative model paths
    sys.path.insert(0, OBJDETEC)
    try:
        import cv2
        from src.detect import Detector
        from src.terrain_risk import TerrainAnalyzer
        from src.decision import should_proceed
        from src.utils import compute_cluster_density, estimate_object_proximity, get_sample_images
        from src.camera_stream import estimate_object_distances_mm
        det, ta = Detector("models/best.pt"), TerrainAnalyzer()
        out = {}
        for p in get_sample_images():
            img = cv2.imread(p)
            boxes, labels, _ = det.detect_objects(img)
            depth, status, risk, _ = ta.get_risk_assessment(cv2.resize(img, (320, 240)))   # main.py's input size
            fs = (img.shape[1], img.shape[0])
            dens = compute_cluster_density(boxes, fs[0] * fs[1])
            prox = estimate_object_proximity(boxes, depth, fs)
            fake_mm = (1000 + 4000 * (1 - depth)).astype(np.uint16)                       # fixed synthetic mm map
            dist = estimate_object_distances_mm(boxes, cv2.resize(fake_mm, fs), fs)
            out[os.path.basename(p)] = dict(
                boxes=boxes, labels=labels, status=status, risk=round(float(risk), 5), density=round(float(dens), 6),
                proximity=round(float(prox), 5), decision=should_proceed(risk, prox, dens),
                distances_mm=[None if d is None else round(d, 1) for d in dist], depth_hash=h(depth, 4))
        return out
    finally:
        os.chdir(cwd)
        sys.path.remove(OBJDETEC)


def run_sensors(tag):
    snapdir = os.path.join(RUNS, f"{tag}_snapshots")
    os.makedirs(snapdir, exist_ok=True)
    for f in os.listdir(snapdir):
        os.remove(os.path.join(snapdir, f))
    subprocess.run([sys.executable, "-I", "-B", os.path.join(SENSORS, "demo_sensors.py"), "--snapshot",
                    os.path.join(snapdir, "s.png")], check=True, capture_output=True)
    pngs = {f: hashlib.sha1(open(os.path.join(snapdir, f), "rb").read()).hexdigest()[:16]
            for f in sorted(os.listdir(snapdir))}
    sys.path.insert(0, SENSORS)
    import mujoco
    import demo_sensors as D
    from cubli_sensors import CubliSensors
    xml, assets, _ = D.build_xml()
    m = mujoco.MjModel.from_xml_string(xml, assets)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cs = CubliSensors(m, d)
    rv = D.Rover(m, d, cs)
    rv.reset(2, 0)
    rv.deploy_now()
    pts = [cs.lidar() for _ in range(10)][-1]
    z = cs.depth()
    rgb = cs.rgb()
    out = dict(snapshots=pngs, lidar_n=int(len(pts)), lidar_hash=h(pts, 4), depth_hash=h(z, 4),
               depth_valid=round(float((z > 0).mean()), 5), rgb_hash=h(rgb),
               check_target={f"{i},{j}": rv.check_target(i, j) for i, j in ((7, 0), (20, 0), (3, 0))})
    cs.close()
    return out


def main():
    if sys.argv[1] == "run":
        tag = sys.argv[2]
        os.makedirs(RUNS, exist_ok=True)
        res = dict(objdetec=run_objdetec(), cubli_sensors=run_sensors(tag))
        path = os.path.join(RUNS, f"{tag}.json")
        json.dump(res, open(path, "w"), indent=1, default=str)
        print(json.dumps(res, indent=1, default=str)[:3000])
        print("saved", path)
    elif sys.argv[1] == "compare":
        a, b = (json.load(open(os.path.join(RUNS, f"{t}.json"))) for t in sys.argv[2:4])
        diffs = []

        def walk(x, y, path):
            if isinstance(x, dict):
                for k in sorted(set(x) | set(y)):
                    walk(x.get(k), y.get(k), f"{path}/{k}")
            elif x != y:
                diffs.append((path, x, y))
        walk(a, b, "")
        for p, x, y in diffs:
            print(f"DIFF {p}: {x} -> {y}")
        print(f"{sys.argv[2]} vs {sys.argv[3]}: {'IDENTICAL' if not diffs else f'{len(diffs)} differences'} "
              f"({len(a['objdetec'])} images, {len(a['cubli_sensors']['snapshots'])} sensor snapshots)")
        sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
