#!/usr/bin/env python3
"""RULE 0: the perception stack is frozen. This records it and proves it was not touched.

    python tools/perception_freeze.py record   # writes perception_freeze/hashes.json + INVENTORY.md
    python tools/perception_freeze.py check    # re-hashes, prints git diff/status per repo, exits 1 on change

Only reads the perception folders; never writes to them.
"""
import ast
import json
import os
import hashlib
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "perception_freeze")
HOME = os.path.expanduser("~")
STAGE3 = os.path.join(HOME, "Desktop", "Aayou", "ASU", "EPICS", "NASA Psyche", "Stage 3")

# (repo root, files relative to it). Models are hashed too: a changed weight file changes results.
REPOS = {
    "objdetec (MuJoCo Sim copy, used by the adapters)": (
        os.path.join(STAGE3, "MuJoCo Sim", "psyche-objdetec"),
        ["main.py", "OAKD-obj-detec.py", "src/camera_stream.py", "src/decision.py", "src/detect.py",
         "src/terrain_risk.py", "src/utils.py", "scripts/terrain_demo.py", "scripts/quantize_profile.py",
         "legacy/terrain_risk_cubli.py", "models/best.pt", "models/yolov8n.pt", "models/midas_small.onnx",
         "models/midas_small.onnx.data", "models/midas_small_openvino/midas_small.xml",
         "models/midas_small_openvino/midas_small.bin"]),
    "objdetec (~/psyche-objdetec, older fork)": (
        os.path.join(HOME, "psyche-objdetec"),
        ["main.py", "src/camera_stream.py", "src/decision.py", "src/detect.py", "src/terrain_risk.py",
         "src/utils.py", "models/best.pt", "models/yolov8n.pt"]),
    "cubli_sensors (LiDAR / stereo depth / camera)": (
        os.path.join(HOME, "cubli_sensors"),
        ["cubli_sensors.py", "demo_sensors.py", "psyche_terrain.py", "tools/import_cubli_stl.py",
         "assets/cubli_frame.obj", "assets/cubli_wheel.obj", "assets/cubli_motor.obj",
         "assets/cubli_hardware.obj"]),
}
ROLE = {  # what each file does in the perception stack (for the inventory)
    "src/detect.py": "object detection (YOLOv8)",
    "src/terrain_risk.py": "depth analysis (MiDaS-small ONNX) -> slope/roughness/drop risk",
    "src/decision.py": "navigation gate from risk + proximity + density",
    "src/utils.py": "detection post-processing (density, depth-sampled proximity), HUD drawing",
    "src/camera_stream.py": "camera input (webcam / OAK-D Lite RGB + stereo mm) + metric object distance",
    "main.py": "live detection + terrain-risk pipeline",
    "OAKD-obj-detec.py": "OAK-D Lite detection + real depth test",
    "scripts/terrain_demo.py": "terrain-risk live demo / benchmark",
    "scripts/quantize_profile.py": "MiDaS backend benchmark",
    "legacy/terrain_risk_cubli.py": "legacy Depth-Anything terrain risk (reference only)",
    "cubli_sensors.py": "simulated camera image processing, stereo depth analysis, LiDAR",
    "demo_sensors.py": "sensor demo; ray-based obstacle/drop check (Rover.check_target)",
    "psyche_terrain.py": "demo terrain for the sensor module",
    "tools/import_cubli_stl.py": "CAD import for the sensor demo",
}


def git(root, *args):
    r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    return r.stdout.strip()


def defs(path):
    if not path.endswith(".py"):
        return []
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError:
        return ["(not parseable)"]
    out = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            meths = [m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            out.append(f"class {n.name}: " + ", ".join(meths))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"def {n.name}")
    return out


def snapshot():
    snap = {}
    for name, (root, files) in REPOS.items():
        ent = dict(root=root, head=git(root, "rev-parse", "HEAD"), branch=git(root, "rev-parse", "--abbrev-ref", "HEAD"),
                   status=git(root, "status", "--porcelain"), files={})
        for f in files:
            p = os.path.join(root, f)
            ent["files"][f] = git(root, "hash-object", f) if os.path.exists(p) else None
        diff = subprocess.run(["git", "-C", root, "diff", "HEAD", "--", *files], capture_output=True, text=True).stdout
        ent["diff_lines"] = len(diff.splitlines())
        ent["diff_sha1"] = hashlib.sha1(diff.encode()).hexdigest()   # pre-existing local edits must stay identical
        snap[name] = ent
    return snap


def record():
    os.makedirs(OUT, exist_ok=True)
    snap = snapshot()
    json.dump(snap, open(os.path.join(OUT, "hashes.json"), "w"), indent=1)
    lines = ["# Perception stack inventory (FROZEN: do not edit, rename, move or reformat)", "",
             "Recorded by `tools/perception_freeze.py record`. Hashes are `git hash-object` (blob SHA-1).", ""]
    for name, ent in snap.items():
        lines += [f"## {name}", f"`{ent['root']}`  branch `{ent['branch']}` @ `{ent['head'][:10]}`", ""]
        if ent["status"]:
            lines += ["Pre-existing uncommitted changes (were already there, left as is):", "```", ent["status"], "```", ""]
        lines += ["| file | role | git blob hash | functions / classes |", "|---|---|---|---|"]
        for f, h in ent["files"].items():
            d = "<br>".join(defs(os.path.join(ent["root"], f))) or "-"
            lines.append(f"| `{f}` | {ROLE.get(f, 'model weights' if f.startswith('models') else '')} | `{h}` | {d} |")
        lines.append("")
    open(os.path.join(OUT, "INVENTORY.md"), "w").write("\n".join(lines))
    print(f"recorded {sum(len(e['files']) for e in snap.values())} files -> {OUT}")


def check():
    base = json.load(open(os.path.join(OUT, "hashes.json")))
    now = snapshot()
    bad = 0
    for name, ent in base.items():
        cur = now[name]
        print(f"== {name}\n   {ent['root']}")
        print(f"   HEAD {ent['head'][:10]} -> {cur['head'][:10]}   {'same' if ent['head'] == cur['head'] else 'CHANGED'}")
        print(f"   status before: {ent['status'] or '(clean)'!r}")
        print(f"   status now   : {cur['status'] or '(clean)'!r}")
        files = list(ent["files"])
        pre_mod = [ln.split(maxsplit=1)[1] for ln in ent["status"].splitlines() if ln.split(maxsplit=1)[1] in files]
        if cur["diff_lines"] == 0:
            print("   git diff HEAD -- <perception files>: (empty)")
        else:
            same = cur["diff_sha1"] == ent["diff_sha1"]
            bad += 0 if same else 1
            print(f"   git diff HEAD -- <perception files>: {cur['diff_lines']} lines = your uncommitted edit to "
                  f"{', '.join(pre_mod)} that existed before this work; "
                  f"{'byte-identical to the baseline' if same else 'CHANGED since the baseline'}")
        for f, h in ent["files"].items():
            if cur["files"][f] != h:
                bad += 1
                print(f"   CHANGED {f}: {h} -> {cur['files'][f]}")
        if all(cur["files"][f] == h for f, h in ent["files"].items()):
            print(f"   all {len(files)} file hashes identical to the recorded baseline")
    print("RESULT:", "UNCHANGED" if bad == 0 else f"{bad} FILES CHANGED")
    return bad


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "record":
        record()
    else:
        sys.exit(1 if check() else 0)
