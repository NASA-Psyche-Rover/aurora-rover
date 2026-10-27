"""
Builds the sim's assets from the team's Cubli assembly STL.

    pip install trimesh fast-simplification
    python build_assets.py path/to/DISPLAY_Cubli_Assembly_Full.STL

Writes assets/frame.stl, assets/wheel.stl and assets/mass_parts.json.
Run it again whenever the CAD changes. The sim itself does not need trimesh.

How the STL is split (by geometry, the STL has no part names):
  wheel   : the 126 mm ring on each wheel axis (rotates)
  rotor   : the 40 mm motor bell + 13 mm hub on each axis (rotates with the wheel)
  motor   : the rest of each GM3506 (stator, base, encoder) (fixed to the frame)
  payload : the boxes in the middle of the cube (battery / electronics bay)
  struct  : everything else (printed panels, frame, mounts, screws)
Masses are NOT taken from here. The sim assigns a mass to each class from its
PARAMETERS block; this file only stores where each class sits and its shape.
"""
import json
import os
import sys

import numpy as np
import trimesh

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
FRAME_FACES = 60000
WHEEL_FACES = 8000
# wheel axes found in the CAD, in the cube frame (STL centred, mm): +Z face, +Y face, -X face
AXES = [(2, +1), (1, +1), (0, -1)]


def on_axis(c, tol=6.0):
    for ax, sgn in AXES:
        o = [i for i in range(3) if i != ax]
        if abs(c[o[0]]) < tol and abs(c[o[1]]) < tol and c[ax] * sgn > 20:
            return ax
    return None


def classify(p, c):
    e = np.sort(p.extents)[::-1]
    ax = on_axis(c)
    if ax is not None and abs(c[ax]) < 70:          # 70 mm keeps the face panels out
        if 120 < e[0] < 130:
            return f"wheel{ax}"
        if (39 < e[0] < 41 and 12.5 < e[2] < 14) or (12 < e[0] < 14 and 16 < e[1] < 18):
            return f"rotor{ax}"
        if e[0] < 41:
            return f"motor{ax}"
        return "struct"                               # the 102 mm motor mounts
    if 59 < e[0] < 68 or (49 < e[0] < 51 and 33 < e[1] < 35):
        return "payload"
    return "struct"


def lump(p):
    """volume, centroid, inertia about centroid (unit density), robust to open meshes."""
    if p.is_watertight and p.volume > 0:
        q = p
        fill = 1.0
    else:
        q = p.convex_hull
        fill = 0.35                                  # open/printed parts: hull x a fill guess
    v = q.volume * fill
    c = q.center_mass
    I = q.moment_inertia * fill
    return v, c, I


def main(path):
    os.makedirs(OUT, exist_ok=True)
    m = trimesh.load(path, force="mesh")
    C = (m.bounds[0] + m.bounds[1]) / 2.0
    m.apply_translation(-C)
    print(f"loaded {len(m.faces)} faces, extents {m.extents.round(2)} mm")
    parts = m.split(only_watertight=False)

    classes = {}
    meshes = {"frame": [], "wheel": []}
    for p in parts:
        if len(p.faces) < 4 or p.area < 1.0:
            meshes["frame"].append(p)
            continue
        c = p.centroid
        k = classify(p, c)
        v, cm, I = lump(p)
        a = classes.setdefault(k, dict(V=0.0, S1=np.zeros(3), S2=np.zeros((3, 3))))
        # second moment about the cube origin, unit density: I_c + v(|c|^2 E - c c^T)
        a["V"] += v
        a["S1"] += v * cm
        a["S2"] += I + v * (cm @ cm * np.eye(3) - np.outer(cm, cm))
        if k.startswith("wheel2") or k.startswith("rotor2"):
            meshes["wheel"].append(p)
        elif k.startswith("wheel") or k.startswith("rotor"):
            pass                                     # the other two wheels reuse the +Z mesh
        else:
            meshes["frame"].append(p)

    # wheel template: +Z wheel, moved so its centre is the origin and its axis is +Z
    wheel = trimesh.util.concatenate(meshes["wheel"])
    wz = [p for p in parts if classify(p, p.centroid) == "wheel2"][0]
    wc = wz.center_mass.copy()
    wheel.apply_translation(-wc)
    frame = trimesh.util.concatenate(meshes["frame"])

    def simplify(mesh, n):
        if len(mesh.faces) <= n:
            return mesh
        return mesh.simplify_quadric_decimation(face_count=n)

    frame = simplify(frame, FRAME_FACES)
    wheel = simplify(wheel, WHEEL_FACES)
    frame.apply_scale(1e-3)
    wheel.apply_scale(1e-3)
    frame.export(os.path.join(OUT, "frame.stl"))
    wheel.export(os.path.join(OUT, "wheel.stl"))

    centres = {}
    for ax, sgn in AXES:
        w = classes[f"wheel{ax}"]
        centres[ax] = (w["S1"] / w["V"] * 1e-3).tolist()

    out = dict(
        source=os.path.basename(path),
        units="m, unit density (multiply by kg/m^3 = mass/V)",
        edge=float(m.extents.max() * 1e-3),
        wheel_axes=[dict(axis=ax, sign=sgn, centre=centres[ax]) for ax, sgn in AXES],
        classes={k: dict(V=a["V"] * 1e-9, S1=(a["S1"] * 1e-12).tolist(), S2=(a["S2"] * 1e-15).tolist())
                 for k, a in classes.items()},
    )
    with open(os.path.join(OUT, "mass_parts.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, a in sorted(classes.items()):
        print(f"  {k:8s} V={a['V']/1000:7.1f} cc  centroid={np.round(a['S1']/a['V'],1)} mm")
    print(f"frame {len(frame.faces)} faces, wheel {len(wheel.faces)} faces -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
