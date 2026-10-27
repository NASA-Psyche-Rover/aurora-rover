#!/usr/bin/env python3
"""STEP 2 evidence: render the same moment (cube resting at its start pose) with the same camera geometry in
the existing sim (MuJoCo) and in v2 (MuJoCo live-window look). The v2 Blender Cycles still of the same moment
comes from render_blender.py (see GRAPHICS_DECISION.md). Then builds graphics/side_by_side.png.

    ./vpy tools/graphics_compare.py mujoco      # existing_mujoco.png, v2_mujoco.png
    ./vpy tools/graphics_compare.py combine     # side_by_side.png (needs graphics/v2_blender.png)
"""
import os
import sys

import imageio.v2 as imageio
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import movement_compare as MC  # noqa: E402

OUT = os.path.join(MC.HERE, "graphics")
W, H = 1280, 720
# Blender "cine" shot: 0.72 m, 10 deg above, orbit azimuth 35 deg, 40 mm lens on 36 mm sensor (fovy ~ 27 deg)
DIST, AZ, EL, FOVY = 0.72, 35.0, 10.0, 2 * np.degrees(np.arctan(36 / 16 * 9 / 2 / 40))


def shot(m, d, target):
    m.vis.global_.fovy = FOVY
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = target + np.array([0, 0, 0.035])
    cam.distance, cam.azimuth, cam.elevation = DIST, AZ + 180.0, -EL
    r.update_scene(d, cam)
    img = r.render()
    r.close()
    return img


def mujoco_shots():
    os.makedirs(OUT, exist_ok=True)
    P = MC.load(MC.OLD, "old_gfx")
    w = P.World(g=P.G_PSYCHE)
    w.reset()
    for _ in range(int(1.0 / P.DT)):
        mujoco.mj_step(w.m, w.d)
    imageio.imwrite(os.path.join(OUT, "existing_mujoco.png"), shot(w.m, w.d, w.pos()))
    V = MC.load(os.path.join(MC.HERE, "psyche_cubli_v2.py"), "v2_gfx")
    vw = V.build_world(V.DEFAULT_SITE)
    for _ in range(int(1.0 / V.DT)):
        vw.step()
    imageio.imwrite(os.path.join(OUT, "v2_mujoco.png"), shot(vw.m, vw.d, vw.com()))
    print("wrote", OUT)


def label(img, text):
    import cv2
    img = np.ascontiguousarray(img[..., :3])
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def combine():
    import cv2
    tiles = []
    for f, t in (("existing_mujoco.png", "EXISTING sim (MuJoCo)"), ("v2_mujoco.png", "v2 live window (MuJoCo)"),
                 ("v2_blender.png", "v2 film (Blender Cycles)")):
        im = imageio.imread(os.path.join(OUT, f))[..., :3]
        tiles.append(label(cv2.resize(im, (W // 2, H // 2), interpolation=cv2.INTER_AREA), t))
    imageio.imwrite(os.path.join(OUT, "side_by_side.png"), np.hstack(tiles))
    print("wrote", os.path.join(OUT, "side_by_side.png"))


if __name__ == "__main__":
    {"mujoco": mujoco_shots, "combine": combine}[sys.argv[1]]()
