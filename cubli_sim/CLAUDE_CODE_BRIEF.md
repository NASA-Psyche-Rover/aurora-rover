# Brief for Claude Code: Psyche Cubli v2

Paste this into Claude Code from inside the `MuJoCo Sim` folder.

---
You are working on the AURORA team's Psyche Cubli simulator in this folder. Read `README_v2.md`, `psyche_cubli_v2.py` and `render_blender.py` first.

Rules (do not break these):
1. **MuJoCo is the only source of motion.** Never animate the cube by hand in Blender. Any visual change goes in `render_blender.py`. Any motion change goes in the physics or controller in `psyche_cubli_v2.py`, followed by re-recording.
2. **Keep every number labelled** CAD / DATASHEET / PAPER / YOUR NUMBER / PLACEHOLDER. Never present a placeholder as measured.
3. **Psyche facts come from Shepard et al. 2021:** 278 x 238 x 171 km, 2.287e19 kg, 4.195948 h spin. Surface gravity is computed (0.089–0.123 m/s²). Do not hard-code the old 0.144 value.
4. **Cube contact uses the rounded hull** (corner spheres + edge capsules on contype 1; a box on contype 2 for boulders). Do not switch back to a box on the heightfield: it jitters and the cube floats away in low gravity.
5. **Before claiming anything works**, run `python psyche_cubli_v2.py test` (all rolls OK, balance engages) and render one still with `--still 26 --quality preview`, then look at it.
6. On a Mac the 3D window needs `mjpython`. Blender runs headless with `-b -P render_blender.py -- ...`.

Typical tasks:
- New mission: add steps to `DEMO` or use `record --goto X Y`, then render.
- New shot style: add an entry to `SHOTS` and to the `auto` sequence in `make_cameras`.
- Real masses measured: update the PARAMETERS block, re-run `test`, then re-record.
---
