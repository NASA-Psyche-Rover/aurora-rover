# Psyche Cubli v2: simulator + film renderer

## Quick start (any laptop: macOS, Windows, Linux)

This folder is in the team repo on branch `cubli-sim`, next to `perception/`. Paste one line into a terminal. It downloads the branch, installs everything into a private `.venv`, and opens the MuJoCo sim plus the sensor dashboard:

macOS / Linux:
```
git clone -b cubli-sim https://github.com/NASA-Psyche-Rover/aurora-rover.git && cd aurora-rover/cubli_sim && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && .venv/bin/python run.py
```
Windows (PowerShell):
```
git clone -b cubli-sim https://github.com/NASA-Psyche-Rover/aurora-rover.git; cd aurora-rover/cubli_sim; python -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt; .venv\Scripts\python run.py
```
**Next time:** run only the last part from `aurora-rover/cubli_sim`: `.venv/bin/python run.py`, or `.venv\Scripts\python run.py` on Windows.

**Notes:**
- Python 3.10–3.12 works best.
- The first start takes a minute: it builds the textures once, and YOLO loads on first use.
- NVIDIA GPU: install the CUDA build of PyTorch from pytorch.org before the requirements, so YOLO runs on the GPU.
- Blender (only for the B live view and the film): it's found automatically if it's installed in the usual place, or set `CUBLI_BLENDER` to its path.

**What it uses:**
- the team's perception from `../perception`, unchanged;
- `cubli_sensors/`, a copy of the frozen `~/cubli_sensors` (`cubli_sensors.py`, identical bytes).

To use other copies, set `CUBLI_PERCEPTION_DIR` / `CUBLI_SENSORS_DIR`.

**Rock detection on the camera:**
- **Where:** tab **1 CAMERA**, and the top-left tile of **0 GRID** (the other grid tiles are depth, LiDAR 3D and LiDAR top).
- **What it shows:** the team's `OAKD-obj-detec.py` pipeline from `perception/`: YOLO boxes with their stereo distance, in-path obstacles, the drop/slope edge, and the PROCEED / CAUTION / STOP decision with its reason.
- **What it runs on:** the real 640×480 OAK-D stream. The drawing is laid over the sharper 960×720 render.
- **Y** turns it on or off. YOLO runs in its own thread, so the picture never waits for it.
- **No names (for now):** YOLO boxes are drawn green with their stereo distance only. Until `perception/models/best.pt` holds rock weights, YOLO falls back to COCO class names ("bird", "cow"), so the sim drops them and every detection is an "object". This is in `sensor_bridge.py` only; `perception/` is unchanged.

## Sensor head on a fake gimbal, richer terrain (branch `worktree-sensors-gimbal-terrain`)

**Sensor head.** The Mid-360 and the OAK-D Lite are now one head. The OAK-D sits on a pan/tilt turret on top of the Mid-360, on a 22 mm post. The head rides a slide mast on a gimbal at the cube centre.

- **What S does:** press **S** (L does the same). The mast pushes the head out of *whichever face is on top*, and both sensors switch on when it's fully out. S again: the camera goes back to level, the mast comes down, and both sensors switch off.
- **Levelling:** the gimbal keeps the head level and turns it to face the last roll, hop or balance direction. When the cube balances on an edge, the mast extends further (115 mm on a face, 146 mm on an edge), so the Mid-360 window always sits 19 mm above the cube's highest point.
- **It's fake:** the CAD has no room for it, and on a wheel face the head passes through the reaction wheel. The levelling is kinematic, so no gimbal motors or reaction torques are modelled. Every new number is labelled PLACEHOLDER in the PARAMETERS block.
- **Mass:** the hatch (8 g) and the camera arm (15 g) are gone, so the whole cube is 1266 g instead of 1289 g.

**Sensors you can trust.** Checked by `tools/sensor_check.py`, which now passes 21/21.

| Was | Now |
|---|---|
| The camera hung off a side face 13 cm up, and the cube's own frame filled the bottom-right corner. YOLO called it a "person" at 0.20 m, so detection said STOP. The frame also left holes in the near-ground depth, which the team pipeline reads as "DROP AHEAD". | The camera is 23 cm up and 78 mm above the top face. A segmentation render shows the cube covers 0.00% of the image at all 12 headings. The team pipeline (`aurora-rover/perception`) made 0 drop calls and 0 STOPs in 72 frames. |
| The LiDAR tilted with the cube, so on an edge or a side face half of its scan went into the sky. | It stays level on every face and while balancing. |
| The LiDAR point rate was tied to the dashboard's frame rate. | 200,000 points/s of sim time, the Mid-360 datasheet rate. The view keeps the last 0.5 s of points. |
| Old points survived a stow and redeploy. | The buffer is cleared at every power-up. |
| The depth view's "Ahead: obstacle" check assumed the sensor sat on the cube, so its ground estimate was 13 cm off. | The bridge feeds the module the camera's true height above the terrain (`cube_half`). |
| The camera rendered at 640×480 and was enlarged on screen, so it looked soft. | It renders at 960×720 and is shown 1:1. Detection still gets the real 640×480 OAK-D stream. |

The perception code in `~/cubli_sensors` and `aurora-rover` is unchanged. Every fix is on the sim side, in `sensor_bridge.py`.

**What's still real, not a bug:**
- **Black ring around the cube in the LiDAR views:** the Mid-360 can't look more than 7° below horizontal. From 17 cm up it doesn't reach the ground until about 1.4 m out.
- **Black areas behind rocks and crater rims:** occlusion shadows.
- **Sparse cloud on flat ground:** only rays below horizontal hit flat ground, so about 14k of every 100k rays come back.
- **Rocks labelled "bird", "cow" or "bench":** the team's `models/best.pt` is empty, so YOLO uses COCO classes. "bench 0.36 m" is the cube's own shadow on the ground (low 22° sun).
- **Cube marker in the LiDAR 3D / top views:** it's drawn at the LiDAR itself, not on the ground, because the frozen module assumes the sensor sits on the cube. The points themselves are cast from the true sensor position.

**Terrain.**
- **Craters:** 70 random ones plus 5 showcase craters (0.9–4 m) placed in view of the start.
- **Boulders:** 160 random ones (60% of them 1.2–7 m from the start) plus 10 showcase boulders (5–40 cm). They all keep their collision shapes, so the LiDAR, the camera and the cube all treat them as solid.
- **Pebbles:** 3,000 visual-only pebbles.
- **Labels:** the showcase features are labelled ARTISTIC.

**Dashboard.** A **? KEYS** button at the right end of the tab bar (or the **?** key) lists every key over the current view. Nothing else about the look changed.

**Checks:**
- `python psyche_cubli_v2.py test`: all rolls, balance and release OK.
- `tools/sensor_check.py`: 21/21.
- A brake-phase bug is fixed. The brake could jump over its ±4 rad/s "stopped" window and chatter forever; it now also stops when the wheel speed crosses zero.
- `recordings/demo.npz` and `deploy.npz` are re-recorded with the new head and terrain. Blender reads the gimbal's quaternion, and old takes still render.

### How Blender fits in

- **MuJoCo is the simulator.** Drive the cube in the MuJoCo window; it does all the physics.
- **Blender only draws.** It never simulates anything. It runs in two ways:
  1. **Live view (press B):** Blender opens and redraws the scene in real time with EEVEE (fast, nearly photoreal) while you drive in MuJoCo. You can't drive from Blender. It's a nicer-looking window on the same sim.
  2. **Film:** record a take, then let Blender path-trace it frame by frame with Cycles (slow, full quality):
     - Press **V** in the sim to start a take and **V** again to save it. It lands in `recordings/take_*.npz`. A scripted mission works too: `python psyche_cubli_v2.py record --mission deploy`.
     - Then render the take: `$BLENDER -b -P render_blender.py -- --rec recordings/<take>.npz --quality preview`. Try `--still 9` first for one frame.
     - Then build the video: `python psyche_cubli_v2.py video --frames renders/<take>/frames --out renders/<take>.mp4`.
     - Or double-click `Render Film.command` to do all three for the demo.
- **Can the MuJoCo window and the dashboard be one window?** Not as it stands. The MuJoCo window needs `mjpython`, and OpenCV windows can't open in that process, so the dashboard runs as its own process. The launcher starts both, and keys typed in either drive the sim. MuJoCo 3.13's viewer can draw images (`set_images`), so a picture-in-picture sensor view inside the sim window is possible. The catch is that the sensor rendering would then share the physics window's frame time.

## Branch `psyche-v4-live`: what's new

Double-click **`Run Psyche Cubli.command`**. Three windows can be open at once:

| Window | What it is | How it opens |
|---|---|---|
| MuJoCo sim | The physics, at Psyche gravity in real time (×1). Drive it here. The controls sidebar is on the right. | Opens first |
| Sensor dashboard | Tabs: 1 camera, 2 depth, 3 LiDAR 3D, 4 LiDAR top, 5 3rd person, 6 split, 7 detect, 0 grid. Drag to orbit, scroll to zoom. Keys typed here also drive the sim. | Opens by itself a few seconds later |
| Blender live view | Photoreal EEVEE view that follows the sim live, built with the same scene builders as the film. | Press **B** in the sim (allow 10–20 s to build) |

**Movement:** the bounce and glitches came from the regolith contact model and the 3 mm roughness in the physics ground.
- Fixes: dissipative contact, 0.5 mm physics roughness, and a real-time default instead of ×4. At ×4, Psyche gravity looked like Moon gravity.
- Measured over 4 rolls: airborne phases went from 180 to 21, and micro-hops from 1532 to 591.
- To reproduce: `tools/motion_diag.py` and `tools/contact_sweep.py`.

**Jump:**
- **Enter** jumps onto an edge and balances there (LQR). Enter again lets go.
- **Space** is the hop. At Psyche gravity it leaves the ground for up to about 1.5 s, the centre of mass rises about 3 cm, and it lands about 10 cm away. It's small because one 58 g wheel at about 1900 rpm stores only about 0.016 N·m·s of angular momentum. A 0.3 N·m mechanical brake (`MECH_BRAKE`) adds only about 0.25 s of airtime.

**Graphics:** the physics ground and boulders are unchanged, but now hidden. Visible instead:
- a smooth textured terrain mesh, using `psyche_visuals.py`
- faceted rocks over each collision boulder
- pebbles
- CAD materials without the glow

**7 DETECT** runs the team's current pipeline from GitHub (`~/aurora-rover/perception/OAKD-obj-detec.py`: YOLO, metric stereo depth, drop and slope detection, decision) on the simulated OAK-D. If that clone is missing or fails, it falls back to the local `psyche-objdetec`. In the repo, `models/best.pt` is empty, so YOLO runs the generic COCO classes. COCO has no rock class, and it calls the cube's own frame a "person".

## Branch `psyche-v3-integration`: what changed

| | |
|---|---|
| Movement + graphics | v2 kept entirely, by evidence: `MOVEMENT_DECISION.md`, `GRAPHICS_DECISION.md` |
| Stowed sensors | Mid-360 LiDAR on a 3 cm/s slide mast in a closed bay behind a hatch (+X face). OAK-D Lite on a flip-out arm with pan/tilt (−Z face). Real bodies, so the centre of mass moves when they deploy. |
| Blocked when stowed | Both sensors face bay walls a few cm away: the LiDAR gets 0 points, the camera image is black and has no valid depth. Done with geometry only. |
| Rolls / hops / release | Stow the sensors first, automatically; the sidebar says so. Balancing works with the sensors deployed, or while they deploy. |
| Perception | Your code is untouched: `sensor_bridge.py` is the adapter (see below). Freeze record: `perception_freeze/`. |
| Film | `record --mission deploy`; `render_blender.py` rebuilds the mast, hatch and arm and animates them from the recorded joints. The tuned film values are unchanged. |
| Window | A sidebar on the right lists every control with its key and live status. The Control panel has sliders for the hatch, mast, arm, pan and tilt. **N** shows a sensor picture (camera / depth / LiDAR 3D / LiDAR top). |

**Decisions you need to make** (both are in the PARAMETERS block):
1. `V_BUS = 14.8` is a PLACEHOLDER; your number was 11.1 V. With the Mid-360, OAK-D Lite and their mechanisms (389 g), the cube weighs 1.29 kg. At 11.1 V it cannot roll anywhere on Psyche: it needs 1452–1730 rpm and the motor tops out at 1379. At 14.8 V (4S) it rolls at every site except the north pole, which is marginal.
2. `SENSORS_IN_TOTAL_MASS = False`. Your 900 g listed motors, ESP32, breadboard, wires, batteries and printed sides, but no sensors. Setting it to True makes the budget negative: "everything else" would be −54 g.

**Keys:**

| Group | Keys |
|---|---|
| Drive | arrows = roll, Space = hop, T = go to target |
| Balance | Enter = balance / let go |
| Sensors | S (or L) = sensor head deploy/stow (both switch on when out), O = LiDAR on/off, C = camera on/off, J / ; = pan, I / K = tilt |
| Camera | F = follow / free, N = sensor picture |
| Sim | G = gravity, P = pause, − / = = speed, R = reset, H = status text |
| Record | V = record a take |

Clashes:
- L, K, R and G are also MuJoCo scene-flag shortcuts (additive, skybox, reflection, fog). The window now restores scene flags, visual flags, geom groups and actuator groups every frame, so they only do our action.
- C was "cancel target" in the old `psyche_cubli.py`. 0/1/2 and `,`/`.` from the old sim are not used.

**Adapters** (`sensor_bridge.py`). Your perception code is imported unchanged:
- `cubli_sensors.CubliSensors` runs on `lidar_site` (LiDAR) and on `cubli_cam` + `cam_site` (camera), with `gimbal=False` so poses come from the real joints. LiDAR off uses its own `lidar_source` hook.
- `SimOakDLite.read()` returns `(BGR frame, uint16 depth in mm)`, exactly like `OakDLiteCamera.read()`.
- `ObjDetec.run()` calls `Detector`, `TerrainAnalyzer`, `should_proceed` and the distance helpers from `MuJoCo Sim/psyche-objdetec`. It needs ultralytics and onnxruntime; the system `python3` has both, the sim venv does not.

**Checks:**
```
python3 -B tools/perception_freeze.py check              # perception files unchanged (git diff empty)
python3 -B tools/perception_regression.py compare before after
./vpy tools/movement_tests.py                            # rolls x4, balance, release, hop, idle drift
python3 -B tools/sensor_check.py                         # deploy/stow, blocked when stowed, balance deployed
CUBLI_SELFTEST=1 python "$VENV/bin/mjpython" psyche_cubli_v2.py   # window: scripted keys + log, closes after 15 s
```
(`vpy` = `NASA Psyche/venv/bin/python`. The venv's `mjpython` launcher has a space in its shebang path, so call it through the venv python, as `Run Psyche Cubli.command` now does. The venv also needed `opencv-python`, for `cubli_sensors`.)

---

Unzip into `NASA Psyche/Stage 3/MuJoCo Sim/`. The old `psyche_cubli.py` is not touched.

Two tools, one source of truth:

| | What it is | Use it for |
|---|---|---|
| `psyche_cubli_v2.py` | **MuJoCo physics**: your CAD cube, GM3506 motor model, ESP32-style controller, Psyche gravity from its real shape | driving the cube, testing control, recording takes |
| `render_blender.py` | **Blender Cycles path tracer** that replays a recorded take frame-for-frame | the film: photoreal surface, sunlight, dust, motion blur, camera cuts |

MuJoCo decides every motion. Blender only paints it. Nothing in the film is animated by hand.

## 1. Drive it (real time)
Double-click **`Run Psyche Cubli.command`** (first time: right-click, Open, Open).
Arrows roll, Enter balances on an edge, Space hops, T goes to the camera target, G changes gravity, **V records a take**.
Options from Terminal: `mjpython psyche_cubli_v2.py --site north_pole --slope 15`

## 2. Make the film
1. Install Blender 4.2 or newer: https://www.blender.org/download/ (or `brew install --cask blender`)
2. Double-click **`Render Film.command`**. It records the demo mission, renders a preview with the orbit intro, then opens the mp4.
3. For full quality, from Terminal: `./"Render Film.command" final` (or `film` for 600 samples).

Manual control:
```
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
python psyche_cubli_v2.py record                                   # -> recordings/demo.npz
$BLENDER -b -P render_blender.py -- --rec recordings/demo.npz --still 26        # one test frame
$BLENDER -b -P render_blender.py -- --rec recordings/demo.npz --quality final --intro
python psyche_cubli_v2.py video --frames renders/demo/frames --out renders/demo.mp4
```
Useful flags: `--speed 1` (true real time; default 2x), `--start 20 --end 45` (sim seconds), `--cam auto|cine|low|wide`,
`--stars none` (what a real camera would show), `--engine eevee` (fast drafts), `--save-blend` (open and tweak in Blender).

**Render time.** The demo is 82 s of sim time, which is 41 s of video at 2x, or about 1,230 frames. On a MacBook Air, expect roughly 2–5 s per frame at `preview` and 20–60 s at `final`. Do short `--start/--end` clips first, or run `final` overnight or on a lab PC with an NVIDIA GPU (Blender is free on Windows/Linux too: copy this folder over).

## What is physical vs artistic
**Physical:** cube geometry and mass layout (CAD), GM3506 torque/speed/current, electrical braking, 200 Hz control with IMU noise, Psyche gravity and spin from the Shepard et al. 2021 shape, real curvature, sunlight at 2.9 AU (162 W/m², 0.18° sun, so razor-sharp shadows), no atmosphere (black sky, shadows lit only by ground bounce), dust thrown in ballistic arcs under Psyche gravity, wheel motion blur from real wheel speed.

**Artistic or unknown:** small-scale terrain, boulder and pebble shapes (no spacecraft has imaged Psyche's surface yet; the NASA Psyche spacecraft arrives in 2029), regolith friction, camera moves, faint stars.

## Numbers you can replace (top of `psyche_cubli_v2.py`)
`WHEEL_MASS` 33 g, `PAYLOAD_MASS` 250 g, `ROTOR_MASS` 25 g, `I_LIMIT` 2 A, `V_BUS` 11.1 V, `MECH_BRAKE` 0 (set e.g. 0.3 N·m to simulate a brake).
If the CAD changes: `pip install trimesh fast-simplification`, then `python build_assets.py path/to/assembly.STL`
