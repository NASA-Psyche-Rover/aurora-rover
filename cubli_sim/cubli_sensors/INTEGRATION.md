# Integrating `cubli_sensors` with your MuJoCo sim

`CubliSensors` only **reads** `data`. It never calls `mj_step`. You step; you call the sensors.

## 1. What to add to your XML

```xml
<mujoco>
  <statistic extent="1"/>                       <!-- zfar below is scaled by extent -->
  <visual>
    <global offwidth="1280" offheight="960"/>   <!-- offscreen buffer >= sensor size -->
    <map znear="0.01" zfar="60"/>               <!-- 1 cm near plane; depth up to 60 m -->
  </visual>

  <worldbody>
    <body name="cubli" pos="0 0 0.075">         <!-- your existing cube body -->
      <!-- EVERY visual geom of the cube goes in group 3: sensors hide it, world view shows it -->
      <geom type="box" size=".075 .075 .075" group="3" .../>

      <!-- Sensor head. Axes of this site DEFINE the sensor frame: x forward, y left, z up. -->
      <site name="lidar_site" pos="0 0 0.10" group="3"/>

      <!-- Front camera, 15 deg down. xyaxes = (camera right, camera up) in the body frame:
           right = -y, up = (sin15, 0, cos15).  fovy is overwritten with 55 by the module. -->
      <camera name="cubli_cam" pos="0.03 0 0.095" xyaxes="0 -1 0 0.2588 0 0.9659" fovy="55"/>
    </body>
  </worldbody>
</mujoco>
```

Notes
- The site and camera must be on the cube body (or any rigid child of it). Their mounting
  is measured once from the model's reference pose, so you can move them freely in the XML.
- Mount height matters for the stereo "OBSTACLE" box. With a 15 deg downtilt the image
  centre ray hits flat floor at `3.86 x camera height`; below ~13 cm of height the floor
  itself reads under 0.5 m. The demo mounts the camera on the head, ~17 cm above ground.
- Other things in your scene that should be invisible to the sensors (e.g. a marker
  mesh) also go in group 3. Anything in groups 0-2, 4-5 is seen by camera, depth and LiDAR.
- The module writes `model.cam_fovy` (to 55), and raises `model.vis.global_.offwidth/
  offheight` if they are smaller than the sensor size. Both are render settings only.

## 2. Example loop

```python
import math, cv2, mujoco
from cubli_sensors import CubliSensors

model = mujoco.MjModel.from_xml_path("cubli.xml"); data = mujoco.MjData(model)
sensors = CubliSensors(model, data)               # body="cubli", "lidar_site", "cubli_cam"
while True:
    mujoco.mj_step(model, data)                   # YOUR physics; the sensors never step it
    vx, vy = data.qvel[0], data.qvel[1]           # free joint: world-frame linear velocity
    if math.hypot(vx, vy) > 0.02: sensors.heading = math.atan2(vy, vx)   # direction of travel
    cv2.imshow("sensors", sensors.compose()); k = cv2.waitKeyEx(1)       # Ex: arrow keys orbit
    if k == 27: break                             # Esc
    sensors.handle_key(k)                         # 0-6 views, G/L/P/F toggles, arrows orbit, -/= zoom
```

(Kinematic sims work too: set `qpos`, call `mj_forward`, then the sensors. See `demo_sensors.py`.)

## 3. API

| | |
|---|---|
| `rgb()` | HxWx3 uint8 RGB (lights off: near-black + noise) |
| `depth()` | HxW float32 metres, `0` = invalid (stereo noise, 0.2-8 m, ~98% invalid with lights off) |
| `lidar()` | Nx3 float32 points, **current sensor frame** (x fwd, y left, z up). Mid-360 model: non-repetitive pattern, 200k pts/s; each call fires `lidar_rays_per_call` rays (default 6000 ~ one 30 fps frame) and returns the newest `lidar_buffer` hits (default 60k), motion-compensated with the sim pose |
| `world()` | HxWx3 uint8 RGB orbiting 3rd-person camera, cube visible |
| `compose(view)` | BGR image with tab bar + footer; 1-6 single/split views, 0 = 2x2 grid (default `sensors.view`) |
| `on_mouse` | pass to `cv2.setMouseCallback(win, sensors.on_mouse)`: click tabs, left-drag orbit, right-drag or wheel zoom |
| `clear_lidar()` | drop the accumulated cloud (call after teleporting the cube) |
| `mast` | metres the sensor head is extended along sensor +z (pop-up mast); 0 = as placed in the XML |
| `follow` | 3rd person orbit follows `heading` (True) or stays world-fixed (False); key F |
| `cube_half` | half edge (0.075) used to find the ground plane under the cube for the depth ROI |
| constructor knobs | `size=(640,480)`, `shadows=True`, `lidar_rays_per_call`, `lidar_buffer`, `lidar_threads` (default min(4, cores-1)) |
| `gimbal`, `lights`, `heading` | state. `heading` = world yaw (rad) of travel; gimbal ON levels the head and yaws it to `heading` |
| `psyche` | property; setting it writes `model.opt.gravity[2]` (0.144 or 9.81) so *your* sim feels it too |
| `head_pose()` | `(origin, 3x3 R)` of the sensor head in the world (handy to draw a head marker) |
| `lidar_source` | optional callable returning Nx3; replaces the raycast (see below) |
| `timing` | dict of per-stage ms (EMA): `lidar rgb depth world compose` |

## 4. Depth "Ahead" box

The ROI reads the median stereo depth. OBSTACLE (red) needs >=5% of the ROI to be more than 4 cm
above the cube's ground plane (pixels lifted to world height with the camera pose), closer than
0.5 m. Flat floor shows as `Ahead: X m (floor)`. A camera mounted low and tilted down always sees
floor in the centre, so plain median depth would flag OBSTACLE permanently.

## 5. Cubli CAD model (demo)

`python tools/import_cubli_stl.py "<path>.STL"` splits the assembly STL into parts, groups them
(frame / wheel / motor / hardware), decimates (454k -> 48k triangles), rotates it so the
wheel-free faces are front (+x) and top (+z), and writes `assets/cubli_*.obj`. The demo uses those
when present, else a primitive cube. In your XML, reference them as mesh geoms in group 3.

## 6. Swapping in the real Livox Mid-360 later

Every LiDAR view goes through `lidar()`. Set `sensors.lidar_source = my_func` where
`my_func()` returns the latest Nx3 points in the Mid-360 frame (x fwd, y left, z up, metres).
Nothing else changes. Not built yet: the UDP 56301 packet parser (the sensor is at 192.168.1.115).
Note the real sensor's frame has its own mounting rotation; apply it inside `my_func`.

## 7. Platform notes

- **macOS:** the offscreen `mujoco.Renderer` and `cv2.imshow` work together on the main
  thread under plain `python` (tested: macOS, MuJoCo 3.13, OpenCV 5.0; the demo loop ran
  at ~40 fps cycling through all views).
  **`mjpython` is not compatible with `cv2.namedWindow`** (tested: raises
  `cv2.error: Unknown C++ exception`). You only need `mjpython` if your sim uses
  `mujoco.viewer.launch_passive`; in that case don't open the OpenCV window in that process
  (render `compose()` frames and show them from a separate plain-python process, or save them).
- **Jetson / Linux:** the module sets `MUJOCO_GL=egl` before importing mujoco if you haven't
  set it (headless-safe). Set it yourself first if you prefer `osmesa`/`glfw`. Deps are
  `mujoco opencv-python numpy`. **Not tested on a Jetson from here.** Run
  `python demo_sensors.py --bench` there to get real numbers.
- Speed knobs: `CubliSensors(..., size=(480, 360))`, `shadows=False`, fewer `lidar_rays_per_call`,
  and `demo_sensors.py --lite`. View 0 (grid) is the most expensive view.
- LiDAR raycasts run on `lidar_threads` threads (`mj_multiRay` releases the GIL; measured 15.8 ms
  -> 4.1 ms for 6000 rays with 4 threads on an 8-core Mac).
- Keep ray-cast scenes to a few large meshes: `mj_multiRay` tests every geom per ray, so hundreds of
  small rock geoms are slow. The demo bakes all rocks into one mesh per material.

## 8. Known limits

- MuJoCo prints `ARB_clip_control unavailable ... depth accuracy will be limited` on macOS. The
  depth precision loss only matters at long range; stereo noise at 8 m (~0.28 m sigma) dominates.
- Stereo is modelled from the rendered depth plus disparity noise and a depth-edge dropout, not by
  rendering two cameras. It does not model texture-dependent matching (only the lights-off case).
- The Mid-360 pattern is approximated with a low-discrepancy (R2) sequence over its FOV: same
  rate, FOV and fill-in-over-time behaviour, not the exact rosette. With -7..+52 deg most rays
  go to the sky on open ground (~17% return here), as on the real sensor; there is a blind ring of
  ~1.5 m radius around the cube.
- Sensors ignore the cube's own geometry (group 3), as a self-filter would; the real frame
  bars and wheels would occlude part of the view when the head is stowed inside.
- The demo terrain (`psyche_terrain.py`) is a Psyche *analog*: no surface images of Psyche exist
  yet. It is visual + ray-cast only (`contype=0`); a dynamic sim should collide with its own hfield.
