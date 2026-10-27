"""Adapters between the v2 sim and the FROZEN perception code. Nothing here edits, copies or wraps-by-patching
that code: it is imported as-is and given the names, frames and data formats it already expects.

  cubli_sensors.CubliSensors  (~/cubli_sensors: LiDAR, stereo depth, camera image processing)
      - LiDAR:  site "lidar_site" (x fwd, y left, z up = mast axis), on body "lidar"
      - camera: camera "cubli_cam" + site "cam_site", on body "cam_head" (turret on top of the LiDAR, after pan/tilt)
      - the cube's own geoms stay in group 3 (hidden from sensors), as the module expects
      - module gimbal = False: the poses come from the sim's own (fake) gimbal, mast, pan and tilt joints
      - "LiDAR off" uses the module's own `lidar_source` swap-in hook (returns no points)
      - LiDAR point rate follows SIM time at the Mid-360's 200,000 points/s (the module's fixed rays-per-call would
        tie it to the dashboard's frame rate instead)
      - "camera off" (head stowed, or switched off with C) = a black frame and no depth, like an unpowered OAK-D
      - the module's depth "Ahead" check finds the ground `cube_half` below its body (it assumes the sensor sits
        on the cube); the head is on a mast, so cube_half is set every frame to the camera's true height above
        the terrain under the cube (what the rover knows from its mast position)
  psyche-objdetec (MuJoCo Sim/psyche-objdetec: object detection + MiDaS depth risk + decision)
      - SimOakDLite has OakDLiteCamera's interface: read() -> (BGR uint8 frame, uint16 depth in mm), close()
      - ObjDetec runs Detector / TerrainAnalyzer / should_proceed / estimate_object_distances_mm unchanged
        (needs ultralytics + onnxruntime, which the system python3 has; the sim venv does not)
"""
import contextlib
import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))


def _first_dir(*cands, need=None):
    """First existing folder (that holds `need`, if given); else the last candidate."""
    for c in cands:
        if c and os.path.isdir(c) and (need is None or os.path.exists(os.path.join(c, need))):
            return c
    return cands[-1]


# Where the perception code lives: an env var wins, then the copy next to this file (aurora-rover branch cubli-sim:
# cubli_sim/cubli_sensors and ../perception), then the original clones in the home folder.
SENSORS_DIR = _first_dir(os.environ.get("CUBLI_SENSORS_DIR"), os.path.join(HERE, "cubli_sensors"),
                         os.path.join(HOME, "cubli_sensors"), need="cubli_sensors.py")
OBJDETEC_DIR = _first_dir(os.environ.get("CUBLI_OBJDETEC_DIR"),
                          os.path.join(HOME, "Desktop", "Aayou", "ASU", "EPICS", "NASA Psyche", "Stage 3", "MuJoCo Sim",
                                       "psyche-objdetec"))


def _import_cubli_sensors():
    if SENSORS_DIR not in sys.path:
        sys.path.append(SENSORS_DIR)
    import cubli_sensors
    return cubli_sensors


LIDAR_PTS_PER_S = 200_000      # DATASHEET: Livox Mid-360 point rate
LIDAR_MAX_PER_CALL = 30_000    # cap per dashboard frame (sim sped up / slow frame): ~30 ms of ray casting
LIDAR_WINDOW_S = 0.5           # points kept on screen = the last 0.5 s of scanning (100k points)


class SensorBridge:
    """The team's LiDAR + camera code, attached to the sensor head of the v2 cube."""

    def __init__(self, world, cam_size=(960, 720)):
        CS = _import_cubli_sensors()
        self.CS = CS
        self.w = world
        m, d = world.m, world.d
        self.lidar = CS.CubliSensors(m, d, body="lidar", lidar_site="lidar_site", body_cam="lidar_cam",
                                     size=cam_size, lidar_rays_per_call=0,
                                     lidar_buffer=int(LIDAR_PTS_PER_S * LIDAR_WINDOW_S))
        self.cam = CS.CubliSensors(m, d, body="cam_head", lidar_site="cam_site", body_cam="cubli_cam",
                                   size=cam_size)
        for s in (self.lidar, self.cam):
            s.gimbal = False
        self._off = lambda: np.zeros((0, 3), np.float32)
        self._t_scan = None
        self._epoch = None

    @property
    def cam_live(self):
        rig = self.w.rig
        return rig.cam_on and rig.deployed

    def sync_power(self):
        """LiDAR on/off from the rig switch, through the module's own swap-in hook. The module integrates points
        over time in the world frame; while the head is not fully out (stowed, moving, or switched off) the real
        sensor would publish only what it sees now, so the integration window is dropped (module's clear_lidar).
        Rays fired this call = Mid-360 rate x sim time since the last call."""
        rig = self.w.rig
        live = rig.lidar_on and rig.deployed
        self.lidar.lidar_source = None if live else self._off
        if not live or rig.lidar_epoch != self._epoch:     # off, or power-cycled since the last look
            self.lidar.clear_lidar()
            self._epoch = rig.lidar_epoch
        t = float(self.w.d.time)
        dt = 0.0 if self._t_scan is None or t < self._t_scan else t - self._t_scan
        self._t_scan = t
        self.lidar.lidar_rays_per_call = int(min(LIDAR_PTS_PER_S * dt, LIDAR_MAX_PER_CALL))

    def points(self):
        self.sync_power()
        return self.lidar.lidar()

    def lidar_view(self, view=4):
        """3 = 3D cloud, 4 = top-down (the module's own drawing)."""
        self.sync_power()
        return self.lidar.compose(view)

    def sync_cam(self):
        """Ground plane for the module's depth-ROI obstacle check: true camera height above the terrain."""
        w, c = self.w, self.cam
        Rb = w.d.xmat[c._bid].reshape(3, 3)
        com = w.com()
        above = float(w.d.xpos[c._bid][2] - w.height_at(com[0], com[1]))
        c.cube_half = above / max(np.abs(Rb[2]).sum(), 1e-6)

    def cam_view(self, view=1):
        """1 = RGB, 2 = stereo depth with the 'Ahead' ROI (the module's own drawing); black when the camera is off.
        Same layout as the module's compose(): tab bar, body, footer."""
        if not self.cam_live:
            w, CS = self.cam._w, self.CS
            body = off_image(w, self.cam._h, "CAMERA OFF", self.off_reason())
            return np.vstack([np.zeros((CS.TAB_H, w, 3), np.uint8), body, np.zeros((CS.FOOTER_H, w, 3), np.uint8)])
        self.sync_cam()
        return self.cam.compose(view)

    def off_reason(self):
        rig = self.w.rig
        if not rig.deployed:
            return "sensor head stowed: press S to deploy" if not rig.want_up else f"head {rig.state('cam').lower()}"
        return "switched off: press C"

    def close(self):
        self.lidar.close()
        self.cam.close()


def off_image(w, h, title, reason=""):
    """Black frame for a sensor with no power, in the module's text style."""
    import cv2
    img = np.zeros((h, w, 3), np.uint8)
    for text, y, sc, col in ((title, h // 2 - 8, 0.9, (200, 200, 200)), (reason, h // 2 + 26, 0.55, (140, 140, 140))):
        (tw, _), _b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
        cv2.putText(img, text, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
    return img


class SimOakDLite:
    """Drop-in for src.camera_stream.OakDLiteCamera: same read()/close(), fed by the simulated OAK-D Lite.
    size = the stream size the real pipeline asks the OAK-D for (OakDLiteCamera(rgb_size=(640, 480)) in the team
    code); the sim renders sharper and is resized down, depth with nearest-neighbour so no fake depths appear."""

    def __init__(self, bridge, size=None):
        self.b = bridge
        self.size = size

    def read(self):
        import cv2
        h, w = self.b.cam._h, self.b.cam._w
        if self.b.cam_live:
            self.b.sync_cam()
            rgb = cv2.cvtColor(self.b.cam.rgb(), cv2.COLOR_RGB2BGR)
            depth_mm = np.round(self.b.cam.depth() * 1000.0).astype(np.uint16)
        else:                                            # no power: no image, no depth
            rgb, depth_mm = np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.uint16)
        if self.size and tuple(self.size) != (w, h):
            rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
            depth_mm = cv2.resize(depth_mm, self.size, interpolation=cv2.INTER_NEAREST)
        return rgb, depth_mm

    def close(self):
        pass


@contextlib.contextmanager
def _in_objdetec():
    cwd = os.getcwd()
    os.chdir(OBJDETEC_DIR)                    # the repo uses repo-relative model paths
    sys.path.insert(0, OBJDETEC_DIR)
    try:
        yield
    finally:
        os.chdir(cwd)
        sys.path.remove(OBJDETEC_DIR)


AURORA_PERCEPTION = _first_dir(os.environ.get("CUBLI_PERCEPTION_DIR"),              # team repo (GitHub), read-only
                              os.path.join(HERE, "..", "perception"), os.path.join(HOME, "aurora-rover", "perception"),
                              need="OAKD-obj-detec.py")


class TeamPerception:
    """The team's CURRENT rover autonomy pipeline from GitHub (aurora-rover/perception/OAKD-obj-detec.py):
    Perception (metric stereo depth -> terrain/drops, YOLO -> boxes with distances, decision latch), annotate()
    and status_panel(), driven exactly like its main() but with the simulated OAK-D Lite instead of the USB
    camera. Runs in the team's own Worker thread so the display never waits on it."""

    def __init__(self, K, root=AURORA_PERCEPTION):
        import importlib.util
        if not os.path.exists(os.path.join(root, "OAKD-obj-detec.py")):
            raise FileNotFoundError(f"clone NASA-Psyche-Rover/aurora-rover to {os.path.dirname(root)}")
        cwd = os.getcwd()
        os.chdir(root)                         # model paths in the repo are relative to perception/
        sys.path.insert(0, root)
        try:
            spec = importlib.util.spec_from_file_location("aurora_oakd_obj_detec", os.path.join(root, "OAKD-obj-detec.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.detector = mod.ObjectDetector(mod.ROCK_MODEL_PATHS, conf=mod.ROCK_CONF)
        finally:
            os.chdir(cwd)
        self.mod, self.K = mod, np.asarray(K, float)
        self.note = "Detected rocks" if self.detector.is_rock_model else "Detected objects (COCO: no rock weights)"
        self.panel_h = mod.TARGET_HEIGHT // 2
        self.worker = mod.Worker(mod.Perception(self.detector, self.K, (mod.PANEL_W, self.panel_h)))
        self.worker.start()
        self.t_last, self.frame_s = None, 1 / 10

    def step(self, frame_bgr, depth_mm):
        """Submit the newest simulated 640x480 frame to the team's worker; return its latest result (None until the
        first pass is done). The worker always takes the newest frame, so the display never waits for YOLO."""
        import time
        self.worker.submit((frame_bgr, depth_mm))
        now = time.time()
        if self.t_last is not None:
            self.frame_s = 0.9 * self.frame_s + 0.1 * (now - self.t_last)
        self.t_last = now
        return self.worker.output

    def overlay(self, img, small, res):
        """The team's annotate() (rock boxes + distances, in-path obstacles, drop/slope edge) drawn on the 640x480
        frame the pipeline sees, laid over a sharper render of the same view, plus the decision as in their panel."""
        import cv2
        mod = self.mod
        ann = small.copy()
        mod.annotate(ann, res["terrain"], res["rocks"], self.K)
        h, w = img.shape[:2]
        drawn = (cv2.absdiff(ann, small).max(2) > 0).astype(np.uint8) * 255
        mask = cv2.resize(drawn, (w, h), interpolation=cv2.INTER_LINEAR) > 64
        out = img.copy()
        out[mask] = cv2.resize(ann, (w, h), interpolation=cv2.INTER_LINEAR)[mask]
        col = mod.DECISION_COLORS[res["decision"]]
        text = f"{res['decision']}   {res['reason']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (10, h - th - 26), (10 + tw + 20, h - 10), col, -1)
        mod.put(out, text, (20, h - 18), 0.7, (0, 0, 0), 2)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), col, 3)
        return out

    def display(self, frame_bgr, depth_mm):
        """Submit the newest simulated frame; return the team's display (camera+boxes | heatmap / status)."""
        import cv2
        mod = self.mod
        res = self.step(frame_bgr, depth_mm)
        if res is None:
            img = frame_bgr.copy()
            cv2.putText(img, "team perception: first pass running...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            return img, None
        view = frame_bgr.copy()
        mod.annotate(view, res["terrain"], res["rocks"], self.K)
        items = sorted(((label, d, inc) for _, label, d, inc in res["rocks"]), key=lambda it: (it[1] is None, it[1]))
        panel = mod.status_panel((mod.PANEL_W, mod.TARGET_HEIGHT - self.panel_h), res["decision"], res["reason"],
                                 res["hazard_m"], res["hazard_source"], res["terrain"], items,
                                 (1.0 / max(self.frame_s, 1e-3), self.worker.step_s), self.note)
        cv2.rectangle(view, (0, 0), (mod.TARGET_WIDTH - 1, mod.TARGET_HEIGHT - 1), mod.DECISION_COLORS[res["decision"]], 3)
        return np.hstack([view, np.vstack([res["heatmap"], panel])]), res["decision"]

    def close(self):
        self.worker.stop()


class ObjDetec:
    """The team's detection + terrain-risk + decision pipeline (main.py's per-frame logic), called unchanged."""

    def __init__(self):
        with _in_objdetec():
            from src.detect import Detector
            from src.terrain_risk import TerrainAnalyzer
            from src import decision, utils, camera_stream
            self.det = Detector("models/best.pt")
            self.ta = TerrainAnalyzer()
        self.decision, self.utils, self.cs = decision, utils, camera_stream

    def run(self, frame_bgr, depth_mm):
        import cv2
        with _in_objdetec():
            boxes, labels, _ = self.det.detect_objects(frame_bgr)
            depth_map, status, risk, _ = self.ta.get_risk_assessment(cv2.resize(frame_bgr, (320, 240)))
        fs = (frame_bgr.shape[1], frame_bgr.shape[0])
        density = self.utils.compute_cluster_density(boxes, fs[0] * fs[1])
        proximity = self.utils.estimate_object_proximity(boxes, depth_map, fs)
        return dict(boxes=boxes, labels=labels, terrain=status, risk=float(risk), density=float(density),
                    proximity=float(proximity),
                    decision=self.decision.should_proceed(risk, proximity, density),
                    distances_mm=self.cs.estimate_object_distances_mm(boxes, depth_mm, fs),
                    nearest_m=self.cs.nearest_object_distance_m(boxes, depth_mm, fs) if boxes else None)
