#!/usr/bin/env python3
"""Sensor dashboard for the live sim: a separate OpenCV window (the MuJoCo window runs under mjpython, where
OpenCV windows cannot open). It rebuilds the same world, mirrors the sim's state over a local socket, and draws
what the cube's sensors see with the team's own code:

    1 CAMERA      OAK-D Lite RGB (cubli_sensors) + the team's rock detection on top (aurora-rover: YOLO boxes with
                  stereo distances, in-path obstacles, drop/slope edge, decision). Y turns the detection on / off
    2 DEPTH       stereo depth + 'Ahead' ROI (cubli_sensors)
    3 LIDAR 3D    Mid-360 cloud, drag to orbit, wheel to zoom (cubli_sensors)
    4 LIDAR TOP   top-down cloud, wheel to zoom (cubli_sensors)
    5 3RD PERSON  chase view of the cube, drag to orbit
    6 SPLIT       3rd person | LiDAR 3D
    7 DETECT      the team's rover autonomy display (aurora-rover/perception: YOLO + metric depth + decision),
                  or the local psyche-objdetec pipeline if the GitHub clone is missing / fails
    0 GRID        camera + detection, depth, LiDAR 3D, LiDAR top

Every other key (arrows, Enter, Space, S, O, C, J/;, I/K, G, P, -/=, R, V ...) is sent to the sim, so you can drive
from either window. ? (or the "? KEYS" button) lists every key. Q closes the dashboard. Started automatically by
psyche_cubli_v2.py.
"""
import argparse
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import psyche_cubli_v2 as V  # noqa: E402
from sensor_bridge import ObjDetec, SensorBridge, SimOakDLite, TeamPerception, _import_cubli_sensors  # noqa: E402

W, H = 1280, 720                      # canvas (body); tab bar + footer added
TAB_H, FOOT_H = 34, 30
TABS = [(1, "1 CAMERA"), (2, "2 DEPTH"), (3, "3 LIDAR 3D"), (4, "4 LIDAR TOP"), (5, "5 3RD PERSON"),
        (6, "6 SPLIT"), (7, "7 DETECT"), (0, "0 GRID")]
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEYS_BTN = "? KEYS"
KEY_HELP = [          # (heading, [(keys, what)]) -- the overlay behind the "? KEYS" button
    ("THIS WINDOW", [("1-7, 0", "switch view (or click a tab)"), ("Y", "rock detection on the camera on / off"),
                     ("drag", "orbit: LiDAR 3D, 3rd person, split, grid"),
                     ("wheel", "zoom: LiDAR 3D / top, 3rd person"), ("? or /", "show / hide this list"),
                     ("Q / Esc", "close the dashboard (the sim keeps running)")]),
    ("SENSORS", [("S", "deploy / stow the sensor head (both switch on when out)"), ("O", "LiDAR on / off"),
                 ("C", "camera on / off"), ("J / ;", "camera pan left / right"), ("I / K", "camera tilt up / down")]),
    ("DRIVE", [("arrows", "roll one face (relative to the sim window's camera)"), ("Space", "hop"),
               ("Enter", "balance on an edge / let go"), ("T", "go to where the sim window looks")]),
    ("SIM", [("G", "next gravity (sites, Moon, Earth)"), ("P", "pause"), ("- / =", "slower / faster"),
             ("R", "reset"), ("V", "record a take"), ("B", "Blender live view"), ("F / H", "sim camera follow / status text")]),
]
# cv2.waitKeyEx arrow codes (macOS Cocoa / Linux GTK) -> GLFW key codes the sim understands
ARROWS = {63232: 265, 63233: 264, 63234: 263, 63235: 262, 65362: 265, 65364: 264, 65361: 263, 65363: 262}


def fit(img, w, h):
    """Letterbox img into w x h."""
    s = min(w / img.shape[1], h / img.shape[0])
    im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 12, np.uint8)
    y, x = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
    out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


class Link(threading.Thread):
    """Receives the sim state; sends key presses back."""

    def __init__(self, port, key):
        super().__init__(daemon=True)
        from multiprocessing.connection import Client
        self.conn = Client(("127.0.0.1", port), authkey=bytes.fromhex(key))
        self.state, self.alive, self.lock = None, True, threading.Lock()

    def run(self):
        try:
            while True:
                st = self.conn.recv()
                with self.lock:
                    self.state = st
        except (EOFError, OSError):
            self.alive = False

    def send_key(self, code):
        try:
            with self.lock:
                self.conn.send(("key", int(code)))
        except OSError:
            self.alive = False


class Dashboard:
    def __init__(self, site, link):
        self.link = link
        self.w = V.build_world(site)
        self.CS = _import_cubli_sensors()
        self.br = SensorBridge(self.w)                         # 960 x 720: shown 1:1 in the single views
        self.oak = SimOakDLite(self.br, size=(640, 480))       # the team pipeline's OAK-D stream size
        self.r = mujoco.Renderer(self.w.m, H, W)
        self.help = False
        self.det_on = True                                     # team detection drawn on the camera view (Y)
        self.chase = dict(yaw=-35.0, pitch=18.0, dist=0.9)
        self.view, self.drag, self.panes, self.tabs = 1, None, [], []
        self.det, self.det_kind, self.det_err = None, "", ""
        self.reset_count = None
        cam = self.w.m.camera("cubli_cam").id
        f = (480 / 2) / math.tan(math.radians(self.w.m.cam_fovy[cam]) / 2)
        self.K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]])
        self.fps = 0.0

    # ---- state mirror ---------------------------------------------------
    def apply(self, st):
        w = self.w
        w.d.qpos[:] = st["qpos"]
        w.d.time = st["time"]
        w.m.opt.gravity[:] = st["gravity"]
        rig = w.rig
        rig.lidar_on, rig.cam_on, rig.want_up = st["lidar_on"], st["cam_on"], st["want_up"]
        rig.lidar_epoch = st["lidar_epoch"]
        rig.target.update(st["targets"])
        if st["resets"] != self.reset_count:            # sim reset: old LiDAR points are invalid
            self.reset_count = st["resets"]
            self.br.lidar.clear_lidar()
        mujoco.mj_forward(w.m, w.d)

    # ---- views ----------------------------------------------------------
    def body(self, inst, view):
        """A view through the bridge (sensor power, true camera height), without the module's own bars."""
        br = self.br
        img = br.cam_view(view) if inst is br.cam else br.lidar_view(view)
        img = img[self.CS.TAB_H:img.shape[0] - self.CS.FOOTER_H]
        if inst is br.lidar and not (self.w.rig.lidar_on and self.w.rig.deployed):
            reason = br.off_reason().replace("press C", "press O")
            for text, dy, sc in (("LIDAR OFF", -8, 0.9), (reason, 26, 0.55)):
                (tw, _), _b = cv2.getTextSize(text, FONT, sc, 1)
                cv2.putText(img, text, ((img.shape[1] - tw) // 2, img.shape[0] // 2 + dy), FONT, sc,
                            (200, 200, 200) if dy < 0 else (140, 140, 140), 1, cv2.LINE_AA)
        return img

    def third_person(self):
        c = mujoco.MjvCamera()
        com = self.w.com()
        c.lookat[:] = com + np.array([0, 0, 0.04])
        c.distance, c.azimuth, c.elevation = self.chase["dist"], 135.0 + self.chase["yaw"], -self.chase["pitch"]
        self.r.update_scene(self.w.d, c)
        img = cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, "3RD PERSON  drag: orbit  wheel: zoom", (8, 26), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def ensure_det(self):
        """The detection pipeline, started on first use: the team's GitHub one, else the local psyche-objdetec."""
        if self.det is None and not self.det_err:
            try:
                if os.environ.get("CUBLI_NO_TEAM"):
                    raise RuntimeError("disabled by CUBLI_NO_TEAM")
                self.det, self.det_kind = TeamPerception(self.K), "team (aurora-rover/perception)"
            except Exception as e:                  # noqa: BLE001
                try:
                    self.det, self.det_kind = ObjDetec(), f"local psyche-objdetec (GitHub pipeline unavailable: {e})"
                except Exception as e2:            # noqa: BLE001
                    self.det_err = f"no detection pipeline: {e2}"
        return self.det

    def camera_view(self, rgb=None, z=None):
        """Camera at full render size with the team's detection laid over it (the frame it gets is the real
        640x480 OAK-D stream). rgb / z: already rendered this frame (the grid shares them with the depth tile)."""
        br = self.br
        if not br.cam_live:
            return self.body(br.cam, 1)
        br.sync_cam()
        img = cv2.cvtColor(br.cam.rgb() if rgb is None else rgb, cv2.COLOR_RGB2BGR)
        note = ""
        if self.det_on:
            det = self.ensure_det()
            if isinstance(det, TeamPerception):
                z = br.cam.depth() if z is None else z
                small = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                dmm = cv2.resize(np.round(z * 1000.0).astype(np.uint16), (640, 480), interpolation=cv2.INTER_NEAREST)
                res = det.step(small, dmm)
                note = "   + team rock detection (Y: off)" if res else "   team rock detection: first pass..."
                if res:
                    img = det.overlay(img, small, res)
            else:
                note = "   (rock detection unavailable: see 7 DETECT)"
        br.cam._title(img, "CAMERA  OAK-D Lite RGB  55 deg V-FOV" + note)
        return img

    def detect(self):
        self.ensure_det()
        frame, depth_mm = self.oak.read()
        if self.det_err:
            img = frame.copy()
            cv2.putText(img, self.det_err[:90], (10, 30), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            return img
        try:
            if isinstance(self.det, TeamPerception):
                img, _ = self.det.display(frame, depth_mm)
            else:
                res = self.det.run(frame, depth_mm)
                img = frame.copy()
                for (x1, y1, x2, y2), lab in zip(res["boxes"], res["labels"]):
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, lab, (x1, max(y1 - 6, 12)), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                col = {"STOP": (0, 0, 255), "CAUTION": (0, 200, 255)}.get(res["decision"], (0, 255, 0))
                near = f"{res['nearest_m']:.2f} m" if res["nearest_m"] else "-"
                for i, t in enumerate([f"Decision: {res['decision']}", f"Terrain: {res['terrain']} ({res['risk']:.2f})",
                                       f"Nearest object (stereo): {near}"]):
                    cv2.putText(img, t, (12, 30 + 26 * i), FONT, 0.65, col if i == 0 else (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:                      # noqa: BLE001
            self.det_err = f"detection failed: {e}"
            img = frame.copy()
        cv2.putText(img, f"7 DETECT: {self.det_kind}"[:110], (8, img.shape[0] - 10), FONT, 0.42, (200, 200, 200), 1,
                    cv2.LINE_AA)
        return img

    def render(self):
        v = self.view
        self.panes = []
        if v == 1:
            img = fit(self.camera_view(), W, H)
        elif v == 2:
            img = fit(self.body(self.br.cam, 2), W, H)
        elif v == 3:
            img = fit(self.body(self.br.lidar, 3), W, H)
            self.panes = [(0, 0, W, H, "cloud")]
        elif v == 4:
            img = fit(self.body(self.br.lidar, 4), W, H)
            self.panes = [(0, 0, W, H, "bev")]
        elif v == 5:
            img = fit(self.third_person(), W, H)
            self.panes = [(0, 0, W, H, "chase")]
        elif v == 6:
            img = np.hstack([fit(self.third_person(), W // 2, H), fit(self.body(self.br.lidar, 3), W // 2, H)])
            self.panes = [(0, 0, W // 2, H, "chase"), (W // 2, 0, W, H, "cloud")]
        elif v == 7:
            img = fit(self.detect(), W, H)
        else:
            br = self.br
            if br.cam_live:                           # one render of the camera + depth serves both tiles
                br.sync_cam()
                rgb, z = br.cam.rgb(), br.cam.depth()
                cam_t, dep_t = self.camera_view(rgb, z), br.cam._depth_view(z)
            else:
                cam_t, dep_t = self.body(br.cam, 1), self.body(br.cam, 2)
            tiles = [cam_t, dep_t, self.body(br.lidar, 3), self.body(br.lidar, 4)]
            tiles = [fit(t, W // 2, H // 2) for t in tiles]
            img = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            self.panes = [(0, H // 2, W // 2, H, "cloud")]
        self.panes = [(x0, y0 + TAB_H, x1, y1 + TAB_H, k) for x0, y0, x1, y1, k in self.panes]
        if self.help:
            img = self.key_help(img)
        return np.vstack([self.tabbar(), img, self.footer()])

    def key_help(self, img):
        """Every key, over a dimmed view, in the dashboard's own colours."""
        img = (img * 0.25).astype(np.uint8)
        cols = [KEY_HELP[:2], KEY_HELP[2:]]
        for c, groups in enumerate(cols):
            x, y = 70 + c * (W // 2), 70
            for head, rows in groups:
                cv2.putText(img, head, (x, y), FONT, 0.6, (200, 120, 20), 2, cv2.LINE_AA)
                y += 32
                for keys, what in rows:
                    cv2.putText(img, keys, (x, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, what, (x + 110, y), FONT, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
                    y += 28
                y += 26
        cv2.putText(img, "Keys typed here go to the sim too. Press ? again to close.", (70, H - 30), FONT, 0.5,
                    (150, 150, 150), 1, cv2.LINE_AA)
        return img

    def tabbar(self):
        bar = np.full((TAB_H, W, 3), 22, np.uint8)
        self.tabs, x = [], 4
        for v, name in TABS:
            (tw, _), _b = cv2.getTextSize(name, FONT, 0.55, 1)
            x1 = x + tw + 22
            on = v == self.view
            cv2.rectangle(bar, (x, 4), (x1, TAB_H - 1), (200, 120, 20) if on else (55, 55, 55), -1)
            cv2.putText(bar, name, (x + 11, 23), FONT, 0.55, (255, 255, 255) if on else (190, 190, 190), 1, cv2.LINE_AA)
            self.tabs.append((x, x1, v))
            x = x1 + 4
        (tw, _), _b = cv2.getTextSize(KEYS_BTN, FONT, 0.55, 1)
        x0 = W - tw - 26
        cv2.rectangle(bar, (x0, 4), (W - 4, TAB_H - 1), (200, 120, 20) if self.help else (55, 55, 55), -1)
        cv2.putText(bar, KEYS_BTN, (x0 + 11, 23), FONT, 0.55, (255, 255, 255) if self.help else (190, 190, 190), 1,
                    cv2.LINE_AA)
        self.tabs.append((x0, W - 4, "help"))
        return bar

    def footer(self):
        st = self.link.state or {}
        bar = np.full((FOOT_H, W, 3), 28, np.uint8)
        txt = (f"{st.get('gravity_name', '?')}  |g| {np.linalg.norm(st.get('gravity', [0, 0, 0])):.4f} m/s^2   "
               f"sim x{st.get('speed', 0):g}{'  PAUSED' if st.get('paused') else ''}   LiDAR {st.get('lidar_state', '?')}"
               f"   camera {st.get('cam_state', '?')}{'   [REC]' if st.get('recording') else ''}   "
               f"dashboard {self.fps:4.1f} fps   > {st.get('status', '')}")
        cv2.putText(bar, txt[:170], (8, 20), FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        (tw, _), _b = cv2.getTextSize("SIMULATED", FONT, 0.55, 2)
        cv2.putText(bar, "SIMULATED", (W - tw - 8, 21), FONT, 0.55, (0, 170, 255), 2, cv2.LINE_AA)
        return bar

    # ---- input ------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < TAB_H:
                for x0, x1, v in self.tabs:
                    if x0 <= x < x1:
                        if v == "help":
                            self.help = not self.help
                        else:
                            self.view, self.help = v, False
                return
            for x0, y0, x1, y1, k in self.panes:
                if x0 <= x < x1 and y0 <= y < y1:
                    self.drag = (k, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:   # macOS doesn't set the button flag on moves
            k, px, py = self.drag
            self.orbit(k, x - px, y - py)
            self.drag = (k, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            f = 0.9 if cv2.getMouseWheelDelta(flags) > 0 else 1 / 0.9
            for x0, y0, x1, y1, k in self.panes:
                if x0 <= x < x1 and y0 <= y < y1:
                    self.zoom(k, f)

    def orbit(self, k, dx, dy):
        if k == "cloud":
            self.br.lidar.cloud_orbit.drag(dx, dy)
        elif k == "chase":
            self.chase["yaw"] -= dx * 0.35
            self.chase["pitch"] = float(np.clip(self.chase["pitch"] + dy * 0.35, -5, 89))

    def zoom(self, k, f):
        if k == "cloud":
            self.br.lidar.cloud_orbit.zoom(f)
        elif k == "bev":
            self.br.lidar.bev_range = float(np.clip(self.br.lidar.bev_range * f, 2, 40))
        elif k == "chase":
            self.chase["dist"] = float(np.clip(self.chase["dist"] * f, 0.3, 12))

    def key(self, k):
        if k < 0:
            return True
        if k in ARROWS:
            self.link.send_key(ARROWS[k])
            return True
        ch = chr(k & 0xFF) if k < 256 else ""
        if ch and ch in "01234567":
            self.view, self.help = int(ch), False
        elif ch in ("?", "/"):
            self.help = not self.help
        elif ch in ("y", "Y"):
            self.det_on = not self.det_on
        elif ch in ("q", "Q", "\x1b"):
            return False
        elif ch in ("\r", "\n"):
            self.link.send_key(257)
        elif ch:
            self.link.send_key(ord(ch.upper()))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--site", default=V.DEFAULT_SITE)
    ap.add_argument("--snapshot", default=None, help="test: save each tab to this prefix and exit")
    a = ap.parse_args()
    link = Link(a.port, os.environ["CUBLI_DASH_KEY"])
    link.start()
    dash = Dashboard(a.site, link)
    t0 = time.time()
    while link.state is None and time.time() - t0 < 30:
        time.sleep(0.05)
    if a.snapshot:                                    # headless test: one image per tab
        for v, _ in TABS:
            dash.view = v
            for _ in range(12 if v in (3, 4, 6, 0) else 2):
                dash.apply(link.state)
                img = dash.render()
            if v == 7:
                t1 = time.time()
                while time.time() - t1 < 20:
                    dash.apply(link.state)
                    img = dash.render()
                    if dash.det_err or (dash.det is not None and not isinstance(dash.det, TeamPerception)) or \
                            (isinstance(dash.det, TeamPerception) and dash.det.worker.output is not None):
                        img = dash.render()
                        break
                    time.sleep(0.1)
            cv2.imwrite(f"{a.snapshot}_{v}.png", img)
            print("wrote", f"{a.snapshot}_{v}.png", flush=True)
        return
    win = "Cubli sensor dashboard (SIMULATED)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, dash.on_mouse)
    last = time.time()
    while link.alive:
        if link.state is not None:
            with link.lock:
                st = link.state
            dash.apply(st)
        cv2.imshow(win, dash.render())
        now = time.time()
        dash.fps = 0.9 * dash.fps + 0.1 / max(now - last, 1e-3)
        last = now
        if not dash.key(cv2.waitKeyEx(1)):
            break
    cv2.destroyAllWindows()
    if dash.det is not None and hasattr(dash.det, "close"):
        dash.det.close()


if __name__ == "__main__":
    main()
