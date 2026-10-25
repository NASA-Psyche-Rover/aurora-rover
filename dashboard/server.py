#!/usr/bin/env python3
"""
AURORA live dashboard server.

One ROS 2 node that subscribes to the sensor topics, builds the three panel
streams, and serves them with Flask on http://localhost:8080:

  /stream/detection  YOLO boxes + metric distances + drop/slope edge
  /stream/depth      the metric distance heatmap (inset on the detection panel)
  /stream/cloud      3D render of the live Livox point cloud (orbit camera, moved by dragging)
  /api/metrics       JSON for the metrics strips
  /api/view          POST {dyaw, dpitch, zoom, reset} to steer the cloud camera

It never opens a sensor itself, so it works on live drivers or on a replayed
bag alike. Detection reuses perception/OAKD-obj-detec.py (Perception,
annotate) so the dashboard shows exactly what that script decides.

Topics (all overridable): /oak/rgb/image_raw, /oak/stereo/depth,
/oak/rgb/camera_info, /livox/lidar.
"""
import os

# Same BLAS-thread cap as OAKD-obj-detec.py, set before numpy loads (see the
# comment there): busy-waiting BLAS threads starve everything else on the Jetson.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import importlib.util
import sys
import threading
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERCEPTION = os.path.join(REPO, "perception")
sys.path.insert(0, PERCEPTION)         # perception's `src` package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(PERCEPTION)     # perception code resolves models/... relative to its own folder

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from flask import Flask, Response, jsonify, request, send_from_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from cloud_render import CloudAccumulator, CloudRenderer

JPEG_QUALITY = 80
STALE_S = 2.0            # a panel with no new data for this long shows NO SIGNAL
CLOUD_FPS = 30
VIDEO_FPS = 30           # camera feed rate; independent of how fast detection runs


def load_detection_module():
    """OAKD-obj-detec.py has a hyphen, so it can't be a normal import."""
    path = os.path.join(PERCEPTION, "OAKD-obj-detec.py")
    spec = importlib.util.spec_from_file_location("oakd_obj_detec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def no_signal_jpeg(text, size=(640, 480)):
    img = np.full((size[1], size[0], 3), (22, 18, 14), np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, text, ((size[0] - tw) // 2, (size[1] + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 140, 140), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


class Channel:
    """Latest JPEG for one stream plus a condition variable, so each MJPEG
    client wakes exactly when a new frame lands instead of polling."""

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.jpeg = None
        self.captured = 0.0       # when the sensor frame behind `jpeg` arrived (X-Capture-Time header; end-to-end lag checks)
        self.stamp = 0.0          # wall time of the last publish
        self.count = 0
        self._cv = threading.Condition()
        self._rate = None         # smoothed frames per second
        self.readers = 0          # open MJPEG connections
        self.last_poll = 0.0      # last time a page asked for a frame; producers pause when nobody is watching

    def publish(self, jpeg, captured=0.0):
        now = time.time()
        with self._cv:
            if self.stamp:
                inst = 1.0 / max(now - self.stamp, 1e-3)
                self._rate = inst if self._rate is None else 0.85 * self._rate + 0.15 * inst
            self.jpeg, self.stamp, self.captured = jpeg, now, captured
            self.count += 1
            self._cv.notify_all()

    def watched(self):
        return self.readers > 0 or time.time() - self.last_poll < 1.5

    def next_frame(self, after):
        """Newest JPEG once it is newer than frame number `after` (waits up to 1 s). The page asks
        for a frame only after it has shown the previous one, so a slow browser skips frames
        instead of queueing them (an MJPEG stream queues inside the browser and the lag grows)."""
        self.last_poll = time.time()
        with self._cv:
            self._cv.wait_for(lambda: self.count != after, timeout=1.0)
            jpeg, count = self.jpeg, self.count
        if jpeg is None or not self.online():
            time.sleep(0.25)
            return self.placeholder, count
        return jpeg, count

    def age_s(self):
        return time.time() - self.stamp if self.stamp else None

    def online(self):
        a = self.age_s()
        return a is not None and a < STALE_S

    def fps(self):
        return round(self._rate, 1) if self.online() and self._rate else 0.0

    def mjpeg(self):
        last = -1
        with self._cv:
            self.readers += 1
        try:
            while True:
                with self._cv:
                    self._cv.wait_for(lambda: self.count != last, timeout=1.0)
                    jpeg, last, cap = self.jpeg, self.count, self.captured
                if jpeg is None or not self.online():
                    jpeg = self.placeholder
                    time.sleep(0.25)
                yield b"--frame\r\nContent-Type: image/jpeg\r\nX-Capture-Time: " + b"%.3f" % cap + b"\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        finally:      # client left (or zoomed to another panel)
            with self._cv:
                self.readers -= 1


def encode(img, quality=JPEG_QUALITY, sharp_colour=False):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if sharp_colour:   # full-resolution chroma: 4:2:0 smears the 1-2 px coloured points of the cloud
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    return cv2.imencode(".jpg", img, params)[1].tobytes()


class Dashboard(Node):
    def __init__(self, args):
        super().__init__("aurora_dashboard")
        self.bridge = CvBridge()
        self.det = load_detection_module()
        self.channels = {
            "detection": Channel(no_signal_jpeg("NO CAMERA SIGNAL")),
            "depth": Channel(no_signal_jpeg("NO DEPTH", (320, 240))),
            "cloud": Channel(no_signal_jpeg("NO LIDAR SIGNAL", (960, 720))),
        }
        self.cloud_acc = CloudAccumulator(window_s=args.cloud_window, max_points=args.cloud_max_points,
                                         max_frames=args.cloud_frames)
        self.cloud_renderer = CloudRenderer()
        self.det_state = {}       # latest detection result summary for /api/metrics
        self.det_lock = threading.Lock()
        self.K = None
        self._rgb_msg = self._depth_msg = None   # raw messages; converted by the detection thread when it needs them
        self._rgb_n = 0
        self._rgb_t = 0.0         # arrival time of _rgb_msg
        self._det_stop = threading.Event()
        self._det_stamp = 0.0     # when detection last produced a result (the video stream pauses when unwatched)
        self._heatmap = None      # newest depth heatmap; encoded by its own thread at a lower rate
        self._last_det = None     # (terrain, rocks, decision) of the newest detection, drawn onto every video frame

        self.create_subscription(Image, args.image_topic, self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, args.depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, args.camera_info_topic, self.on_info, 1)
        self.create_subscription(PointCloud2, args.lidar_topic, self.on_cloud, qos_profile_sensor_data)

        threading.Thread(target=self.detection_loop, daemon=True).start()
        threading.Thread(target=self.cloud_loop, daemon=True).start()
        threading.Thread(target=self.video_loop, daemon=True).start()
        threading.Thread(target=self.heatmap_loop, daemon=True).start()
        self.get_logger().info("dashboard subscribed; waiting for sensor topics")

    # ---------------- ROS callbacks (cheap: store / encode only) ----------------
    def on_rgb(self, msg):
        self._rgb_msg = msg
        self._rgb_t = time.time()
        self._rgb_n += 1

    def on_depth(self, msg):
        self._depth_msg = msg

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def on_cloud(self, msg):
        if not self.channels["cloud"].watched():     # nobody is looking at the cloud: skip parsing it too
            return
        from aurora_lidar_overlay import pointcloud_utils
        xyz = pointcloud_utils.pointcloud2_to_arrays(msg)["xyz"]
        self.cloud_acc.add(xyz)

    # ---------------- detection thread ----------------
    def detection_loop(self):
        det = self.det
        detector = det.ObjectDetector(det.ROCK_MODEL_PATHS, conf=det.ROCK_CONF)
        note = "Detected rocks" if detector.is_rock_model else "Detected objects"
        perception = None
        last_n = -1
        t_prev = time.time()
        frame_s = 1 / 30
        while not self._det_stop.is_set():
            rgb_msg, depth_msg, K, n = self._rgb_msg, self._depth_msg, self.K, self._rgb_n
            if rgb_msg is None or depth_msg is None or K is None or n == last_n:
                time.sleep(0.01)
                continue
            last_n = n
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")   # uint16 mm
            if perception is None:
                perception = det.Perception(detector, K, (320, 240))
            try:
                t0 = time.time()
                r = perception((rgb, depth))
                step_s = time.time() - t0
            except Exception as e:     # keep the dashboard up; show the error in metrics
                self.get_logger().error(f"detection failed: {e!r}")
                with self.det_lock:
                    self.det_state = {"error": repr(e)}
                time.sleep(0.5)
                continue

            self._last_det = (r["terrain"], r["rocks"], r["decision"])
            self._heatmap = r["heatmap"]
            self._det_stamp = time.time()

            now = time.time()
            frame_s = 0.9 * frame_s + 0.1 * (now - t_prev)
            t_prev = now
            t = r["terrain"]
            objects = sorted(
                ({"label": lab, "dist_m": None if d is None else round(float(d), 2), "in_path": bool(inc)}
                 for _, lab, d, inc in r["rocks"]),
                key=lambda o: (o["dist_m"] is None, o["dist_m"]))
            with self.det_lock:
                self.det_state = {
                    "decision": r["decision"], "reason": r["reason"],
                    "hazard_m": None if r["hazard_m"] is None else round(float(r["hazard_m"]), 2),
                    "hazard_source": r["hazard_source"],
                    "terrain": {
                        "ground_found": bool(t.ground_found), "hazard": det.TERRAIN_NAMES.get(t.hazard_kind),
                        "hazard_m": None if t.hazard_kind is None else round(float(t.hazard_m), 2),
                        "camera_height_m": round(float(t.camera_height_m), 2) if t.ground_found else None,
                    },
                    "objects": objects[:6], "detector": note,
                    "fps": round(1.0 / frame_s, 1), "latency_ms": round(step_s * 1000),
                }

    # ---------------- heatmap thread ----------------
    def heatmap_loop(self):
        """The small depth inset does not need every detection frame; 10 Hz keeps JPEG work off the detector."""
        last = None
        while not self._det_stop.is_set():
            hm = self._heatmap
            if hm is not None and hm is not last and self.channels["depth"].watched():
                last = hm
                self.channels["depth"].publish(encode(hm, 75))
            time.sleep(0.1)

    # ---------------- video thread ----------------
    def video_loop(self):
        """Camera feed at the camera's own rate with the newest detection drawn on it, so the
        video stays smooth even though detection only updates ~10 times a second."""
        det, last_n, period = self.det, -1, 1.0 / VIDEO_FPS
        while not self._det_stop.is_set():
            t0 = time.time()
            msg, n, t_rgb, K, ld = self._rgb_msg, self._rgb_n, self._rgb_t, self.K, self._last_det
            if msg is None or n == last_n or K is None or not self.channels["detection"].watched():
                time.sleep(0.005 if self.channels["detection"].watched() else 0.1)
                continue
            last_n = n
            view = self.bridge.imgmsg_to_cv2(msg, "bgr8").copy()
            if ld is not None:
                terrain, rocks, decision = ld
                det.annotate(view, terrain, rocks, K)
                h, w = view.shape[:2]
                cv2.rectangle(view, (0, 0), (w - 1, h - 1), det.DECISION_COLORS[decision], 4)
            self.channels["detection"].publish(encode(view), t_rgb)
            time.sleep(max(period - (time.time() - t0), 0.0))

    # ---------------- cloud render thread ----------------
    def cloud_loop(self):
        period = 1.0 / CLOUD_FPS
        # JPEG encoding (~20 ms, releases the GIL) runs in its own thread so it overlaps the next
        # render; the slot holds only the newest frame, so a slow encode drops frames, never lags
        slot, ready = [None], threading.Event()

        def encode_loop():
            while not self._det_stop.is_set():
                if not ready.wait(0.5):
                    continue
                ready.clear()
                img, slot[0] = slot[0], None
                if img is not None:
                    self.channels["cloud"].publish(encode(img, 80, sharp_colour=True))

        threading.Thread(target=encode_loop, daemon=True).start()
        last = (-1, -1)
        while not self._det_stop.is_set():
            t0 = time.time()
            # redraw only when a new LiDAR frame arrived or the view moved; no age fade, so a frame
            # looks the same for its whole life and the cloud never pulses
            state = (self.cloud_acc.version, self.cloud_renderer.view_version)
            if state != last and self.channels["cloud"].watched() and self.cloud_acc.age_s() is not None and self.cloud_acc.age_s() < STALE_S:
                last = state
                pts = self.cloud_acc.points()
                slot[0] = self.cloud_renderer.render(pts, None, self.cloud_acc.window_s, bgra=True)
                ready.set()
            time.sleep(max(period - (time.time() - t0), 0.005))

    # ---------------- metrics ----------------
    def metrics(self):
        det = self.det
        ch = self.channels
        msgs_hz, pts_hz = self.cloud_acc.rates()
        with self.det_lock:
            d = dict(self.det_state)
        d.update(online=time.time() - self._det_stamp < STALE_S)
        return {
            "time": time.time(),
            "watched": {k: c.watched() for k, c in ch.items()},
            "detection": d,
            "cloud": {"online": ch["cloud"].online(), "render_fps": ch["cloud"].fps(),
                      "lidar_hz": round(msgs_hz, 1), "points_per_s": int(pts_hz),
                      "points_shown": self.cloud_acc.n_points(), "window_s": self.cloud_acc.window_s},
            "thresholds": {
                "object_stop_m": det.STOP_DISTANCE_M, "object_caution_m": det.CAUTION_DISTANCE_M,
                "drop_stop_m": det.TERRAIN_STOP_DISTANCE_M, "drop_caution_m": det.TERRAIN_CAUTION_DISTANCE_M,
                "corridor_half_width_m": det.mt.CORRIDOR_HALF_WIDTH_M,
            },
        }

    def shutdown(self):
        self._det_stop.set()


def make_app(node):
    app = Flask(__name__, static_folder=None)
    static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    @app.get("/")
    def index():
        return send_from_directory(static, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(static, name)

    @app.get("/stream/<name>")
    def stream(name):
        ch = node.channels.get(name)
        if ch is None:
            return ("unknown stream", 404)
        return Response(ch.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/frame/<name>")
    def frame(name):
        ch = node.channels.get(name)
        if ch is None:
            return ("unknown stream", 404)
        jpeg, count = ch.next_frame(request.args.get("after", -1, type=int))
        return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store", "X-Frame-Count": str(count), "X-Capture-Time": "%.3f" % ch.captured})

    @app.get("/api/metrics")
    def metrics():
        return jsonify(node.metrics())

    @app.post("/api/view")
    def view():
        body = request.get_json(silent=True) or {}
        r = node.cloud_renderer
        if body.get("reset"):
            r.reset()
        if body.get("dyaw") or body.get("dpitch"):
            r.rotate(float(body.get("dyaw", 0)), float(body.get("dpitch", 0)))
        if body.get("zoom"):
            r.zoom(float(body["zoom"]))
        return jsonify(ok=True)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let other machines view it")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--image-topic", default="/oak/rgb/image_raw")
    ap.add_argument("--depth-topic", default="/oak/stereo/depth")
    ap.add_argument("--camera-info-topic", default="/oak/rgb/camera_info")
    ap.add_argument("--lidar-topic", default="/livox/lidar")
    ap.add_argument("--cloud-frames", type=int, default=2, help="newest LiDAR frames drawn in the cloud panel (~0.1 s each); steadier than a time window")
    ap.add_argument("--cloud-window", type=float, default=0.5, help="oldest a drawn LiDAR frame may be, in seconds (safety cap if the sensor stalls)")
    ap.add_argument("--cloud-max-points", type=int, default=100_000, help="most points drawn in the cloud panel (render cost grows with it)")
    args, ros_args = ap.parse_known_args()

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    rclpy.init(args=ros_args)
    node = Dashboard(args)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    app = make_app(node)
    print(f"[dashboard] http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
