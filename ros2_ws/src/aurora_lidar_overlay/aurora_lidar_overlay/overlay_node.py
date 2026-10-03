#!/usr/bin/env python3

# Copyright 2026 AURORA Rover Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ROS 2 node that draws the live LiDAR point cloud onto the camera image.

Renders it Waymo-style and republishes the result as a plain
sensor_msgs/Image. Works identically live or on `ros2 bag play` -- it just
subscribes to topics, and a bag publishes those same topics during replay.
See the package README for exact commands.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
from std_srvs.srv import Trigger

from aurora_lidar_overlay import calibration_io, overlay_math, pointcloud_utils, tf_utils

try:
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import ExtrapolationException, LookupException, ConnectivityException

    _TF2_AVAILABLE = True
except ImportError:  # pragma: no cover - tf2_ros always present in a real ROS install
    _TF2_AVAILABLE = False


def _default_config_path(*parts) -> str:
    try:
        share = get_package_share_directory("aurora_lidar_overlay")
    except Exception:
        share = os.path.join(os.path.dirname(__file__), "..")
    return os.path.join(share, "config", *parts)


_TUNE_KEYS = ("tune_roll_deg", "tune_pitch_deg", "tune_yaw_deg", "tune_x_cm", "tune_y_cm", "tune_z_cm")


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class OverlayNode(Node):
    def __init__(self):
        super().__init__("aurora_lidar_overlay_node")
        self._declare_parameters()
        self._read_parameters()

        self.bridge = CvBridge()

        # Static fallback extrinsic (used whenever use_tf is False, or when
        # a TF lookup fails and we fall back rather than drop the frame).
        self.T_cam_lidar_static = calibration_io.load_extrinsic_matrix(
            self.extrinsic_yaml, invert=self.invert_extrinsic
        )
        self.get_logger().info(f"Loaded static extrinsic from {self.extrinsic_yaml}:\n{self.T_cam_lidar_static}")

        # Fallback intrinsics (used until/unless a CameraInfo message
        # arrives on camera_info_topic -- e.g. during early startup, or
        # forever if the driver never publishes one).
        try:
            self.K, self.D, _, _ = calibration_io.load_intrinsics(self.intrinsic_yaml)
            self.get_logger().info(f"Loaded fallback intrinsics from {self.intrinsic_yaml}")
        except FileNotFoundError:
            self.K = None
            self.D = None
            self.get_logger().warn(
                f"No fallback intrinsics at {self.intrinsic_yaml} -- waiting for "
                f"a CameraInfo message on {self.camera_info_topic} before publishing anything."
            )
        self._have_camera_info = False

        self.tf_buffer = None
        self.tf_listener = None
        if self.use_tf:
            if not _TF2_AVAILABLE:
                self.get_logger().error(
                    "use_tf=True but tf2_ros is not importable; falling back to static YAML extrinsic."
                )
                self.use_tf = False
            else:
                self.tf_buffer = Buffer()
                self.tf_listener = TransformListener(self.tf_buffer, self)

        self._tune = {k: 0.0 for k in _TUNE_KEYS}
        self._accel = None  # low-passed Livox IMU accel (g); +up at rest
        self._last_aim_log_t = 0.0
        self.add_on_set_parameters_callback(self._on_params)
        self._apply_tune_params()
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.on_imu, qos_profile_sensor_data)
        self.save_srv = self.create_service(Trigger, "~/save_extrinsic", self.on_save_extrinsic)

        self._warned_no_deskew = False
        self._frame_count = 0
        self._last_fps_t = time.time()
        self._last_fps_count = 0
        self._last_offset_log_t = 0.0

        self.image_sub = Subscriber(self, Image, self.image_topic, qos_profile=qos_profile_sensor_data)
        self.lidar_sub = Subscriber(self, PointCloud2, self.lidar_topic, qos_profile=qos_profile_sensor_data)
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.ts.registerCallback(self.on_synced)

        self.caminfo_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.on_camera_info, qos_profile_sensor_data
        )

        self.pub = self.create_publisher(Image, self.output_topic, 1)

        self.get_logger().info(
            f"aurora_lidar_overlay ready: image={self.image_topic} lidar={self.lidar_topic} "
            f"camera_info={self.camera_info_topic} -> output={self.output_topic}"
        )

    # ------------------------------------------------------------------ params

    def _declare_parameters(self):
        p = self.declare_parameter
        p("image_topic", "/oak/rgb/image_raw")
        p("camera_info_topic", "/oak/rgb/camera_info")
        p("lidar_topic", "/livox/lidar")
        p("output_topic", "/aurora/lidar_overlay/image")

        p("extrinsic_yaml", _default_config_path("calibration", "extrinsics.yaml"))
        p("intrinsic_yaml", _default_config_path("calibration", "intrinsics.yaml"))
        p("invert_extrinsic", False)

        p("use_tf", False)
        p("lidar_frame", "livox_frame")
        p("camera_frame", "oak_camera")
        p("tf_timeout_sec", 0.1)

        p("sync_slop", 0.1)
        p("sync_queue_size", 10)

        p("color_by", "depth")  # depth | range | intensity | height
        p("max_range_m", 25.0)
        p("min_range_m", 0.3)
        p("point_size_near_px", 6)
        p("point_size_far_px", 1)
        p("skip", 1)
        p("max_points", 200000)

        p("deskew_enabled", True)
        p("deskew_reference_frame", "odom")

        p("log_time_offset_period_sec", 5.0)

        # Live aim-and-tune: a correction applied in the CAMERA frame on top of
        # the extrinsic YAML (x right, y down, z forward). Effect on the overlay:
        # +pitch moves it UP in the image (verified live), +yaw moves it RIGHT,
        # +roll rotates it clockwise. x/y/z_cm shift the LiDAR origin in camera axes.
        for k in _TUNE_KEYS:
            p(k, 0.0)
        p("imu_topic", "/livox/imu")
        p("show_front_marker", True)
        p("aim_log_period_sec", 2.0)

    def _read_parameters(self):
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.image_topic = g("image_topic")
        self.camera_info_topic = g("camera_info_topic")
        self.lidar_topic = g("lidar_topic")
        self.output_topic = g("output_topic")

        self.extrinsic_yaml = g("extrinsic_yaml")
        self.intrinsic_yaml = g("intrinsic_yaml")
        self.invert_extrinsic = g("invert_extrinsic")

        self.use_tf = g("use_tf")
        self.lidar_frame = g("lidar_frame")
        self.camera_frame = g("camera_frame")
        self.tf_timeout_sec = g("tf_timeout_sec")

        self.sync_slop = g("sync_slop")
        self.sync_queue_size = g("sync_queue_size")

        self.color_by = g("color_by")
        self.max_range_m = g("max_range_m")
        self.min_range_m = g("min_range_m")
        self.point_size_near_px = g("point_size_near_px")
        self.point_size_far_px = g("point_size_far_px")
        self.skip = max(1, int(g("skip")))
        self.max_points = int(g("max_points"))

        self.deskew_enabled = g("deskew_enabled")
        self.deskew_reference_frame = g("deskew_reference_frame")

        self.log_time_offset_period_sec = g("log_time_offset_period_sec")
        self.imu_topic = g("imu_topic")
        self.show_front_marker = g("show_front_marker")
        self.aim_log_period_sec = g("aim_log_period_sec")

    # -------------------------------------------------------------- callbacks

    def _on_params(self, params):
        for prm in params:
            if prm.name in _TUNE_KEYS and not isinstance(prm.value, (int, float)):
                return SetParametersResult(successful=False, reason=f"{prm.name} must be a number")
        for prm in params:
            if prm.name in _TUNE_KEYS:
                self._tune[prm.name] = float(prm.value)
            elif prm.name == "show_front_marker":
                self.show_front_marker = bool(prm.value)
        return SetParametersResult(successful=True)

    def _apply_tune_params(self):
        for k in _TUNE_KEYS:
            self._tune[k] = float(self.get_parameter(k).value)

    def _tuned_extrinsic(self) -> np.ndarray:
        """T_cam_lidar_static with the live camera-frame correction applied."""
        t = self._tune
        D = np.eye(4)
        D[:3, :3] = _rot("z", t["tune_roll_deg"]) @ _rot("x", t["tune_pitch_deg"]) @ _rot("y", t["tune_yaw_deg"])
        D[:3, 3] = np.array([t["tune_x_cm"], t["tune_y_cm"], t["tune_z_cm"]]) / 100.0
        return D @ self.T_cam_lidar_static

    def on_imu(self, msg: Imu):
        a = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self._accel = a if self._accel is None else 0.95 * self._accel + 0.05 * a

    def _front_vs_gravity(self):
        """(elevation_deg of LiDAR +x above horizon, roll_deg, text) from the IMU, or None."""
        if self._accel is None or np.linalg.norm(self._accel) < 0.3:
            return None
        a = self._accel / np.linalg.norm(self._accel)  # unit "up" in the LiDAR frame
        elev = float(np.degrees(np.arcsin(np.clip(a[0], -1, 1))))
        roll = float(np.degrees(np.arctan2(a[1], a[2])))
        word = "level" if abs(elev) < 5 else ("pointing UP" if elev > 0 else "pointing DOWN")
        return elev, roll, f"LiDAR front {elev:+.1f} deg above horizon ({word}), roll {roll:+.1f} deg"

    def on_save_extrinsic(self, request, response):
        T = self._tuned_extrinsic()
        try:
            calibration_io.save_extrinsic_matrix(
                self.extrinsic_yaml, T, {"source": "live_tuned_overlay_node", "tuning_applied": dict(self._tune)}
            )
        except Exception as e:  # noqa: BLE001
            response.success, response.message = False, f"save failed: {e}"
            return response
        # The saved matrix already contains the tuning: make it the new base and zero the deltas.
        self.T_cam_lidar_static = calibration_io.load_extrinsic_matrix(self.extrinsic_yaml)
        self.set_parameters([rclpy.parameter.Parameter(k, value=0.0) for k in _TUNE_KEYS])
        response.success, response.message = True, f"saved tuned extrinsic to {self.extrinsic_yaml}"
        self.get_logger().info(response.message)
        return response

    def _aim_report(self, xyz, T, w, h, now_sec):
        if now_sec - self._last_aim_log_t < self.aim_log_period_sec or xyz.shape[0] == 0:
            return
        self._last_aim_log_t = now_sec
        pix, front, cam = overlay_math.project_points(xyz, T, self.K, self.D)
        rng_ok = (cam[:, 2] >= self.min_range_m) & (cam[:, 2] <= self.max_range_m)
        inb = overlay_math.in_bounds_mask(pix, w, h)
        n_img = int(np.sum(front & rng_ok & inb))
        n_front = int(np.sum(front & rng_ok))
        hint = ""
        if n_img < 0.02 * max(1, n_front):
            off = front & rng_ok & ~inb
            if np.any(off):
                m = pix[off].mean(axis=0)
                dirs = []
                if m[0] < 0.2 * w:
                    dirs.append("LEFT")
                elif m[0] > 0.8 * w:
                    dirs.append("RIGHT")
                if m[1] < 0.2 * h:
                    dirs.append("UP")
                elif m[1] > 0.8 * h:
                    dirs.append("DOWN")
                if dirs:
                    hint = f" | most LiDAR points fall off-image to the {'/'.join(dirs)}: rotate camera that way"
            elif n_front == 0:
                hint = " | no LiDAR points in front of the camera: camera is facing away from the LiDAR front"
        g = self._front_vs_gravity()
        self.get_logger().info(
            f"aim: {n_img} pts in image ({n_front} in front of camera, {xyz.shape[0]} total)"
            f" | {g[2] if g else 'IMU: no data'}{hint}"
        )

    def on_camera_info(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64).reshape(1, -1) if len(msg.d) else np.zeros((1, 5))
        if not self._have_camera_info:
            self.get_logger().info(f"Received first CameraInfo on {self.camera_info_topic}; using it for K/D.")
        self.K, self.D = K, D
        self._have_camera_info = True

    def _resolve_extrinsic(self, stamp):
        """
        Return the T_cam_lidar to use for this frame.

        Also returns whether it came from a live TF lookup (vs. the static
        YAML fallback).
        """
        if not self.use_tf:
            return self.T_cam_lidar_static, False
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.lidar_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout_sec),
            )
            T = tf_utils.transform_msg_to_matrix(tf_msg.transform.translation, tf_msg.transform.rotation)
            return T, True
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF lookup {self.camera_frame}<-{self.lidar_frame} failed ({e}); "
                f"using static YAML extrinsic for this frame.",
                throttle_duration_sec=5.0,
            )
            return self.T_cam_lidar_static, False

    def _maybe_deskew(self, xyz, timestamps):
        """
        Apply best-effort per-point motion compensation.

        Only runs when the cloud carries a per-point `timestamp` field AND
        use_tf is enabled with a working TF tree (e.g. odom->lidar_frame
        from wheel odometry or FAST-LIO). Silently (but with a one-time log)
        does nothing otherwise -- this is a "when available" enhancement,
        not a hard requirement, per the task brief.
        """
        if not self.deskew_enabled or timestamps is None or not self.use_tf or self.tf_buffer is None:
            return xyz

        t0, t1 = float(np.min(timestamps)), float(np.max(timestamps))
        if t1 - t0 < 1e-6:
            return xyz  # effectively a single instant; nothing to deskew

        try:
            T0 = self.tf_buffer.lookup_transform(
                self.deskew_reference_frame, self.lidar_frame, rclpy.time.Time(seconds=t0)
            )
            T1 = self.tf_buffer.lookup_transform(
                self.deskew_reference_frame, self.lidar_frame, rclpy.time.Time(seconds=t1)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            if not self._warned_no_deskew:
                self.get_logger().info(
                    f"Per-point deskew unavailable ({e}); publishing raw (non-deskewed) points. "
                    f"This message is logged once.",
                )
                self._warned_no_deskew = True
            return xyz

        M0 = tf_utils.transform_msg_to_matrix(T0.transform.translation, T0.transform.rotation)
        M1 = tf_utils.transform_msg_to_matrix(T1.transform.translation, T1.transform.rotation)

        frac = np.clip((timestamps - t0) / (t1 - t0), 0.0, 1.0)
        per_point = tf_utils.interpolate_transforms(M0, M1, frac)  # (N,4,4), reference_frame -> lidar_frame_at_t

        # Bring every point into the lidar frame *as of t1* (the scan's
        # reference time) so the whole cloud is self-consistent before the
        # static/TF extrinsic (which is defined for a single instant) is
        # applied: p_ref = per_point @ p_point ; p_lidar_at_t1 = inv(M1) @ p_ref.
        p_ref = tf_utils.apply_transforms(per_point, xyz)
        M1_inv = np.linalg.inv(M1)
        xyz_deskewed = (p_ref @ M1_inv[:3, :3].T) + M1_inv[:3, 3]
        return xyz_deskewed

    def _draw_front_marker(self, img, T):
        """White crosshair where the LiDAR +x (front) axis lands; magenta ticks mark the image centre."""
        h, w = img.shape[:2]
        cv2.drawMarker(img, (w // 2, h // 2), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 1)
        pix, front, _ = overlay_math.project_points(np.array([[10.0, 0.0, 0.0]]), T, self.K, self.D)
        if front[0] and np.all(np.isfinite(pix[0])):
            u, v = int(round(pix[0][0])), int(round(pix[0][1]))
            if 0 <= u < w and 0 <= v < h:
                cv2.drawMarker(img, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
                cv2.putText(img, "LiDAR front", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return
        cv2.putText(img, "LiDAR front: off-image", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def on_synced(self, image_msg: Image, cloud_msg: PointCloud2):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        img_t = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        cloud_t = cloud_msg.header.stamp.sec + cloud_msg.header.stamp.nanosec * 1e-9
        offset = img_t - cloud_t
        if now_sec - self._last_offset_log_t > self.log_time_offset_period_sec:
            self.get_logger().info(f"image<->lidar time offset: {offset * 1000.0:+.1f} ms")
            self._last_offset_log_t = now_sec

        if self.K is None:
            self.get_logger().warn(
                "No intrinsics yet (no CameraInfo, no fallback yaml) -- dropping frame.",
                throttle_duration_sec=5.0,
            )
            return

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        want_fields = ["x", "y", "z", "intensity", "timestamp"]
        arrays = pointcloud_utils.pointcloud2_to_arrays(cloud_msg, fields=want_fields, skip=self.skip)
        xyz = arrays["xyz"]

        if xyz.shape[0] > 0:
            mask = pointcloud_utils.finite_mask(xyz)
            xyz = xyz[mask]
            intensity = arrays.get("intensity")
            if intensity is not None:
                intensity = intensity[mask]
            timestamps = arrays.get("timestamp")
            if timestamps is not None:
                timestamps = timestamps[mask]
        else:
            intensity = arrays.get("intensity")
            timestamps = arrays.get("timestamp")

        if self.max_points and xyz.shape[0] > self.max_points:
            stride = int(np.ceil(xyz.shape[0] / self.max_points))
            xyz = xyz[::stride]
            if intensity is not None:
                intensity = intensity[::stride]
            if timestamps is not None:
                timestamps = timestamps[::stride]

        if timestamps is not None and timestamps.shape[0] == xyz.shape[0] and xyz.shape[0] > 0:
            xyz = self._maybe_deskew(xyz, timestamps)

        T_cam_lidar, from_tf = self._resolve_extrinsic(cloud_msg.header.stamp)
        if not from_tf:
            T_cam_lidar = self._tuned_extrinsic()

        out_image = overlay_math.draw_overlay(
            cv_image,
            xyz,
            T_cam_lidar,
            self.K,
            self.D,
            intensity=intensity,
            color_by=self.color_by,
            max_range_m=self.max_range_m,
            min_range_m=self.min_range_m,
            point_size_near_px=self.point_size_near_px,
            point_size_far_px=self.point_size_far_px,
        )

        h_img, w_img = out_image.shape[:2]
        self._aim_report(xyz, T_cam_lidar, w_img, h_img, now_sec)
        if self.show_front_marker:
            self._draw_front_marker(out_image, T_cam_lidar)

        out_msg = self.bridge.cv2_to_imgmsg(out_image, encoding="bgr8")
        out_msg.header = image_msg.header
        self.pub.publish(out_msg)

        self._frame_count += 1
        self._last_fps_count += 1
        if now_sec - self._last_fps_t > 5.0:
            fps = self._last_fps_count / (now_sec - self._last_fps_t)
            self.get_logger().info(f"overlay FPS: {fps:.1f} ({xyz.shape[0]} pts/frame)")
            self._last_fps_t = now_sec
            self._last_fps_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = OverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
