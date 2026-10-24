#!/usr/bin/env python3
"""
Synthetic sensor publisher for developing the dashboard without hardware.

Publishes the same topics the real bridges do -- /oak/rgb/image_raw,
/oak/stereo/depth (16UC1 mm), /oak/rgb/camera_info, /livox/lidar -- from a
made-up scene: flat ground seen by a low, down-pitched camera, one box 1.2 m
ahead, and a LiDAR cloud of the same floor, box and a back wall. The scene is
only roughly consistent between sensors; it exists to exercise the pipeline
(overlay node, detection, cloud renderer, web UI), not to validate geometry.

Run (ROS sourced):  python3 dashboard/tools/fake_sensors.py
"""
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

W, H, F = 640, 480, 500.0
CAM_HEIGHT_M, PITCH_DEG = 0.25, 15.0
BOX_DIST_M = 0.8


def make_depth_mm():
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ray = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u, dtype=float)], axis=-1)
    p = np.radians(PITCH_DEG)
    # camera pitched down about its x axis: rotate rays into a gravity-aligned frame (y down)
    y_world = ray[..., 1] * np.cos(p) + ray[..., 2] * np.sin(p)
    z_world = -ray[..., 1] * np.sin(p) + ray[..., 2] * np.cos(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y_world > 1e-3, CAM_HEIGHT_M / y_world, np.inf)   # distance along the ray to the ground
    depth = t * ray[..., 2]          # z-depth in the camera frame
    box = (np.abs(u - 380) < 90) & (v > 120) & (v < 400)
    depth = np.where(box, np.minimum(depth, BOX_DIST_M), depth)
    depth[~np.isfinite(depth) | (depth > 8.0)] = 0
    return (depth * 1000).astype(np.uint16)


def make_rgb():
    img = np.full((H, W, 3), (70, 80, 90), np.uint8)
    img[240:, :] = (95, 105, 110)
    cv2.rectangle(img, (290, 120), (470, 400), (60, 90, 140), -1)
    cv2.putText(img, "SYNTHETIC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def make_cloud(rng):
    n = 12000
    floor = np.c_[rng.uniform(0.3, 7, n), rng.uniform(-3, 3, n), np.full(n, -CAM_HEIGHT_M)]
    wall = np.c_[np.full(n // 2, 7.0), rng.uniform(-3, 3, n // 2), rng.uniform(-CAM_HEIGHT_M, 1.8, n // 2)]
    side = np.c_[rng.uniform(0.3, 7, n // 3), np.full(n // 3, 3.0), rng.uniform(-CAM_HEIGHT_M, 1.8, n // 3)]
    box = np.c_[rng.uniform(BOX_DIST_M, BOX_DIST_M + 0.3, 2500), rng.uniform(-0.35, 0.05, 2500),
                rng.uniform(-CAM_HEIGHT_M, 0.1, 2500)]
    pts = np.vstack([floor, wall, side, box]) + rng.normal(0, 0.01, (n + n // 2 + n // 3 + 2500, 3))
    return pts.astype(np.float32)


class Fake(Node):
    def __init__(self):
        super().__init__("fake_sensors")
        self.bridge = CvBridge()
        self.rng = np.random.default_rng(0)
        self.rgb, self.depth = make_rgb(), make_depth_mm()
        self.cloud = make_cloud(self.rng).tobytes()   # static scene: build once, restamp each tick
        self.img_pub = self.create_publisher(Image, "/oak/rgb/image_raw", qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, "/oak/stereo/depth", qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, "/oak/rgb/camera_info", 5)
        self.cloud_pub = self.create_publisher(PointCloud2, "/livox/lidar", qos_profile_sensor_data)
        self.create_timer(1 / 30, self.camera_tick)
        self.create_timer(0.1, self.lidar_tick)

    def camera_tick(self):
        stamp = self.get_clock().now().to_msg()
        rgb = self.rgb.copy()
        cv2.putText(rgb, f"{time.time() % 1000:7.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for pub, img, enc in ((self.img_pub, rgb, "bgr8"), (self.depth_pub, self.depth, "16UC1")):
            m = self.bridge.cv2_to_imgmsg(img, encoding=enc)
            m.header.stamp, m.header.frame_id = stamp, "oak_camera"
            pub.publish(m)
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, "oak_camera"
        info.width, info.height, info.distortion_model = W, H, "plumb_bob"
        info.k = [F, 0.0, W / 2, 0.0, F, H / 2, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        info.r = np.eye(3).reshape(-1).tolist()
        info.p = [F, 0.0, W / 2, 0.0, 0.0, F, H / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)

    def lidar_tick(self):
        n = len(self.cloud) // 12
        msg = PointCloud2()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="livox_frame")
        msg.height, msg.width = 1, n
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 12, 12 * n, True
        msg.data = self.cloud
        self.cloud_pub.publish(msg)


def main():
    rclpy.init()
    node = Fake()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
