#!/usr/bin/env python3
"""
Publishes the OAK-D-Lite's RGB feed as a standard sensor_msgs/Image topic,
plus a matching sensor_msgs/CameraInfo on a sibling topic (same header
stamp/frame on every tick) so consumers -- notably
aurora_lidar_overlay's overlay_node -- get real intrinsics instead of a
guessed fallback.

No depthai-ros build on this box -- reuses the repo's own validated v3
DepthAI pipeline (src/camera_stream.py's OakDLiteCamera, the same class
scripts/live_camera_lidar_fusion.py drives) instead of standing up a second,
separate camera driver just for this integration.

CameraInfo needs a real calibration file (see
aurora_lidar_overlay's README > Calibration for how to generate one --
factory read or, better, a checkerboard capture at this exact resolution).
There's no silent fallback here: if `intrinsics_yaml` doesn't point at a
real file, this node logs an error and publishes Image only, rather than
publishing a CameraInfo built from a guess that looks legitimate to a
downstream consumer.

Run (after `colcon build` + sourcing install/setup.bash):
    ros2 run sensor_bridges oak_image_publisher --ros-args \\
        -p intrinsics_yaml:=/path/to/intrinsics.yaml
"""
import os
import sys

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

# colcon copies this file into ros2_ws/install/... at build time, so a
# relative Path(__file__).parents[N] walk breaks between "run from source"
# and "run installed" -- an env var (with this machine's real clone path as
# the fallback) is the reliable way to find src/camera_stream.py either way.
REPO_ROOT = os.environ.get("PSYCHE_OBJDETEC_ROOT", "/home/aurora/psyche-objdetec/perception")
sys.path.insert(0, REPO_ROOT)

from src.camera_stream import OakDLiteCamera  # noqa: E402


def _load_camera_info(path, width, height, frame_id):
    """Builds a static CameraInfo from an aurora_lidar_overlay-schema
    intrinsics.yaml (camera_matrix/distortion_coefficients keys -- shared
    with ros2_camera_lidar_fusion's format)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1)

    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = "plumb_bob"
    msg.k = K.reshape(-1).tolist()
    msg.d = D.tolist()
    msg.r = np.eye(3).reshape(-1).tolist()
    P = np.zeros((3, 4))
    P[:3, :3] = K
    msg.p = P.reshape(-1).tolist()
    return msg


class OakImagePublisher(Node):
    def __init__(self):
        super().__init__("oak_image_publisher")
        self.declare_parameter("topic", "/oak/rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/oak/rgb/camera_info")
        self.declare_parameter("depth_topic", "/oak/stereo/depth")   # "" disables; 16UC1 millimeters, RGB-aligned
        self.declare_parameter("frame_id", "oak_camera")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("intrinsics_yaml", "")
        topic = self.get_parameter("topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        width = self.get_parameter("width").value
        height = self.get_parameter("height").value
        intrinsics_yaml = self.get_parameter("intrinsics_yaml").value

        self.pub = self.create_publisher(Image, topic, 5)
        depth_topic = self.get_parameter("depth_topic").value
        self.depth_pub = self.create_publisher(Image, depth_topic, 5) if depth_topic else None
        self._last_depth_id = None
        self.bridge = CvBridge()

        self.camera_info_msg = None
        self.caminfo_pub = None
        if intrinsics_yaml and os.path.isfile(intrinsics_yaml):
            self.camera_info_msg = _load_camera_info(intrinsics_yaml, width, height, self.frame_id)
            self.caminfo_pub = self.create_publisher(CameraInfo, camera_info_topic, 5)
            self.get_logger().info(f"Publishing {camera_info_topic} from {intrinsics_yaml}")
        else:
            self.get_logger().error(
                f"No valid intrinsics_yaml given (got '{intrinsics_yaml}') -- "
                f"NOT publishing {camera_info_topic}. Image-only mode: downstream "
                f"consumers will have to fall back to their own guessed intrinsics. "
                f"Run aurora_lidar_overlay's get_intrinsics_checkerboard (or "
                f"get_intrinsics_factory as a quick placeholder) and pass its output "
                f"here via -p intrinsics_yaml:=<path>."
            )

        self.get_logger().info("Opening OAK-D-Lite ...")
        self.cam = OakDLiteCamera(rgb_size=(width, height), fps=30)
        self.get_logger().info(f"Publishing {topic} (frame_id={self.frame_id})")

        self.timer = self.create_timer(1.0 / 30.0, self.tick)

    def tick(self):
        rgb, depth = self.cam.read()
        if rgb is None:
            return
        stamp = self.get_clock().now().to_msg()

        msg = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

        # read() hands back the cached depth between camera ticks; publish each frame once.
        if self.depth_pub is not None and depth is not None and depth is not self._last_depth_id:
            self._last_depth_id = depth
            dmsg = self.bridge.cv2_to_imgmsg(np.ascontiguousarray(depth, dtype=np.uint16), encoding="16UC1")
            dmsg.header.stamp = stamp
            dmsg.header.frame_id = self.frame_id
            self.depth_pub.publish(dmsg)

        if self.caminfo_pub is not None:
            self.camera_info_msg.header.stamp = stamp
            self.caminfo_pub.publish(self.camera_info_msg)

    def destroy_node(self):
        self.cam.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OakImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
