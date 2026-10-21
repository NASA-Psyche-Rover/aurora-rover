#!/usr/bin/env python3
"""
Publishes the Livox Mid-360 as a standard sensor_msgs/PointCloud2 topic.

ros2_camera_lidar_fusion expects a generic PointCloud2 (it was built against
a RoboSense LiDAR) -- there's no livox_ros_driver2 build on this box, and
building one from source (it needs Livox-SDK2) is a heavier lift than this
sensor needs, since we already have a working raw-UDP decoder (the same one
scripts/lidar_live_map.py and scripts/live_camera_lidar_fusion.py use). This
node reuses that decoder and republishes each incoming UDP packet's points
as one PointCloud2 message -- no ROS2-specific Livox driver required.

Run (after `colcon build` + sourcing install/setup.bash):
    ros2 run sensor_bridges livox_pointcloud_publisher
"""
import socket
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header

HOST_IP = "192.168.1.5"
POINT_PORT = 56301
HEADER_FMT = "<BHHHHBBB12sIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
POINT_DTYPE = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("r", "u1"), ("t", "u1")])
POINT_SIZE = 14


class LivoxPointCloudPublisher(Node):
    def __init__(self):
        super().__init__("livox_pointcloud_publisher")
        self.declare_parameter("topic", "/livox/lidar")
        self.declare_parameter("frame_id", "livox_frame")
        topic = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.pub = self.create_publisher(PointCloud2, topic, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.sock.bind((HOST_IP, POINT_PORT))
        self.sock.settimeout(0.0)  # non-blocking; polled from a timer

        self.packet_count = 0
        # ~10 Hz publish rate: batch packets between ticks rather than
        # publishing one PointCloud2 per ~96-point UDP packet (Livox sends
        # thousands of packets/sec -- that would flood the topic).
        self.timer = self.create_timer(0.1, self.publish_batch)
        self.get_logger().info(f"Publishing {topic} (frame_id={self.frame_id}) from {HOST_IP}:{POINT_PORT}")

    def publish_batch(self):
        batch = []
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                break
            if len(data) < HEADER_SIZE:
                continue
            try:
                hdr = struct.unpack_from(HEADER_FMT, data, 0)
                n = min(hdr[3], len(data[HEADER_SIZE:]) // POINT_SIZE)
            except struct.error:
                continue
            if n == 0:
                continue
            pts = np.frombuffer(data[HEADER_SIZE:HEADER_SIZE + n * POINT_SIZE], dtype=POINT_DTYPE)
            x, y, z = pts["x"].astype(np.float32), pts["y"].astype(np.float32), pts["z"].astype(np.float32)
            valid = ~((x == 0) & (y == 0) & (z == 0))
            if not valid.any():
                continue
            xyz_m = np.stack([x[valid], y[valid], z[valid]], axis=1) / 1000.0
            batch.append(xyz_m)
            self.packet_count += 1

        if not batch:
            return
        points = np.concatenate(batch, axis=0)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        msg = create_cloud_xyz32(header, points.tolist())
        self.pub.publish(msg)

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LivoxPointCloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
