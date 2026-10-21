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
Write a small synthetic rosbag2 with a known scene.

Image + CameraInfo + PointCloud2, all on the default topic names, with a
known scene -- a camera-facing wall at a known, slowly-changing distance --
so the overlay node (and the whole live-vs-replay code path) can be
exercised and visually verified without any real hardware.

Usage:
    ros2 run aurora_lidar_overlay make_synthetic_bag --output /tmp/synthetic_bag --frames 20
    # then, in another terminal, with the overlay node running with
    # use_sim_time:=true against a matching synthetic extrinsics/intrinsics
    # yaml (this script prints the exact commands):
    ros2 bag play /tmp/synthetic_bag --clock
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

from aurora_lidar_overlay import calibration_io, synthetic_scene

WIDTH, HEIGHT = 640, 480
K = np.array([[500.0, 0.0, WIDTH / 2.0], [0.0, 500.0, HEIGHT / 2.0], [0.0, 0.0, 1.0]])
D = np.zeros(5)
T_CAM_LIDAR = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)  # LiDAR: x-forward/y-left/z-up -> camera: x-right/y-down/z-forward, zero translation


def _to_stamp(t_sec: float) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(t_sec)
    msg.nanosec = int((t_sec - int(t_sec)) * 1e9)
    return msg


def make_image_msg(frame_bgr, stamp, frame_id="oak_camera"):
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height, msg.width = frame_bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = frame_bgr.tobytes()
    return msg


def make_camera_info_msg(K, D, width, height, stamp, frame_id="oak_camera"):
    msg = CameraInfo()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.width, msg.height = width, height
    msg.k = K.reshape(-1).tolist()
    msg.d = D.reshape(-1).tolist()
    msg.distortion_model = "plumb_bob"
    msg.r = np.eye(3).reshape(-1).tolist()
    P = np.zeros((3, 4))
    P[:3, :3] = K
    msg.p = P.reshape(-1).tolist()
    return msg


def make_pointcloud2_msg(xyz, stamp, frame_id="livox_frame"):
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    n = xyz.shape[0]
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * n
    msg.is_dense = True
    msg.data = xyz.astype(np.float32).tobytes()
    return msg


def run(output_path, frames=20, fps=10.0, image_topic="/oak/rgb/image_raw", camera_info_topic="/oak/rgb/camera_info",
        lidar_topic="/livox/lidar"):
    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} already exists -- pick a new path or remove it first")

    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=output_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("cdr", "cdr")
    writer.open(storage_options, converter_options)

    writer.create_topic(
        rosbag2_py.TopicMetadata(name=image_topic, type="sensor_msgs/msg/Image", serialization_format="cdr")
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(name=camera_info_topic, type="sensor_msgs/msg/CameraInfo", serialization_format="cdr")
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(name=lidar_topic, type="sensor_msgs/msg/PointCloud2", serialization_format="cdr")
    )

    t0 = 1_700_000_000.0  # arbitrary fixed epoch so bags are reproducible
    for i in range(frames):
        t = t0 + i / fps
        stamp = _to_stamp(t)
        # Slowly breathing distance -- makes drift/misalignment visible across frames.
        distance = 5.0 + 0.5 * np.sin(i / 3.0)

        xyz_lidar, xyz_cam = synthetic_scene.make_synthetic_wall_cloud(
            T_CAM_LIDAR, wall_distance_m=distance, half_size_m=1.2, grid_step_m=0.04, jitter_std_m=0.01, seed=i
        )

        frame = synthetic_scene.blank_image(WIDTH, HEIGHT, value=30)
        corners_lidar = synthetic_scene.wall_corners_lidar(T_CAM_LIDAR, wall_distance_m=distance, half_size_m=1.2)
        from aurora_lidar_overlay import overlay_math

        pixels, in_front, _ = overlay_math.project_points(corners_lidar, T_CAM_LIDAR, K, D)
        if in_front.all():
            pts = pixels.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=True, color=(200, 200, 200), thickness=2)

        t_ns = int(t * 1e9)
        writer.write(image_topic, serialize_message(make_image_msg(frame, stamp)), t_ns)
        writer.write(camera_info_topic, serialize_message(make_camera_info_msg(K, D, WIDTH, HEIGHT, stamp)), t_ns)
        writer.write(lidar_topic, serialize_message(make_pointcloud2_msg(xyz_lidar, stamp)), t_ns)

    del writer
    print(f"Wrote {frames} frames to {output_path}")

    calib_dir = os.path.join(output_path + "_calibration")
    os.makedirs(calib_dir, exist_ok=True)
    calibration_io.save_intrinsics(
        os.path.join(calib_dir, "intrinsics.yaml"), K, D, WIDTH, HEIGHT, extra={"source": "synthetic"}
    )
    calibration_io.save_extrinsic_matrix(
        os.path.join(calib_dir, "extrinsics.yaml"), T_CAM_LIDAR, meta={"source": "synthetic"}
    )
    print(f"Saved matching ground-truth intrinsics/extrinsics to {calib_dir}/")
    print()
    print("Replay it with:")
    print(f"  ros2 bag play {output_path} --clock")
    print("and, in another terminal (with the workspace sourced):")
    print(
        f"  ros2 run aurora_lidar_overlay overlay_node --ros-args "
        f"-p use_sim_time:=true "
        f"-p extrinsic_yaml:={calib_dir}/extrinsics.yaml "
        f"-p intrinsic_yaml:={calib_dir}/intrinsics.yaml"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="Bag directory to create (must not already exist)")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()
    run(args.output, frames=args.frames, fps=args.fps)


if __name__ == "__main__":
    main()
