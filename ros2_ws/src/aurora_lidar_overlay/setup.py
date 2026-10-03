import os
from glob import glob

from setuptools import find_packages, setup

package_name = "aurora_lidar_overlay"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config", "calibration"), glob("config/calibration/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="aurora",
    maintainer_email="aayoushiyer@gmail.com",
    description="Waymo-style depth-colored LiDAR-on-camera overlay for the AURORA rover, live and rosbag-replay.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "overlay_node = aurora_lidar_overlay.overlay_node:main",
            "get_extrinsics = aurora_lidar_overlay.get_extrinsics:main",
            "check_calibration = aurora_lidar_overlay.check_calibration:main",
            "get_intrinsics_factory = aurora_lidar_overlay.get_intrinsics_factory:main",
            "get_intrinsics_checkerboard = aurora_lidar_overlay.get_intrinsics_checkerboard:main",
            "visual_alignment_check = aurora_lidar_overlay.visual_alignment_check:main",
            "make_synthetic_bag = aurora_lidar_overlay.make_synthetic_bag:main",
        ],
    },
)
