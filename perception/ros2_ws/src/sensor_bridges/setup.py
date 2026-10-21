from setuptools import find_packages, setup

package_name = 'sensor_bridges'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aayouiyer',
    maintainer_email='aayoushiyer@gmail.com',
    description=(
        'Publishes the Livox Mid-360 (raw UDP) and OAK-D-Lite (DepthAI) as '
        'standard sensor_msgs/PointCloud2 and sensor_msgs/Image topics.'
    ),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'livox_pointcloud_publisher = sensor_bridges.livox_pointcloud_publisher:main',
            'oak_image_publisher = sensor_bridges.oak_image_publisher:main',
        ],
    },
)
