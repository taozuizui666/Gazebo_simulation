from setuptools import find_packages, setup
import os
from glob import glob
from pathlib import Path

package_name = 'my_gazebo_launch'


def resource_files(directory):
    """Keep model:// assets and world files in their original relative paths."""
    groups = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_file() and not any(part.startswith('.') for part in path.parts):
            target = os.path.join('share', package_name, str(path.parent))
            groups.setdefault(target, []).append(str(path))
    return list(groups.items())


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['vehicle_config.h'], 'rl_training': ['native/*.cpp']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ] + resource_files('sdf') + resource_files('models'),
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='zui',
    maintainer_email='zui@todo.todo',
    description='Fixed-speed car simulation with Teensy-compatible lidar observations',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'action_controller = my_gazebo_launch.action_controller:main',
            'lidar_adapter = my_gazebo_launch.lidar_adapter:main',
            'preview_contract = my_gazebo_launch.preview:main',
        ],
    },
)
