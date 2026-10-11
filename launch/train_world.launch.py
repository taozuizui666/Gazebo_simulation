"""Fixed-speed car with model-ready RPLIDAR observations."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
    RegisterEventHandler, SetEnvironmentVariable,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition

from my_gazebo_launch.config import (
    DEFAULT_CONFIG_HEADER, DEFAULT_ENTITY_NAME, DEFAULT_URDF, DEFAULT_WORLD,
    GAZEBO_MODEL_DIR, GAZEBO_SYSTEM_MODEL_DIR, SIM_CLOCK_PUBLISH_HZ, load_config,
)
from my_gazebo_launch.robot_description import render_robot, render_world
from my_gazebo_launch.world_config import load_world_settings, resolve_world


def _remove_params(context, paths):
    for path in paths:
        Path(path).unlink(missing_ok=True)
    return []


def _start(context):
    share = Path(get_package_share_directory("my_gazebo_launch"))
    header = Path(LaunchConfiguration("config_header").perform(context)).expanduser().resolve()
    config = load_config(header)
    world = resolve_world(LaunchConfiguration("world").perform(context))
    spawn = [LaunchConfiguration(name).perform(context) for name in ("x", "y", "z", "yaw")]
    if any(value == "" for value in spawn):
        defaults = load_world_settings(world).train_spawn_poses[0]
        spawn = [value if value != "" else str(default) for value, default in zip(spawn, defaults)]
    enable_lidar = IfCondition(LaunchConfiguration("enable_lidar")).evaluate(context)
    description = render_robot(share / DEFAULT_URDF, config, enable_lidar=enable_lidar)
    models = [str(share / GAZEBO_MODEL_DIR)]
    if GAZEBO_SYSTEM_MODEL_DIR.is_dir():
        models.append(str(GAZEBO_SYSTEM_MODEL_DIR))
    if os.environ.get("GAZEBO_MODEL_PATH"):
        models.append(os.environ["GAZEBO_MODEL_PATH"])
    common = {"config_header": str(header), "use_sim_time": True}
    # Gazebo uses -p for log playback. Pass ROS parameters through the supported
    # YAML interface of the distro's gazebo_ros launch instead of short options.
    with NamedTemporaryFile(mode="w", prefix="aicar-gazebo-", suffix=".yaml",
                            encoding="utf-8", delete=False) as params:
        clock_hz = max(SIM_CLOCK_PUBLISH_HZ, 1 / config.sim_time_step_sec)
        params.write(f"/**:\n  ros__parameters:\n    publish_rate: {clock_hz}\n")
    with NamedTemporaryFile(mode="w", prefix="aicar-world-", suffix=".sdf",
                            encoding="utf-8", delete=False) as rendered_world:
        rendered_world.write(render_world(world, config, float(
            LaunchConfiguration("real_time_factor").perform(context)),
            LaunchConfiguration("training_step_plugin").perform(context)))
    gazebo_launch = Path(get_package_share_directory("gazebo_ros")) / "launch/gazebo.launch.py"
    return [
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(
            function=_remove_params, kwargs={"paths": [params.name, rendered_world.name]})])),
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", os.pathsep.join(models)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(gazebo_launch)),
            launch_arguments={
                "world": rendered_world.name, "verbose": "true", "params_file": params.name,
                "gui": LaunchConfiguration("gui").perform(context),
                "pause": LaunchConfiguration("pause").perform(context),
                "lockstep": LaunchConfiguration("lockstep").perform(context),
                "seed": LaunchConfiguration("seed").perform(context),
                "server_required": "true",
            }.items()),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": description, "use_sim_time": True}],
             output="screen"),
        # spawn_entity waits for robot_description and the factory service itself.
        Node(package="gazebo_ros", executable="spawn_entity.py", arguments=[
            "-entity", DEFAULT_ENTITY_NAME, "-topic", "robot_description",
            "-x", spawn[0], "-y", spawn[1], "-z", spawn[2], "-Y", spawn[3],
        ], output="screen"),
        Node(package="my_gazebo_launch", executable="action_controller",
             parameters=[common], output="screen"),
        Node(package="my_gazebo_launch", executable="lidar_adapter",
             condition=IfCondition(LaunchConfiguration("enable_lidar")),
             parameters=[common], output="screen"),
    ]


def generate_launch_description():
    defaults = {
        "config_header": str(DEFAULT_CONFIG_HEADER), "world": DEFAULT_WORLD,
        "gui": "true",
        "enable_lidar": "true",
        "pause": "false", "seed": "42",
        "lockstep": "false",
        "training_step_plugin": "",
        "real_time_factor": "1.0",
        "x": "", "y": "", "z": "0.0", "yaw": "",  # 未指定时使用所选地图的出生点。
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()),
        OpaqueFunction(function=_start),
    ])
