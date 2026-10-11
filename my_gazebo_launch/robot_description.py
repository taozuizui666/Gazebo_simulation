"""Render the existing car using the shared speed/geometry/lidar configuration."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .config import (CMD_VEL_TOPIC, CONTACT_TOPIC, RAW_SCAN_TOPIC,
                     DEFAULT_ENTITY_NAME, IGNORED_CONTACT_MODELS)


def _text(parent, name, value):
    element = parent.find(name)
    if element is None:
        element = ET.SubElement(parent, name)
    element.text = str(value)
    return element


def render_robot(template, config, enable_lidar=True):
    root = ET.parse(Path(template)).getroot()
    for name, side in (("left_wheel", 1), ("right_wheel", -1)):
        joint = root.find(f"./joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"车辆模板缺少 {name}。")
        joint.find("origin").set(
            "xyz", f"0 {side * config.wheel_separation_m / 2} {config.wheel_radius_m}")
        # Existing cylinder frame is rotated +pi/2 about X; local -Z is body +Y.
        joint.find("axis").set("xyz", "0 0 -1")
        link = root.find(f"./link[@name='{name}_link']")
        for cylinder in link.findall("./*/geometry/cylinder"):
            cylinder.set("radius", str(config.wheel_radius_m))
        mass = float(link.find("./inertial/mass").get("value"))
        width = float(link.find("./collision/geometry/cylinder").get("length"))
        inertia = link.find("./inertial/inertia")
        radial = mass * (3 * config.wheel_radius_m ** 2 + width ** 2) / 12
        inertia.set("ixx", str(radial))
        inertia.set("iyy", str(radial))
        inertia.set("izz", str(mass * config.wheel_radius_m ** 2 / 2))

    lidar_joint = root.find("./joint[@name='lidar']/origin")
    lidar_joint.set("xyz", f"{config.lidar_x_m} {config.lidar_y_m} {config.lidar_z_m}")
    lidar_joint.set("rpy", f"0 0 {math.radians(config.lidar_mount_yaw_deg)}")
    drive = root.find("./gazebo/plugin[@filename='libgazebo_ros_diff_drive.so']")
    if drive is None:
        raise ValueError("车辆模板缺少 ROS 2 差速速度插件。")
    radius = drive.find("wheel_radius")
    if radius is not None:
        drive.remove(radius)
    _text(drive, "wheel_separation", config.wheel_separation_m)
    _text(drive, "wheel_diameter", 2 * config.wheel_radius_m)
    _text(drive, "max_wheel_acceleration", config.sim_max_wheel_accel_mps2)
    _text(drive, "max_wheel_torque", config.sim_max_wheel_torque_nm)
    _text(drive, "update_rate", config.sim_drive_update_hz)
    _text(drive, "robot_base_frame", "base_link")
    _text(drive, "odometry_frame", "odom")
    _text(drive, "odometry_source", 1)
    _text(drive, "publish_odom", "true")
    _text(drive, "publish_odom_tf", "true")
    _text(drive.find("ros"), "remapping", f"cmd_vel:={CMD_VEL_TOPIC}")

    sensor = root.find("./gazebo[@reference='lidar_link']/sensor")
    if sensor is None:
        raise ValueError("车辆模板缺少雷达。")
    _text(sensor, "update_rate", config.lidar_scan_hz)
    _text(sensor, "always_on", "true")
    ray = sensor.find("ray")
    horizontal = ray.find("./scan/horizontal")
    _text(horizontal, "samples", config.lidar_raw_samples)
    _text(horizontal, "resolution", 1)
    _text(horizontal, "min_angle", -math.pi)
    _text(horizontal, "max_angle", math.pi - 2 * math.pi / config.lidar_raw_samples)
    bounds = ray.find("range")
    _text(bounds, "min", config.lidar_min_range_m)
    _text(bounds, "max", config.lidar_physical_max_range_m)
    _text(bounds, "resolution", 0.00025)
    misplaced_noise = sensor.find("noise")
    if misplaced_noise is not None:
        sensor.remove(misplaced_noise)
    noise = ray.find("noise")
    if noise is None:
        noise = ET.SubElement(ray, "noise")
    _text(noise, "type", "gaussian")
    _text(noise, "mean", 0.0)
    _text(noise, "stddev", config.lidar_noise_stddev_m)
    laser_plugin = sensor.find("plugin")
    old_frame = laser_plugin.find("frameName")
    if old_frame is not None:
        laser_plugin.remove(old_frame)
    _text(laser_plugin, "frame_name", "lidar_link")
    _text(laser_plugin, "output_type", "sensor_msgs/LaserScan")
    _text(laser_plugin.find("ros"), "remapping", f"~/out:={RAW_SCAN_TOPIC}")
    if not enable_lidar:
        root.find("./gazebo[@reference='lidar_link']").remove(sensor)

    # Chassis and wheels cover the car footprint. Ground/self contacts are
    # filtered by the RL environment; contact information is reward-only.
    for link_name in ("base_link", "left_wheel_link", "right_wheel_link"):
        link = root.find(f"./link[@name='{link_name}']")
        collisions = link.findall("collision")
        gazebo = root.find(f"./gazebo[@reference='{link_name}']")
        if gazebo is None:
            gazebo = ET.SubElement(root, "gazebo", reference=link_name)
        contact_sensor = ET.SubElement(gazebo, "sensor", type="contact", name=f"{link_name}_contacts")
        _text(contact_sensor, "always_on", "true")
        _text(contact_sensor, "update_rate", config.sim_drive_update_hz)
        contact = ET.SubElement(contact_sensor, "contact")
        for index, collision in enumerate(collisions):
            name = f"{link_name}_collision_{index}"
            collision.set("name", name)
            # Gazebo Classic's URDF converter appends this suffix to collision names.
            ET.SubElement(contact, "collision").text = name + "_collision"
        plugin = ET.SubElement(contact_sensor, "plugin", name=f"{link_name}_bumper",
                               filename="libgazebo_ros_bumper.so")
        ros = ET.SubElement(plugin, "ros")
        _text(ros, "remapping", f"bumper_states:={CONTACT_TOPIC}")
        _text(plugin, "frame_name", link_name)
    ET.indent(root)
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode")


def render_world(template, config, real_time_factor=1.0, training_step_plugin=""):
    """Set the physics step explicitly so RL's paused multi-step is exact."""
    root = ET.parse(Path(template)).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError("SDF 文件缺少 world。")
    physics = world.find("physics")
    if physics is None:
        physics = ET.SubElement(world, "physics", name="aicar_physics", type="ode")
    _text(physics, "max_step_size", config.sim_time_step_sec)
    if not math.isfinite(real_time_factor) or real_time_factor < 0:
        raise ValueError("real_time_factor 必须为非负有限数；0 表示不限速。")
    _text(physics, "real_time_update_rate", real_time_factor / config.sim_time_step_sec)
    state = ET.SubElement(world, "plugin", name="aicar_world_state",
                          filename="libgazebo_ros_state.so")
    ros = ET.SubElement(state, "ros")
    _text(ros, "namespace", "/")
    _text(state, "update_rate", config.sim_drive_update_hz)
    if training_step_plugin:
        plugin = ET.SubElement(world, "plugin", name="aicar_training_step",
                               filename=str(training_step_plugin))
        _text(plugin, "entity", DEFAULT_ENTITY_NAME)
        _text(plugin, "ignored_models", " ".join(IGNORED_CONTACT_MODELS))
    ET.indent(root)
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode")
