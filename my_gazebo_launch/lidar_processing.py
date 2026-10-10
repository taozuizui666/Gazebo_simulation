"""RPLIDAR A1 native point conventions -> exactly the training distance columns."""

import math
import struct


def float32(value):
    """Match the Teensy float arithmetic used before integer truncation."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def lidar_index(angle_degrees, config):
    if not math.isfinite(angle_degrees) or not 0 <= angle_degrees < 360:
        return -1
    angle_degrees = float32(angle_degrees)
    if angle_degrees >= 360:
        return -1
    angle_integer = int(angle_degrees)
    return int(float32(float32(angle_integer / 360.0) * config.lidar_point_count))


def point_distance_mm(distance_mm, config):
    if not math.isfinite(distance_mm) or not 0 < distance_mm < config.lidar_max_distance_mm:
        return 0
    distance_mm = float32(distance_mm)
    return int(distance_mm) if distance_mm < config.lidar_max_distance_mm else 0


def native_points_to_distances(points, config):
    """points: (clockwise_angle_degrees, distance_mm, quality), in acquisition order.

    Each observation starts with zeros; a bin keeps its last valid point in THIS
    scan. No previous scan is silently reused. The untouched rear sector is zero.
    """
    distances = [0] * config.lidar_point_count
    for angle, distance, quality in points:
        index = lidar_index(angle, config)
        if quality > 0 and config.is_active_index(index):
            value = point_distance_mm(distance, config)
            if value > 0:
                distances[index] = value
    return distances


def scan_to_distances(ranges_m, angle_min, angle_increment, range_min, range_max, config):
    """Convert ROS CCW/m LaserScan into A1 CW/q6/q2 native point conventions.

    lidar_link +X is the A1 zero direction. Mounting yaw belongs to the URDF, not
    to this conversion. A full scan is ordered by increasing native angle.
    """
    if (not math.isfinite(angle_min) or not math.isfinite(angle_increment)
            or angle_increment == 0):
        raise ValueError("LaserScan 的角度参数无效。")
    if not math.isfinite(range_min) or not math.isfinite(range_max) or not 0 <= range_min < range_max:
        raise ValueError("LaserScan 的量程参数无效。")
    points = []
    # Respect this scan's reported validity range. Do not turn the A1 datasheet's
    # typical 0.15 m lower bound into another hard filter on real driver data.
    physical_min = range_min
    physical_max = min(range_max, config.lidar_physical_max_range_m)
    for i, distance_m in enumerate(ranges_m):
        if not math.isfinite(distance_m) or not physical_min <= distance_m < physical_max:
            continue
        angle_ros = angle_min + i * angle_increment
        angle_cw = (-math.degrees(angle_ros)) % 360.0
        # Native angle_q6 / 64, and distance_q2 / 4; then Teensy uint16 truncation.
        angle_q6 = int(math.floor(angle_cw * 64 + 0.5)) % (360 * 64)
        distance_q2 = int(math.floor(float32(float32(distance_m) * 1000) * 4 + 0.5))
        points.append((angle_q6 / 64.0, distance_q2 / 4.0, 1))
    points.sort(key=lambda point: point[0])
    return native_points_to_distances(points, config)


def training_row(distances, label, config):
    """A no-header CSV row: N integer millimetre columns followed by one label."""
    if len(distances) != config.lidar_point_count:
        raise ValueError(f"距离列必须为 {config.lidar_point_count} 个。")
    if isinstance(label, bool) or not isinstance(label, int) or label not in config.labels:
        raise ValueError("训练行只能使用有效转向标签，不包含停车或录制标志。")
    for index, value in enumerate(distances):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < config.lidar_max_distance_mm:
            raise ValueError("距离必须为有效范围内的毫米整数。")
        if not config.is_active_index(index) and value != 0:
            raise ValueError("未采集扇区必须为零。")
    return ",".join(map(str, [*distances, label])) + "\n"
