"""Target wheel speeds, independent of ROS, PWM, motor current and PID gains."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WheelTargets:
    left_mps: float
    right_mps: float
    left_rad_s: float
    right_rad_s: float
    left_rpm: float
    right_rpm: float
    linear_mps: float
    angular_rad_s: float


def wheel_targets(label, config):
    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError("操作标签必须为整数。")
    if label == config.stop_command:
        left = right = 0.0
    elif 0 <= label < config.label_count:
        steering = (config.straight_label - label) / config.straight_label
        delta = config.forward_speed_mps * config.max_wheel_speed_ratio * steering
        left = config.forward_speed_mps - delta
        right = config.forward_speed_mps + delta
    else:
        raise ValueError(f"标签应为 0～{config.label_count - 1}，或停车 {config.stop_command}。")
    left_rad_s, right_rad_s = left / config.wheel_radius_m, right / config.wheel_radius_m
    return WheelTargets(
        left, right, left_rad_s, right_rad_s,
        left_rad_s * 60 / (2 * math.pi), right_rad_s * 60 / (2 * math.pi),
        (left + right) / 2, (right - left) / config.wheel_separation_m)
