"""Read the same numeric C++ macros used by the future Teensy speed controller."""

from dataclasses import dataclass
from pathlib import Path
import math
import re

# 文件和话题地址集中在这里；速度、标签、雷达参数在 vehicle_config.h 顶部。
DEFAULT_CONFIG_HEADER = Path(__file__).with_name("vehicle_config.h")
DEFAULT_WORLD = "assets/worlds/simple/train_world3.sdf"  # world / --world 参数省略时使用。
DEFAULT_URDF = "assets/urdf/simple_car1.urdf"
GAZEBO_MODEL_DIR = "assets/models"
GAZEBO_SYSTEM_MODEL_DIR = Path("/usr/share/gazebo-11/models")
DEFAULT_ENTITY_NAME = "my_car"
LABEL_TOPIC = "/steering_label"
CMD_VEL_TOPIC = "/cmd_vel"
WHEEL_TARGET_TOPIC = "/wheel_targets_mps"  # data[0] 左轮，data[1] 右轮
RAW_SCAN_TOPIC = "/scan_raw"
OBSERVATION_TOPIC = "/lidar_distances"  # 雷达基线/后续学生的数据列；特权教师不读取
CONTACT_TOPIC = "/collision_contacts"  # 仿真训练奖励，绝不作为策略观测
IGNORED_CONTACT_MODELS = ("ground_plane",)  # 地面接触不算撞墙，原生插件与 ROS 诊断共用
SIM_CLOCK_PUBLISH_HZ = 100.0


@dataclass(frozen=True)
class VehicleConfig:
    label_count: int
    forward_speed_mps: float
    max_wheel_speed_ratio: float
    wheel_radius_m: float
    wheel_separation_m: float
    command_publish_hz: float
    command_timeout_ms: int
    stop_command: int
    sim_max_wheel_accel_mps2: float
    sim_max_wheel_torque_nm: float
    sim_drive_update_hz: float
    sim_time_step_sec: float
    lidar_point_count: int
    lidar_front_sector_only: int
    lidar_max_distance_mm: int
    lidar_mount_yaw_deg: float
    lidar_x_m: float
    lidar_y_m: float
    lidar_z_m: float
    lidar_scan_hz: float
    lidar_raw_samples: int
    lidar_min_range_m: float
    lidar_physical_max_range_m: float
    lidar_noise_stddev_m: float

    @property
    def straight_label(self):
        return self.label_count // 2

    @property
    def labels(self):
        return tuple(range(self.label_count))

    def is_active_index(self, index):
        return 0 <= index < self.lidar_point_count and (
            not self.lidar_front_sector_only
            or index < self.lidar_point_count // 4
            or index > 3 * self.lidar_point_count // 4)

    def validate(self):
        if not 3 <= self.label_count <= self.stop_command or self.label_count % 2 != 1:
            raise ValueError("标签数量必须为不超过停车指令的奇数，至少 3 个。")
        if not 0 <= self.max_wheel_speed_ratio <= 1:
            raise ValueError("MAX_WHEEL_SPEED_RATIO 必须在 0～1，保持左右轮不倒转。")
        for name in ("forward_speed_mps", "wheel_radius_m", "wheel_separation_m",
                     "command_publish_hz", "sim_drive_update_hz", "sim_time_step_sec", "lidar_scan_hz",
                     "sim_max_wheel_torque_nm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数。")
        if self.command_timeout_ms <= 0:
            raise ValueError("COMMAND_TIMEOUT_MS 必须为正整数。")
        n = self.lidar_point_count
        if not 4 <= n <= 360 or n % 4:
            raise ValueError("整数角度分桶要求 N 为 4 的倍数，且不超过 360。")
        if self.lidar_front_sector_only not in (0, 1):
            raise ValueError("FRONT_SECTOR_ONLY 必须为 0 或 1。")
        if not 1 < self.lidar_max_distance_mm <= 65535:
            raise ValueError("距离上界必须在 2～65535 毫米。")
        if not 0 < self.lidar_min_range_m < self.lidar_max_distance_mm / 1000:
            raise ValueError("雷达最小量程必须在训练有效距离范围内。")
        if self.lidar_physical_max_range_m < self.lidar_max_distance_mm / 1000:
            raise ValueError("物理量程不得小于训练距离上界。")
        if type(self.lidar_raw_samples) is not int or self.lidar_raw_samples < 2:
            raise ValueError("原始射线数必须为至少 2 的整数；未命中的角度桶保留为零。")
        if self.lidar_noise_stddev_m < 0 or self.sim_max_wheel_accel_mps2 < 0:
            raise ValueError("噪声和加速度上限不能为负。")
        return self


def load_config(header=None):
    path = Path(header) if header else DEFAULT_CONFIG_HEADER
    text = path.read_text(encoding="utf-8")
    numeric = {}
    for name, value in re.findall(r"^\s*#define\s+AICAR_(\w+)\s+([^\n]+)", text, re.M):
        value = value.split("//", 1)[0].strip()
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
            numeric[name.lower()] = value
    values = {}
    for name, field in VehicleConfig.__dataclass_fields__.items():
        if name not in numeric:
            raise ValueError(f"{path}: AICAR_{name.upper()} 缺失或不是普通数字。")
        number = float(numeric[name])
        if not math.isfinite(number):
            raise ValueError(f"{name} 必须为有限数值。")
        if field.type is int:
            if not number.is_integer():
                raise ValueError(f"{name} 必须为整数。")
            values[name] = int(number)
        else:
            values[name] = number
    return VehicleConfig(**values).validate()
