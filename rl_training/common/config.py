"""持续避障巡航配置；车辆/雷达协议直接读取共享 C++ 宏。"""

from dataclasses import dataclass
import math
from pathlib import Path

from my_gazebo_launch.config import (
    DEFAULT_CONFIG_HEADER, DEFAULT_WORLD, DEFAULT_SPAWN_X, DEFAULT_SPAWN_Y,
    DEFAULT_SPAWN_Z, DEFAULT_SPAWN_YAW, load_config,
    IGNORED_CONTACT_MODELS,
)

# ===================== 可修改配置区 =====================
SIMULATION_ROOT = Path(__file__).resolve().parents[2]
RL_MODEL_DIR = SIMULATION_ROOT / "rl_models"
CONFIG_HEADER = DEFAULT_CONFIG_HEADER
WORLD = DEFAULT_WORLD
SPAWN_POSE = (DEFAULT_SPAWN_X, DEFAULT_SPAWN_Y, DEFAULT_SPAWN_Z, DEFAULT_SPAWN_YAW)
# 每个训练实例有固定出生点，碰撞后仍回到该点。两个行驶方向都参加训练。
TRAIN_SPAWN_POSES = (
    SPAWN_POSE,
    (0.05, -3.0, 0.0, math.pi / 2),
    (-1.2, -4.94, 0.0, math.pi),
    (-2.47, -3.0, 0.0, -math.pi / 2),
)
NUM_TRAIN_ENVS = 4
EVAL_SPAWN_POSES = (
    SPAWN_POSE,
    (0.05, -3.7, 0.0, -math.pi / 2),
    (-2.47, -3.6, 0.0, math.pi / 2),
    (-1.45, -4.94, 0.0, 0.0),
    (*SPAWN_POSE[:3], math.pi),
    (-1.8, -1.30, 0.0, 0.0),
    (0.05, -3.0, 0.0, math.pi / 2),
    (-2.47, -3.0, 0.0, -math.pi / 2),
)
# 当前 train_world3 环形道路的外边界；更换世界时同步修改或显式设为 None。
WORKSPACE_BOUNDS = (-2.9, 0.5, -5.45, -0.8)  # xmin, xmax, ymin, ymax

# 两个独立 Gazebo master / ROS domain，不控制手动启动的小车。
TRAIN_ROS_DOMAIN = 81
EVAL_ROS_DOMAIN = 82
TRAIN_GAZEBO_PORT = 11451
EVAL_GAZEBO_PORT = 11452
SEED = 42
DEVICE = "cpu"  # 小型 MLP；主要耗时在仿真采样
TORCH_THREADS = 2
STARTUP_TIMEOUT_SEC = 60.0
GUI_STARTUP_SETTLE_SEC = 3.0  # 等待图形界面初始化，避免其控制消息打断第一次物理步进
DATA_TIMEOUT_SEC = 10.0
RESET_SETTLE_SEC = 0.4
STEP_SIM_TIME_SEC = 0.2  # 仿真时间；仅雷达模式会因低扫描频率增大到至少一圈
STATE_MAX_AGE_SEC = 0.03
COMMAND_HEARTBEAT_SEC = 0.1
MAX_STEPS_PER_EPISODE = 750  # 150 秒；正常巡航的采样截断，不是失败
STUCK_WINDOW_SEC = 3.0
STUCK_MIN_TRAVEL_M = 0.02
MIN_CRUISE_SPEED_RATIO = 0.75  # 3 秒内平均前进速度低于目标的 75% 则失败
SPEED_TOLERANCE_RATIO = 0.20  # 评估统计：目标速度 ±20%
LOOP_WINDOW_SEC = 12.0
LOOP_MIN_DURATION_SEC = 6.0
LOOP_MIN_TURN_RAD = 5.5
LOOP_MAX_SPAN_M = 0.65  # 大圈沿道路巡航可以，小范围反复绕圈不可以

# 任务：持续前进、避障、平滑转向；接触障碍或卡住才失败。
DISTANCE_REWARD_PER_M = 10.0
SPEED_ERROR_PENALTY_PER_SEC = 3.0
NEAR_OBSTACLE_DISTANCE_MM = 250.0
NEAR_OBSTACLE_PENALTY_PER_SEC = 0.5
FRONT_SAFETY_DISTANCE_MM = 450.0
FRONT_SAFETY_HALF_ANGLE_DEG = 20.0
FRONT_RISK_PENALTY_PER_SEC = 2.0
TURN_PENALTY_PER_SEC = 0.15
STEERING_CHANGE_PENALTY = 0.1  # 0↔最后标签时扣满；同一转向保持不扣
COLLISION_REWARD = -20.0
STUCK_REWARD = -20.0
LOOP_REWARD = -20.0
SLOW_REWARD = -20.0
OUT_OF_BOUNDS_REWARD = -20.0

ALGORITHM = "PPO"
LEARNING_RATE = 3e-4
N_STEPS = 512  # 每个并行环境的 rollout 长度
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEFF = 0.005
VALUE_COEFF = 0.5
MAX_GRAD_NORM = 0.5
POLICY_NET_ARCH = [128, 128]
TOTAL_TIMESTEPS = 500000
SAVE_FREQ = 10000
EVAL_FREQ = 10000
EVAL_EPISODES = 3
MAX_TRAINING_HOURS = 4.5  # 单次训练预算；可由 --hours 修改
# ===================== 配置区结束 =====================


@dataclass(frozen=True)
class EnvironmentSettings:
    step_sim_sec: float = STEP_SIM_TIME_SEC
    reset_settle_sec: float = RESET_SETTLE_SEC
    startup_timeout_sec: float = STARTUP_TIMEOUT_SEC
    data_timeout_sec: float = DATA_TIMEOUT_SEC
    state_max_age_sec: float = STATE_MAX_AGE_SEC
    command_heartbeat_sec: float = COMMAND_HEARTBEAT_SEC
    max_steps: int = MAX_STEPS_PER_EPISODE
    stuck_window_sec: float = STUCK_WINDOW_SEC
    stuck_min_travel_m: float = STUCK_MIN_TRAVEL_M
    min_cruise_speed_ratio: float = MIN_CRUISE_SPEED_RATIO
    speed_tolerance_ratio: float = SPEED_TOLERANCE_RATIO
    loop_window_sec: float = LOOP_WINDOW_SEC
    loop_min_duration_sec: float = LOOP_MIN_DURATION_SEC
    loop_min_turn_rad: float = LOOP_MIN_TURN_RAD
    loop_max_span_m: float = LOOP_MAX_SPAN_M
    workspace_bounds: tuple | None = WORKSPACE_BOUNDS

    def validate(self):
        for name, value in self.__dict__.items():
            if name == "workspace_bounds":
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须大于零。")
        if not isinstance(self.max_steps, int):
            raise ValueError("max_steps 必须为整数。")
        if self.loop_window_sec < max(self.loop_min_duration_sec, self.stuck_window_sec):
            raise ValueError("绕圈观察窗口必须覆盖最小判定时长和速度窗口。")
        if self.workspace_bounds is not None:
            bounds = self.workspace_bounds
            if (len(bounds) != 4 or not all(math.isfinite(v) for v in bounds)
                    or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
                raise ValueError("workspace_bounds 应为 xmin, xmax, ymin, ymax。")
        return self


@dataclass(frozen=True)
class RewardSettings:
    distance_per_m: float = DISTANCE_REWARD_PER_M
    speed_error_per_sec: float = SPEED_ERROR_PENALTY_PER_SEC
    near_distance_mm: float = NEAR_OBSTACLE_DISTANCE_MM
    near_penalty_per_sec: float = NEAR_OBSTACLE_PENALTY_PER_SEC
    steering_change_penalty: float = STEERING_CHANGE_PENALTY
    front_distance_mm: float = FRONT_SAFETY_DISTANCE_MM
    front_half_angle_deg: float = FRONT_SAFETY_HALF_ANGLE_DEG
    front_risk_per_sec: float = FRONT_RISK_PENALTY_PER_SEC
    turn_per_sec: float = TURN_PENALTY_PER_SEC
    collision: float = COLLISION_REWARD
    stuck: float = STUCK_REWARD
    loop: float = LOOP_REWARD
    slow: float = SLOW_REWARD
    out_of_bounds: float = OUT_OF_BOUNDS_REWARD


def vehicle_config(header=None):
    return load_config(header or CONFIG_HEADER)
