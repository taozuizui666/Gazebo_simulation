"""特权距离场 PPO 的默认输入/输出路径；通用训练参数见 ../common/config.py。"""

from pathlib import Path

# ===================== 可修改配置区 =====================
OBSERVATION_MODE = "privileged"
SIMULATION_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = SIMULATION_ROOT / "rl_models" / OBSERVATION_MODE
POLICY_DIR = MODEL_DIR / "policies"
LOG_DIR = MODEL_DIR / "logs"
RUNTIME_DIR = MODEL_DIR / "runtime"
# 特权教师输入：距离场与几何安全间距
GRID_ROWS = 17                       # 纵向：从车后到车前
GRID_COLS = 17                       # 横向：从车右到车左
GRID_BACK_M = 0.4
GRID_FRONT_M = 1.4
GRID_HALF_WIDTH_M = 0.9
DISTANCE_CLIP_M = 0.5                # 距离场归一化范围；负值表示障碍内部
BODY_CLEARANCE_RADIUS_M = 0.085      # 奖励用保守车体半径；终止仍以真实接触为准
FORWARD_CHECK_M = 1.5
FORWARD_CHECK_SAMPLES = 61
OBSTACLE_Z_MIN_M = 0.02
OBSTACLE_Z_MAX_M = 0.22
# ===================== 配置区结束 =====================


