"""雷达 PPO 的默认输入/输出路径；通用训练参数见 ../common/config.py。"""

from pathlib import Path

# ===================== 可修改配置区 =====================
OBSERVATION_MODE = "lidar"
SIMULATION_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = SIMULATION_ROOT / "rl_models" / OBSERVATION_MODE
POLICY_DIR = MODEL_DIR / "policies"
LOG_DIR = MODEL_DIR / "logs"
RUNTIME_DIR = MODEL_DIR / "runtime"
# ===================== 配置区结束 =====================
