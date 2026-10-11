"""Resolve the independent artifact directories of the two policy families."""

from importlib import import_module
from pathlib import Path

# 结果图集中存放；权重、日志和原始报告仍在 rl_models。
SIMULATION_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = SIMULATION_ROOT / "rl_models"
RESULTS_DIR = SIMULATION_ROOT / "results"


def training_results_dir(directory):
    """按模型批次映射结果目录；解析 latest_validated 等符号链接。"""
    directory = Path(directory).resolve()
    if directory.is_relative_to(MODELS_DIR):
        relative = directory.relative_to(MODELS_DIR)
    else:
        relative = Path("external") / directory.relative_to(directory.anchor)
    return RESULTS_DIR / "training" / relative


MODES = ("lidar", "privileged")


def family_config(mode):
    if mode not in MODES:
        raise ValueError(f"未知 RL 类型：{mode}")
    return import_module(f"rl_training.{mode}.config")
