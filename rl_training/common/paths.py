"""Resolve the independent artifact directories of the two policy families."""

from importlib import import_module

MODES = ("lidar", "privileged")


def family_config(mode):
    if mode not in MODES:
        raise ValueError(f"未知 RL 类型：{mode}")
    return import_module(f"rl_training.{mode}.config")
