#!/usr/bin/env python3
"""特权距离场 PPO：训练、续训和独立评估入口。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl_training.common.train_rl import main

if __name__ == "__main__":
    main("privileged")
