"""Load immutable SB3 checkpoints saved before or after directory organization."""

import sys

from stable_baselines3 import PPO
from rl_training.lidar import features


def load_model(path, **kwargs):
    # Historical lidar checkpoints pickle this module name. Resolve it in
    # memory instead of rewriting weights or leaving an obsolete code entry.
    sys.modules.setdefault("rl_training.policy", features)
    return PPO.load(path, **kwargs)
