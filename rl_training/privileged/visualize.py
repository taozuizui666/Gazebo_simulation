#!/usr/bin/env python3
"""按地图列表依次打开 Gazebo，观看特权 PPO 控制小车。"""
from pathlib import Path
from dataclasses import asdict, replace
import argparse
import json
import math
import os
import sys

# ===================== 可修改配置区 =====================
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "rl_models/privileged/policies/latest_validated/model.zip"
WORLD_DIR = ROOT / "assets/worlds/simple"
WORLDS = [WORLD_DIR / name for name in (
    "train_world3.sdf",
    "rounded_rectangle_001.sdf",)]  # 可添加其他 SDF 文件名，按列表顺序观看。
OUTPUT_DIR = ROOT / "rl_models/privileged/policies"  # 每张地图单独保存评估报告和轨迹。
EPISODES_PER_MAP = 1                # 按地图 JSON 的评估出生点循环。
MAX_STEPS_PER_EPISODE = 900         # 当前模型每步 0.2 秒，最多约 3 分钟；失败提前结束。
POSITION_JITTER_M = 0.0             # 初始 x/y 分别均匀扰动 ±此值。
YAW_JITTER_DEG = 0.0                # 初始朝向均匀扰动 ±此角度。
SEED = 42
EVAL_ROS_DOMAIN = 82
EVAL_GAZEBO_PORT = 11452
# ===================== 配置区结束 =====================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="检查模型/地图/配置，不启动仿真")
    options = parser.parse_args()

    # 在新 Python 进程中加载 ROS/Gazebo/项目环境，也支持从任意目录启动。
    if os.environ.get("RL_VISUALIZE_ENV_READY") != "1":
        os.execvpe("bash", ["bash", "-c",
            'set -e; source /opt/ros/humble/setup.bash; '
            'source /usr/share/gazebo/setup.bash; source "$1/install/setup.bash"; '
            'export RL_VISUALIZE_ENV_READY=1; shift; exec "$@"',
            "visualize", str(ROOT), sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            os.environ)
    sys.path.insert(0, str(ROOT))
    from my_gazebo_launch.world_config import load_world_settings
    from rl_training.common import config
    from rl_training.common.evaluation import evaluate_model
    from rl_training.common.model_io import load_model
    from rl_training.common.observations import observation_specification
    from rl_training.privileged.observation import GroundTruthMap
    import torch

    if not WORLDS:
        raise ValueError("WORLDS 至少需要一张地图。")
    if any(type(v) is not int or v <= 0 for v in (EPISODES_PER_MAP, MAX_STEPS_PER_EPISODE)):
        raise ValueError("回合数和最大步数必须是正整数。")
    if any(not math.isfinite(v) or v < 0 for v in (POSITION_JITTER_M, YAW_JITTER_DEG)):
        raise ValueError("出生点扰动必须是非负有限数。")
    model_path = MODEL.expanduser().resolve(strict=True)
    saved = json.loads((model_path.parent / "run_config.json").read_text())
    if saved.get("observation_mode") != "privileged":
        raise ValueError("请指定 privileged 模型。")
    header = model_path.parent / "vehicle_config.h"
    vehicle = config.vehicle_config(header)
    if saved["vehicle"] != asdict(vehicle):
        raise ValueError("模型记录与配套车辆配置不一致。")
    spec = observation_specification("privileged", vehicle)
    model = load_model(model_path, device=config.DEVICE)
    if (saved.get("observation_spec", spec) != spec
            or model.observation_space.shape != (spec["dimension"],)
            or model.action_space.n != vehicle.label_count):
        raise ValueError("模型观测/动作空间与当前配置不一致。")
    settings = replace(config.EnvironmentSettings(**saved.get("environment", {})),
                       max_steps=MAX_STEPS_PER_EPISODE).validate()
    worlds = [load_world_settings(path) for path in WORLDS]
    for world in worlds:
        GroundTruthMap(world.world)
        print(f"地图：{world.world}；评估出生点 {len(world.eval_spawn_poses)} 个", flush=True)
    print(f"模型：{model_path}；每张地图 {EPISODES_PER_MAP} 回合；"
          f"每回合最多 {MAX_STEPS_PER_EPISODE * settings.step_sim_sec:g} 仿真秒", flush=True)
    if options.check:
        print("配置检查通过；未启动仿真。")
        return
    torch.set_num_threads(config.TORCH_THREADS)
    for index, world in enumerate(worlds, 1):
        print(f"观看地图 {index}/{len(worlds)}：{world.world.name}", flush=True)
        evaluate_model(argparse.Namespace(
            evaluate=model_path, config_header=header, world=str(world.world),
            eval_spawns=world.eval_spawn_poses, workspace_bounds=world.workspace_bounds,
            output_dir=OUTPUT_DIR, gui=True, eval_episodes=EPISODES_PER_MAP,
            episode_steps=MAX_STEPS_PER_EPISODE, eval_jitter_m=POSITION_JITTER_M,
            eval_jitter_deg=YAW_JITTER_DEG, seed=SEED,
            eval_domain=EVAL_ROS_DOMAIN, eval_port=EVAL_GAZEBO_PORT))


if __name__ == "__main__":
    main()
