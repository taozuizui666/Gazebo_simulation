#!/usr/bin/env python3
"""多地图、随机出生点和扰动联合训练一个特权 PPO 模型。"""
from pathlib import Path
import argparse
import sys

# ===================== 可修改配置区 =====================
ROOT = Path(__file__).resolve().parents[2]
WORLD_DIR = ROOT / "assets/worlds/simple"
TRAIN_WORLDS = [WORLD_DIR / name for name in (
    "train_world3.sdf", "rounded_rectangle_001.sdf", "rounded_rectangle_003.sdf",
    "ellipse_002.sdf", "ellipse_004.sdf")]
# 仅用于模型选择，不参与梯度更新；最终泛化验收应另用未参与选模的地图。
EVAL_WORLDS = [WORLD_DIR / "rounded_rectangle_005.sdf", WORLD_DIR / "ellipse_006.sdf"]
OUTPUT_DIR = ROOT / "rl_models/privileged/policies"
LOG_DIR = ROOT / "rl_models/privileged/logs"
ENABLE_PROGRESS = True             # 自动启动本地手机进度页；False 不管理发布器
PUBLISH_TO_GITHUB = True            # ENABLE_PROGRESS=True 时，同时同步 GitHub
CONFIG_HEADER = ROOT / "my_gazebo_launch/vehicle_config.h"  # 速度/轮距仍以此为准
AUTO_RESUME = True                 # 优先接着 OUTPUT_DIR 中最近的多地图训练
RESUME_MODEL = ROOT / "rl_models/privileged/policies/latest_validated/model.zip"                 # 无自动续训历史时使用；关闭 AUTO_RESUME 可固定指定模型
RESUME_ALLOW_VEHICLE_CHANGE = False
# 训练正常结束后独立验收；Ctrl+C/STOP 不自动启动验收。
AUTO_VALIDATE = True
VALIDATION_WORLDS = [ROOT / "assets/worlds/validation" / name for name in (
    "validation_rounded_rectangle_001.sdf", "validation_ellipse_002.sdf")]
VALIDATION_MODEL_NAME = "best_model.zip"  # 尚无 best 时使用本次 final
VALIDATION_EPISODES_PER_MAP = 8
VALIDATION_MIN_SCENARIOS = 8        # 至少这么多个不同的初始位置/朝向
VALIDATION_EPISODE_SECONDS = 300.0  # 每个回合的仿真时长；不是整轮验收时长
VALIDATION_MIN_SPEED_COMPLIANCE = 0.95  # 每回合至少 95% 时间在目标速度容差内
VALIDATION_POSITION_JITTER_M = 0.03
VALIDATION_YAW_JITTER_DEG = 10.0
VALIDATION_SEED = 20261012
# 所有回合必须零碰撞、卡住、局部绕圈、持续低速和越界才更新 latest_validated。
# 验收在训练预算结束后进行，会额外耗时；不更新策略权重。
NUM_TRAIN_ENVS = 6                  # 独立 Gazebo 实例，共同更新一个 PPO
STEP_SIM_TIME_SEC = 0.2             # 一次动作持续的仿真秒数，不是加速倍率
EPISODES_PER_MAP = 10               # 每个 worker 完成多少回合后换地图
MAX_STEPS_PER_EPISODE = 1000         # 默认每回合最多 200 仿真秒
TRAIN_POSITION_JITTER_M = 0.03      # x/y 分别均匀扰动 ±此值
TRAIN_YAW_JITTER_DEG = 10.0
EVAL_POSITION_JITTER_M = 0.03
EVAL_YAW_JITTER_DEG = 10.0
SPAWN_MARGIN_M = 0.01               # 车体保守半径之外的额外净空
SPAWN_ATTEMPTS = 50                 # 采样失败明确报错，不无限重试
SEED = 42
TOTAL_TIMESTEPS = 10_000_000         # 所有 worker 合计采样步数
MAX_TRAINING_HOURS = 4.0/6.0            # 与总步数任一先达到就保存退出
EVAL_FREQ = 20_000                  # 合计采样步数；评估时暂不训练
EVAL_EPISODES_PER_MAP = 8           # 按地图 JSON 的评估出生点循环，固定种子
SAVE_FREQ = 20_000
N_STEPS = 512                      # 每个 worker 每次 PPO 更新前采样的步数
BATCH_SIZE = 256                   # 必须整除 NUM_TRAIN_ENVS × N_STEPS
LEARNING_RATE = 3e-4
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEFF = 0.005
VALUE_COEFF = 0.5
MAX_GRAD_NORM = 0.5
POLICY_NET_ARCH = [128, 128]
DEVICE = "cpu"
TORCH_THREADS = 2
TRAIN_ROS_DOMAIN = 81               # worker i 使用基础值 + 2*i
EVAL_ROS_DOMAIN = 82
TRAIN_GAZEBO_PORT = 11451
EVAL_GAZEBO_PORT = 11452
# 失败条件/速度要求；保持持续前进，禁止卡住和小范围反复绕圈。
STUCK_WINDOW_SEC = 3.0
STUCK_MIN_TRAVEL_M = 0.02
MIN_CRUISE_SPEED_RATIO = 0.75
SPEED_TOLERANCE_RATIO = 0.20
LOOP_WINDOW_SEC = 12.0
LOOP_MIN_DURATION_SEC = 6.0
LOOP_MIN_TURN_RAD = 5.5
LOOP_MAX_SPAN_M = 0.65
# 奖励配置：键名对应 RewardSettings；所有值会写入本次模型配置。
REWARD = dict(distance_per_m=10.0, speed_error_per_sec=3.0,
              near_distance_mm=250.0, near_penalty_per_sec=0.5,
              steering_change_penalty=0.1, front_distance_mm=450.0,
              front_half_angle_deg=20.0, front_risk_per_sec=2.0, turn_per_sec=0.15,
              collision=-20.0, stuck=-20.0, loop=-20.0, slow=-20.0, out_of_bounds=-20.0)
# ===================== 配置区结束 =====================

sys.path.insert(0, str(ROOT))


def build_args():
    from my_gazebo_launch.world_config import load_world_settings
    first = load_world_settings(TRAIN_WORLDS[0])
    return argparse.Namespace(
        observation_mode="privileged", world=str(first.world),
        train_spawns=first.train_spawn_poses, eval_spawns=first.eval_spawn_poses,
        workspace_bounds=None, config_header=CONFIG_HEADER, output_dir=OUTPUT_DIR, log_dir=LOG_DIR,
        resume=RESUME_MODEL, resume_allow_vehicle_change=RESUME_ALLOW_VEHICLE_CHANGE,
        num_envs=NUM_TRAIN_ENVS, rollout_steps=N_STEPS, episode_steps=MAX_STEPS_PER_EPISODE,
        timesteps=TOTAL_TIMESTEPS, hours=MAX_TRAINING_HOURS, seed=SEED,
        eval_freq=EVAL_FREQ, eval_episodes=len(EVAL_WORLDS)*EVAL_EPISODES_PER_MAP,
        save_freq=SAVE_FREQ, device=DEVICE, torch_threads=TORCH_THREADS,
        train_domain=TRAIN_ROS_DOMAIN, eval_domain=EVAL_ROS_DOMAIN,
        train_port=TRAIN_GAZEBO_PORT, eval_port=EVAL_GAZEBO_PORT,
        environment=dict(step_sim_sec=STEP_SIM_TIME_SEC, stuck_window_sec=STUCK_WINDOW_SEC,
            stuck_min_travel_m=STUCK_MIN_TRAVEL_M, min_cruise_speed_ratio=MIN_CRUISE_SPEED_RATIO,
            speed_tolerance_ratio=SPEED_TOLERANCE_RATIO, loop_window_sec=LOOP_WINDOW_SEC,
            loop_min_duration_sec=LOOP_MIN_DURATION_SEC, loop_min_turn_rad=LOOP_MIN_TURN_RAD,
            loop_max_span_m=LOOP_MAX_SPAN_M), reward=REWARD.copy(),
        ppo={name: globals()[name] for name in (
            "LEARNING_RATE", "BATCH_SIZE", "N_EPOCHS", "GAMMA", "GAE_LAMBDA", "CLIP_RANGE",
            "ENTROPY_COEFF", "VALUE_COEFF", "MAX_GRAD_NORM", "POLICY_NET_ARCH")},
        validation=dict(enabled=AUTO_VALIDATE, worlds=[str(p.resolve()) for p in VALIDATION_WORLDS],
            model_name=VALIDATION_MODEL_NAME, episodes_per_map=VALIDATION_EPISODES_PER_MAP,
            min_scenarios=VALIDATION_MIN_SCENARIOS, episode_seconds=VALIDATION_EPISODE_SECONDS,
            min_speed_compliance=VALIDATION_MIN_SPEED_COMPLIANCE,
            jitter_m=VALIDATION_POSITION_JITTER_M, jitter_deg=VALIDATION_YAW_JITTER_DEG,
            seed=VALIDATION_SEED),
        multi_map=dict(train_worlds=[str(p.resolve()) for p in TRAIN_WORLDS],
            eval_worlds=[str(p.resolve()) for p in EVAL_WORLDS], episodes_per_map=EPISODES_PER_MAP,
            eval_episodes_per_map=EVAL_EPISODES_PER_MAP,
            train_jitter_m=TRAIN_POSITION_JITTER_M, train_jitter_deg=TRAIN_YAW_JITTER_DEG,
            eval_jitter_m=EVAL_POSITION_JITTER_M, eval_jitter_deg=EVAL_YAW_JITTER_DEG,
            spawn_margin_m=SPAWN_MARGIN_M, spawn_attempts=SPAWN_ATTEMPTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="检查地图/出生点/配置，不启动仿真或训练")
    options = parser.parse_args()
    from rl_training.common.train_rl import run_training, training_configuration
    args = build_args()
    from rl_training.privileged.auto_resume import resolve_resume, validate_resume
    args.resume = resolve_resume(args, AUTO_RESUME)
    validate_resume(args)
    _, _, _, vehicle, _ = training_configuration(args)
    from rl_training.privileged.auto_validate import validation_worlds, run_validation
    acceptance_worlds = validation_worlds(args)
    if acceptance_worlds:
        print("验收地图：" + ", ".join(str(w.world) for w in acceptance_worlds), flush=True)
    for role in ("train", "eval"):
        print(f"{role}: " + ", ".join(Path(w).name for w in args.multi_map[role + "_worlds"]), flush=True)
    print(f"目标速度 {vehicle.forward_speed_mps} m/s；"
          f"{NUM_TRAIN_ENVS} 个训练实例；控制周期 {STEP_SIM_TIME_SEC} 仿真秒", flush=True)
    if not options.check:
        from rl_training.monitoring.training_progress import training_progress
        with training_progress(ENABLE_PROGRESS, PUBLISH_TO_GITHUB):
            directory = run_training(args)
            run_validation(args, directory)


if __name__ == "__main__":
    main()
