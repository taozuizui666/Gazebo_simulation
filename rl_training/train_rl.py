#!/usr/bin/env python3
"""PPO cruising with fixed-speed actions, physical failure checks and evaluation."""

import argparse
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ament_index_python.packages import get_package_share_directory
from my_gazebo_launch.config import DEFAULT_URDF

from rl_training import config
from rl_training.evaluation import EvaluationScenarios, evaluate_model
from rl_training.policy import LidarFeatures
from rl_training.rl_env import AicarGazeboEnv
from rl_training.vector_env import CruiseSubprocVecEnv
from rl_training.progress_config import LOCAL_UPDATE_SEC
from rl_training.progress_state import LiveState
from rl_training.privileged import observation_specification, GroundTruthMap


def write_metadata(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot_sources(directory, world):
    root = Path(__file__).resolve().parents[1]
    target = directory / "source_snapshot"
    manifest = {}
    for package in ("rl_training", "my_gazebo_launch", "launch"):
        for source in (root / package).rglob("*"):
            if (source.suffix not in (".py", ".cpp", ".h") or "__pycache__" in source.parts
                    or source.name == "publish_progress.py"):
                continue
            relative = source.relative_to(root)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    share = Path(get_package_share_directory("my_gazebo_launch"))
    world_path = Path(world)
    if not world_path.is_absolute():
        world_path = share / world_path
    for filename, source in (("world.sdf", world_path), ("robot.urdf", share / DEFAULT_URDF)):
        relative = Path("resources") / filename
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / relative)
        manifest[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_metadata(target / "manifest.json", manifest)


class CruiseMetricsCallback(BaseCallback):
    """Save driving outcomes locally and stop cleanly when the wall-time budget ends."""

    def __init__(self, directory, max_seconds, start_steps=0):
        super().__init__()
        self.episodes = deque(maxlen=100)
        self.directory = directory
        self.max_seconds = max_seconds
        self.start_steps = start_steps
        self.budget_reached = False
        self.stop_requested = False
        self.started = self.last_status = time.monotonic()
        self.deadline = time.time() + max_seconds
        self.progress = LiveState(directory / "training_status.json")
        self.phase = "collecting"

    def snapshot(self, status="training", steps=None, force=True):
        means = {k: float(np.mean([e[k] for e in self.episodes]))
                 for k in self.episodes[0]} if self.episodes else {}
        elapsed = time.monotonic() - self.started
        steps = self.num_timesteps if steps is None else steps
        self.progress.update(force=force, status=status, phase=self.phase,
                             training_timesteps=steps, elapsed_wall_sec=elapsed,
                             time_budget_sec=self.max_seconds,
                             remaining_wall_sec=max(0., self.max_seconds - elapsed),
                             training_deadline=datetime.fromtimestamp(
                                 self.deadline, ZoneInfo("Asia/Shanghai")).isoformat(),
                             steps_per_second=(steps - self.start_steps) / max(elapsed, 1e-6),
                             recent_episodes=len(self.episodes), metrics=means)

    def _on_training_start(self):
        self.snapshot()

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes.append({
                    "collision_rate": float(info["collided"]),
                    "stuck_rate": float(info["stuck"]),
                    "loop_rate": float(info["looping"]),
                    "slow_rate": float(info["slow"]),
                    "out_of_bounds_rate": float(info["out_of_bounds"]),
                    "completion_rate": float(info["is_success"]),
                    "distance_m": info["episode_distance_m"],
                    "sim_duration_sec": info["episode_sim_time_sec"],
                    "mean_speed_mps": info["episode_mean_speed_mps"],
                    "speed_compliance": info["speed_compliance"],
                })
        elapsed = time.monotonic() - self.started
        if time.monotonic() - self.last_status >= LOCAL_UPDATE_SEC:
            self.snapshot()
            self.last_status = time.monotonic()
        self.budget_reached = elapsed >= self.max_seconds
        self.stop_requested = (self.directory / "STOP").exists()
        return not (self.budget_reached or self.stop_requested)

    def _on_rollout_end(self):
        if self.episodes:
            for key in self.episodes[0]:
                self.logger.record(f"cruise/{key}", float(np.mean([e[key] for e in self.episodes])))


class CruiseEvalCallback(EvalCallback):
    def __init__(self, *args, progress=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress = progress

    def _on_step(self):
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluating:
            self.eval_env.venv.envs[0].env.reset_sequence()
            if self.progress:
                self.progress.phase = "periodic_evaluation"
                self.progress.snapshot(steps=self.num_timesteps)
        previous_best = self.best_mean_reward
        try:
            continuing = super()._on_step()
        finally:
            if evaluating and self.progress:
                self.progress.phase = "collecting"
                self.progress.snapshot(steps=self.num_timesteps)
        if self.best_mean_reward > previous_best:
            self.training_env.save(Path(self.best_model_save_path) / "best_vecnormalize.pkl")
        return continuing


def make_env(role, args, settings, directory, rank=0):
    torch.set_num_threads(1)
    domain = args.train_domain + rank * 2 if role == "train" else args.eval_domain
    port = args.train_port + rank * 2 if role == "train" else args.eval_port
    env_dir = directory / (f"worker_{rank}" if role == "train" else "evaluation_server")
    env = AicarGazeboEnv(
        role=role, domain_id=domain, port=port,
        header=args.config_header, world=args.world,
        spawn_pose=config.TRAIN_SPAWN_POSES[rank % len(config.TRAIN_SPAWN_POSES)]
                   if role == "train" else config.EVAL_SPAWN_POSES[0],
        seed=args.seed + rank if role == "train" else args.seed + 1000,
        log_dir=env_dir, settings=settings, observation_mode=args.observation_mode)
    if role == "eval":
        env = EvaluationScenarios(env)
    return Monitor(env, filename=str(env_dir / "episodes"), info_keywords=(
        "termination_reason", "episode_distance_m", "episode_sim_time_sec",
        "episode_mean_speed_mps", "speed_compliance", "is_success"))


def run_training(args):
    if args.num_envs < 1:
        raise ValueError("num-envs 必须大于零。")
    domains = [args.train_domain + 2*i for i in range(args.num_envs)]
    ports = [args.train_port + 2*i for i in range(args.num_envs)]
    if args.eval_domain in domains or args.eval_port in ports:
        raise ValueError("训练与评估必须使用不同 ROS domain 和 Gazebo port。")
    if any(d < 0 or d > 101 for d in domains + [args.eval_domain]):
        raise ValueError("为避免 DDS 端口冲突，本程序使用 ROS domain 0～101。")
    if min(args.timesteps, args.eval_episodes, args.eval_freq, args.hours) <= 0:
        raise ValueError("训练步数、评估集数、评估间隔和小时预算必须大于零。")
    if args.rollout_steps < 2 or (args.rollout_steps * args.num_envs) % config.BATCH_SIZE:
        raise ValueError("rollout-steps × num-envs 必须是 BATCH_SIZE 的整数倍。")
    settings = replace(config.EnvironmentSettings(), max_steps=args.episode_steps).validate()
    vehicle = config.vehicle_config(args.config_header)
    prefix = "PPO_privileged" if args.observation_mode == "privileged" else "PPO_cruise"
    run_id = f"{prefix}_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    directory = args.output_dir / run_id
    directory.mkdir(parents=True)
    metadata = {
        "schema_version": 3, "task": "continuous_obstacle_avoiding_cruise", "run_id": run_id,
        "observation_mode": args.observation_mode,
        "observation_spec": observation_specification(args.observation_mode, vehicle),
        "status": "starting", "requested_timesteps": args.timesteps,
        "config_header": str(args.config_header.resolve()), "vehicle": asdict(vehicle),
        "environment": asdict(settings), "reward": asdict(config.RewardSettings()),
        "training_spawn_poses": config.TRAIN_SPAWN_POSES,
        "evaluation_spawn_poses": config.EVAL_SPAWN_POSES,
        "ppo": {name: getattr(config, name) for name in (
            "LEARNING_RATE", "BATCH_SIZE", "N_EPOCHS", "GAMMA", "GAE_LAMBDA", "CLIP_RANGE",
            "ENTROPY_COEFF", "VALUE_COEFF", "MAX_GRAD_NORM", "POLICY_NET_ARCH")},
        "observation_preprocessing": ("static world signed-distance field and exact physics state; no lidar"
                                      if args.observation_mode == "privileged" else
                                      "canonical_mm / max_mm; policy sqrt; lidar only"),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.glob("*.py")},
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    if args.observation_mode == "privileged":
        geometry = GroundTruthMap(args.world)
        metadata["world_geometry"] = {"sha256": geometry.sha256, "static_boxes": len(geometry.centers)}
    shutil.copy2(args.config_header, directory / "vehicle_config.h")
    shutil.copy2(Path(config.__file__), directory / "training_config.py")
    snapshot_sources(directory, args.world)
    write_metadata(directory / "run_config.json", metadata)
    print(f"巡航训练：模式 {args.observation_mode}，标签 0～{vehicle.label_count-1}，"
          f"输入 {metadata['observation_spec']['dimension']} 维，"
          f"目标速度 {vehicle.forward_speed_mps} m/s", flush=True)
    print(f"{args.num_envs} 个独立训练仿真，domains={domains}；评估 domain={args.eval_domain}", flush=True)
    print(f"模型目录：{directory}", flush=True)
    training = evaluation = model = None
    budget = None
    error = None
    try:
        factories = [partial(make_env, "train", args, settings, directory, i) for i in range(args.num_envs)]
        vector = CruiseSubprocVecEnv(factories, start_method="spawn") if args.num_envs > 1 else DummyVecEnv(factories)
        training = vector  # still close the owned servers if normalizer/model loading fails
        if args.resume is not None:
            saved = json.loads((args.resume.parent / "run_config.json").read_text())
            if saved.get("schema_version") not in (2, 3):
                vector.close()
                raise ValueError("续训要求本版巡航模型。")
            if saved.get("observation_mode", "lidar") != args.observation_mode:
                raise ValueError("雷达策略与特权教师的输入不同，不能直接混用 --resume。")
            if saved.get("observation_spec", metadata["observation_spec"]) != metadata["observation_spec"]:
                raise ValueError("续训要求相同观测维度、网格范围和归一化定义。")
            if saved["vehicle"] != asdict(vehicle):
                if not args.resume_allow_vehicle_change:
                    vector.close()
                    raise ValueError("续训要求相同车辆配置；确需调速等变更时用 "
                                     "--resume-allow-vehicle-change 微调。")
                changed = {k: (saved["vehicle"].get(k), v) for k, v in asdict(vehicle).items()
                           if saved["vehicle"].get(k) != v}
                print(f"警告：车辆配置相对模型记录已变更 {changed}，作为微调初始化继续训练。",
                      flush=True)
            if args.resume.name == "best_model.zip":
                normalization_path = args.resume.parent / "best_vecnormalize.pkl"
            elif args.resume.stem.startswith("rl_policy_") and args.resume.stem.endswith("_steps"):
                suffix = args.resume.stem.removeprefix("rl_policy_")
                normalization_path = args.resume.parent / f"rl_policy_vecnormalize_{suffix}.pkl"
            else:
                normalization_path = args.resume.parent / "vecnormalize.pkl"
            training = VecNormalize.load(normalization_path, vector)
            training.training, training.norm_reward = True, True
        else:
            training = VecNormalize(vector, norm_obs=False, norm_reward=True, gamma=config.GAMMA)
        evaluation = VecNormalize(
            DummyVecEnv([partial(make_env, "eval", args, settings, directory)]),
            training=False, norm_obs=False, norm_reward=False, gamma=config.GAMMA)
        torch.set_num_threads(config.TORCH_THREADS)
        if args.resume is not None:
            model = PPO.load(args.resume, env=training, device=config.DEVICE,
                             tensorboard_log=str(args.log_dir), n_steps=args.rollout_steps)
            model.set_random_seed(args.seed)
        else:
            policy_kwargs = {"net_arch": config.POLICY_NET_ARCH}
            if args.observation_mode == "lidar":
                policy_kwargs["features_extractor_class"] = LidarFeatures
            model = PPO(
                "MlpPolicy", training, learning_rate=config.LEARNING_RATE,
                n_steps=args.rollout_steps, batch_size=config.BATCH_SIZE, n_epochs=config.N_EPOCHS,
                gamma=config.GAMMA, gae_lambda=config.GAE_LAMBDA, clip_range=config.CLIP_RANGE,
                ent_coef=config.ENTROPY_COEFF, vf_coef=config.VALUE_COEFF,
                max_grad_norm=config.MAX_GRAD_NORM,
                policy_kwargs=policy_kwargs,
                seed=args.seed, device=config.DEVICE, verbose=1, tensorboard_log=str(args.log_dir))
        budget = CruiseMetricsCallback(directory, args.hours * 3600, model.num_timesteps)
        evaluation_callback = CruiseEvalCallback(
            evaluation, progress=budget, best_model_save_path=str(directory),
            log_path=str(directory / "evaluation"),
            eval_freq=max(1, args.eval_freq // args.num_envs),
            n_eval_episodes=args.eval_episodes, deterministic=True)
        callbacks = [
            budget,
            evaluation_callback,
            CheckpointCallback(save_freq=max(1, config.SAVE_FREQ // args.num_envs),
                               save_path=str(directory), name_prefix="rl_policy", save_vecnormalize=True),
        ]
        metadata["status"] = "training"
        metadata["start_timesteps"] = model.num_timesteps
        metadata["target_total_timesteps"] = model.num_timesteps + args.timesteps
        write_metadata(directory / "run_config.json", metadata)
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    tb_log_name=run_id, progress_bar=sys.stdout.isatty(),
                    reset_num_timesteps=args.resume is None)
        metadata["status"] = ("time_budget_reached" if budget.budget_reached else
                              "stopped" if budget.stop_requested else "complete")
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        print("训练已中断，将保存当前模型。", flush=True)
    except Exception as exception:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exception).__name__}: {exception}"
        error = exception
    finally:
        try:
            if model is not None:
                filename = "rl_policy_final" if metadata["status"] in ("complete", "time_budget_reached", "stopped") else "rl_policy_interrupted"
                model.save(directory / filename)
                training.save(directory / "vecnormalize.pkl")
                metadata["saved_model"] = filename + ".zip"
                metadata["training_timesteps"] = model.num_timesteps
            metadata["ended_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
            write_metadata(directory / "run_config.json", metadata)
            if budget is not None:
                budget.phase = "finished"
                budget.snapshot(status=metadata["status"], steps=metadata.get("training_timesteps", 0))
        finally:
            try:
                if training is not None:
                    training.close()
            finally:
                if evaluation is not None:
                    evaluation.close()
    print(f"训练状态：{metadata['status']}；结果：{directory}", flush=True)
    if error is not None:
        raise error
    return directory


def main():
    parser = argparse.ArgumentParser(description="持续避障巡航 PPO；自动管理独立训练/评估仿真")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--eval-episodes", type=int, default=config.EVAL_EPISODES)
    parser.add_argument("--eval-freq", type=int, default=config.EVAL_FREQ)
    parser.add_argument("--rollout-steps", type=int, default=config.N_STEPS)
    parser.add_argument("--episode-steps", type=int, default=config.MAX_STEPS_PER_EPISODE)
    parser.add_argument("--num-envs", type=int, default=config.NUM_TRAIN_ENVS)
    parser.add_argument("--hours", type=float, default=config.MAX_TRAINING_HOURS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--config-header", type=Path, default=config.CONFIG_HEADER)
    parser.add_argument("--world", default=config.WORLD)
    parser.add_argument("--observation-mode", choices=("privileged", "lidar"),
                        help="新训练默认特权地图真值；评估/续训默认自动读取模型记录的模式")
    parser.add_argument("--train-domain", type=int, default=config.TRAIN_ROS_DOMAIN)
    parser.add_argument("--eval-domain", type=int, default=config.EVAL_ROS_DOMAIN)
    parser.add_argument("--train-port", type=int, default=config.TRAIN_GAZEBO_PORT)
    parser.add_argument("--eval-port", type=int, default=config.EVAL_GAZEBO_PORT)
    parser.add_argument("--output-dir", type=Path, default=config.POLICY_DIR)
    parser.add_argument("--log-dir", type=Path, default=config.LOG_DIR)
    parser.add_argument("--evaluate", type=Path, help="独立评估并保存轨迹和速度报告")
    parser.add_argument("--eval-jitter-m", type=float, default=0.0, help="评估起点 x/y 的均匀扰动上限")
    parser.add_argument("--eval-jitter-deg", type=float, default=0.0, help="评估初始朝向的均匀扰动上限")
    parser.add_argument("--gui", action="store_true", help="评估时打开 Gazebo，以正常物理时间观看策略")
    parser.add_argument("--resume", type=Path, help="从本版模型及同目录 vecnormalize.pkl 续训")
    parser.add_argument("--resume-allow-vehicle-change", action="store_true",
                        help="允许车辆参数（如速度）与模型记录不同，将旧模型作为初始化微调")
    args = parser.parse_args()
    if args.evaluate is not None and args.resume is not None:
        parser.error("--evaluate 和 --resume 不能同时使用")
    if args.gui and args.evaluate is None:
        parser.error("--gui 用于 --evaluate 可视化验证")
    model_path = args.evaluate or args.resume
    if model_path is not None:
        metadata_path = model_path.parent / "run_config.json"
        saved = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        saved_mode = saved.get("observation_mode", "lidar")
        if args.observation_mode is not None and args.observation_mode != saved_mode:
            parser.error("观测模式与已保存模型不符；特权教师应新建训练，不能直接续用雷达网络。")
        args.observation_mode = saved_mode
    else:
        args.observation_mode = args.observation_mode or config.OBSERVATION_MODE
    torch.set_num_threads(config.TORCH_THREADS)
    if args.evaluate is not None:
        evaluate_model(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
