"""Repeatable multi-start evaluations and trajectories; no training-data export."""

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
from io import BytesIO
import json
import math
import time
from uuid import uuid4
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import gymnasium as gym
import numpy as np
from my_gazebo_launch.world_config import resolve_world

from . import config
from .paths import training_results_dir
from .rl_env import AicarGazeboEnv
from .progress_state import LiveState
from .observations import observation_specification
from .model_io import load_model


class EvaluationScenarios(gym.Wrapper):
    def __init__(self, env, poses):
        super().__init__(env)
        self.poses = poses
        self.index = 0

    def reset_sequence(self):
        self.index = 0

    def reset(self, **kwargs):
        self.unwrapped.spawn_pose = tuple(self.poses[self.index % len(self.poses)])
        self.index += 1
        return self.env.reset(**kwargs)


def world_polygons(world):
    path = resolve_world(world)
    root = ET.parse(path).getroot()

    def transform(element):
        p = list(map(float, element.findtext("pose", "0 0 0 0 0 0").split()))
        c, s = math.cos(p[5]), math.sin(p[5])
        return np.array([[c, -s, p[0]], [s, c, p[1]], [0, 0, 1]])

    polygons = []
    for model in root.findall("./world/model"):
        for link in model.findall("link"):
            for collision in link.findall("collision"):
                box = collision.find("geometry/box/size")
                if box is None:
                    continue
                w, h, _ = map(float, box.text.split())
                corners = np.array([[-w/2, -h/2, 1], [w/2, -h/2, 1],
                                    [w/2, h/2, 1], [-w/2, h/2, 1]]).T
                matrix = transform(model) @ transform(link) @ transform(collision)
                polygons.append((matrix @ corners)[:2].T)
    return polygons


def save_trajectory_plot(path, trajectories, episodes, world, settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, (axis, speed_axis) = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    for polygon in world_polygons(world):
        axis.add_patch(Polygon(polygon, color="#687787", alpha=.75))
    for index, (trace, episode) in enumerate(zip(trajectories, episodes)):
        data = np.asarray(trace)
        line, = axis.plot(data[:, 1], data[:, 2], linewidth=1.1, label=f"{index+1}: {episode['reason']}")
        axis.plot(data[0, 1], data[0, 2], "o", color=line.get_color(), markersize=4)
        speed_axis.plot(data[:, 0], data[:, 4], linewidth=.7, alpha=.7, color=line.get_color())
    if settings.workspace_bounds:
        xmin, xmax, ymin, ymax = settings.workspace_bounds
        axis.set_xlim(xmin - .1, xmax + .1)
        axis.set_ylim(ymin - .1, ymax + .1)
    axis.set_aspect("equal")
    axis.set_xlabel("World x (m)"); axis.set_ylabel("World y (m)")
    axis.set_title("Independent deterministic policy trajectories")
    axis.legend(fontsize=7, loc="upper right")
    speed_axis.set_xlabel("Simulation time (s)"); speed_axis.set_ylabel("Forward speed (m/s)")
    speed_axis.set_title("Measured speed, including every turn")
    for ax in (axis, speed_axis):
        ax.grid(alpha=.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def evaluate_model(args):
    if args.eval_episodes <= 0:
        raise ValueError("eval-episodes 必须大于零。")
    if not all(math.isfinite(v) and v >= 0 for v in (args.eval_jitter_m, args.eval_jitter_deg)):
        raise ValueError("起点/朝向扰动必须为非负有限值。")
    vehicle = config.vehicle_config(args.config_header)
    saved_path = args.evaluate.parent / "run_config.json"
    saved = {}
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        if saved["vehicle"] != asdict(vehicle):
            raise ValueError("模型车辆配置与当前共享头文件不同。")
    else:
        print("旧模型没有共享配置记录，本次仅作基线对照。", flush=True)
    # Capture the bytes once: best_model.zip may be replaced by a concurrent
    # training process. The report must identify the weights actually tested.
    model_bytes = args.evaluate.read_bytes()
    model_digest = hashlib.sha256(model_bytes).hexdigest()
    model = load_model(BytesIO(model_bytes), device=config.DEVICE)
    mode = saved.get("observation_mode", "lidar")
    spec = observation_specification(mode, vehicle)
    if saved.get("observation_spec", spec) != spec:
        raise ValueError("模型观测配置与当前距离场/雷达定义不一致。")
    if model.observation_space.shape != (spec["dimension"],) or model.action_space.n != vehicle.label_count:
        raise ValueError("模型输入/动作空间与共享配置不同。")
    settings = replace(config.EnvironmentSettings(**saved.get("environment", {})), max_steps=args.episode_steps,
                       workspace_bounds=args.workspace_bounds).validate()
    directory = args.output_dir / f"evaluation_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    env = None
    episodes, traces = [], []
    random = np.random.default_rng(args.seed)
    progress = LiveState(directory / "evaluation_status.json", model_name=args.evaluate.name,
                         model_run=args.evaluate.parent.resolve().name, model_sha256=model_digest,
                         world=args.world, vehicle=asdict(vehicle), settings=asdict(settings),
                         total_episodes=args.eval_episodes, pose_jitter_m=args.eval_jitter_m,
                         pose_jitter_deg=args.eval_jitter_deg, observation_mode=mode)
    progress.update(force=True, status="running", episodes=[])
    try:
        env = AicarGazeboEnv(role="eval", domain_id=args.eval_domain, port=args.eval_port,
                             header=args.config_header, world=args.world, seed=args.seed,
                             spawn_pose=args.eval_spawns[0],
                             log_dir=directory, settings=settings, gui=args.gui,
                             reward_settings=config.RewardSettings(**saved.get("reward", {})),
                             real_time_factor=1.0 if args.gui else 0.0, observation_mode=mode)
        if args.gui:
            # The GUI can still publish its initial pause/control messages after
            # the model is spawned. Let it settle before the first paused step.
            time.sleep(config.GUI_STARTUP_SETTLE_SEC)
        for index in range(args.eval_episodes):
            x, y, z, yaw = args.eval_spawns[index % len(args.eval_spawns)]
            env.spawn_pose = (
                x + random.uniform(-args.eval_jitter_m, args.eval_jitter_m),
                y + random.uniform(-args.eval_jitter_m, args.eval_jitter_m), z,
                yaw + math.radians(random.uniform(-args.eval_jitter_deg, args.eval_jitter_deg)))
            obs, initial = env.reset(seed=args.seed + index)
            episode_started = time.monotonic()
            trace = [[0., *initial["position"], initial["yaw"], 0., vehicle.straight_label, 0.]]
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                trace.append([info["episode_sim_time_sec"], *info["position"], info["yaw"],
                              info["linear_velocity"], int(action), reward])
                if terminated or truncated:
                    break
            entry = {
                "scenario": index % len(args.eval_spawns), "spawn_pose": env.spawn_pose,
                "reason": info["termination_reason"], "steps": info["step"],
                "reward": info["episode_return"], "distance_m": info["episode_distance_m"],
                "sim_duration_sec": info["episode_sim_time_sec"],
                "wall_duration_sec": time.monotonic() - episode_started,
                "mean_speed_mps": info["episode_mean_speed_mps"],
                "speed_compliance": info["speed_compliance"], "success": info["is_success"],
                "collided": info["collided"], "stuck": info["stuck"], "looping": info["looping"],
                "slow": info["slow"], "out_of_bounds": info["out_of_bounds"],
            }
            episodes.append(entry); traces.append(trace)
            progress.update(force=True, status="running", episodes=episodes)
            print(f"评估 {index+1}/{args.eval_episodes}: {entry['reason']}, "
                  f"{entry['sim_duration_sec']:.1f}s, {entry['distance_m']:.2f}m, "
                  f"速度 {entry['mean_speed_mps']:.4f}m/s", flush=True)
    except BaseException as error:
        progress.update(force=True,
                        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error_type=type(error).__name__, episodes=episodes)
        raise
    finally:
        if env is not None:
            env.close()
    reasons = {reason: sum(e["reason"] == reason for e in episodes)
               for reason in ("time_limit", "collision", "stuck", "looping", "slow", "out_of_bounds")}
    report = {
        "observation_mode": mode, "observation_spec": spec,
        "gui": args.gui,
        "lidar_sensor_enabled": mode == "lidar",
        "world_sha256": env.ground_truth_map.sha256 if env.ground_truth_map else None,
        "model": str(args.evaluate.resolve()), "model_sha256": model_digest,
        "world": args.world, "seed": args.seed, "vehicle": asdict(vehicle), "settings": asdict(settings),
        "pose_jitter_m": args.eval_jitter_m, "pose_jitter_deg": args.eval_jitter_deg,
        "episodes": episodes, "end_reasons": reasons,
        "success_rate": sum(e["success"] for e in episodes) / len(episodes),
        "total_distance_m": sum(e["distance_m"] for e in episodes),
        "total_sim_seconds": sum(e["sim_duration_sec"] for e in episodes),
        "passed": all(e["success"] and e["speed_compliance"] >= .95 for e in episodes),
    }
    try:
        (directory / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        np.savez_compressed(directory / "trajectories.npz", **{f"episode_{i}": np.asarray(t) for i, t in enumerate(traces)})
        save_trajectory_plot(training_results_dir(directory) / "trajectories.png", traces, episodes, args.world, settings)
    except BaseException as error:
        progress.update(force=True, status="failed", error_type=type(error).__name__, episodes=episodes)
        raise
    progress.update(force=True, status="complete", episodes=episodes, passed=report["passed"])
    print(f"评估完成：{directory}；结束原因 {reasons}；全部通过={report['passed']}", flush=True)
    return report, directory
