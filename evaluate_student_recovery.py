#!/usr/bin/env python3
"""Run normal/yaw/near-wall recovery episodes for a frozen LiDAR student model."""
from pathlib import Path
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
import os
import statistics
import sys
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DISSERTATION = ROOT.parent
if os.environ.get("R2_EVAL_ENV_READY") != "1":
    os.execvpe("bash", ["bash", "-c",
        'set -e; source /opt/ros/humble/setup.bash; '
        'source /usr/share/gazebo/setup.bash; source "$1/install/setup.bash"; '
        'export R2_EVAL_ENV_READY=1; shift; exec "$@"',
        "evaluate_student_recovery", str(ROOT), sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        os.environ)

import numpy as np

sys.path.insert(0, str(ROOT))
from my_gazebo_launch.config import load_config
from my_gazebo_launch.world_config import load_world_settings
from rl_training.common import config
from rl_training.common.rl_env import AicarGazeboEnv
from rl_training.common.forest_evaluation import corrupt_scan
from rl_training.privileged.observation import GroundTruthMap
from run_random_forest import load_model

CONFIG_HEADER = ROOT / "my_gazebo_launch/vehicle_config.h"
CATEGORIES = ("normal", "yaw", "near_wall")


def resolve_path(value, base):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for candidate in (base / path, DISSERTATION / path, ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    return (base / path).resolve()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summary_for(episodes):
    result = {}
    groups = {}
    for episode in episodes:
        groups.setdefault((episode["scenario"], episode["category"]), []).append(episode)
    for (scenario, category), rows in groups.items():
        n = len(rows)
        success = sum(e["success"] for e in rows)
        recovered = sum(e["recovered"] for e in rows)
        recovered_success = sum(e["recovered_success"] for e in rows)
        recovery_times = [e["recovery_time_sec"] for e in rows if e["recovery_time_sec"] is not None]
        result[f"{scenario}/{category}"] = {
            "scenario": scenario, "category": category,
            "episodes": n, "success": success, "recovered": recovered,
            "recovered_success": recovered_success, "collisions": sum(e["collided"] for e in rows),
            "stuck": sum(e["stuck"] for e in rows), "looping": sum(e["looping"] for e in rows),
            "slow": sum(e["slow"] for e in rows), "out_of_bounds": sum(e["out_of_bounds"] for e in rows),
            "success_rate": success / n, "recovered_success_rate": recovered_success / n,
            "success_wilson95": wilson(success, n), "recovered_success_wilson95": wilson(recovered_success, n),
            "recovery_time_mean_sec": statistics.mean(recovery_times) if recovery_times else None,
            "recovery_time_median_sec": statistics.median(recovery_times) if recovery_times else None,
            "initial_clearance_min_m": min(e["initial_clearance_m"] for e in rows),
            "initial_clearance_mean_m": statistics.mean(e["initial_clearance_m"] for e in rows),
            "min_clearance_min_m": min(e["min_clearance_m"] for e in rows),
            "p05_clearance_mean_m": statistics.mean(e["p05_clearance_m"] for e in rows),
            "mean_action_switches": statistics.mean(e["action_switches"] for e in rows),
            "mean_distance_m": statistics.mean(e["distance_m"] for e in rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="software/ML_model/simulation_tree6/randomForest.joblib")
    parser.add_argument("--world", required=True, help="SDF；相对 Gazebo_simulation")
    parser.add_argument("--start-plan", required=True, help="student_starts 生成的 JSON")
    parser.add_argument("--category", action="append", choices=CATEGORIES, dest="categories")
    parser.add_argument("--scenario", action="append", choices=("normal", "dropout05", "dropout10", "sector"), dest="scenarios")
    parser.add_argument("--episodes-per-category", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--domain", type=int, default=94)
    parser.add_argument("--port", type=int, default=11554)
    parser.add_argument("--output-dir", default=str(ROOT / "results/student_recovery"))
    parser.add_argument("--recovery-window-sec", type=float, default=60.0)
    parser.add_argument("--recovery-hold-sec", type=float, default=10.0)
    parser.add_argument("--recovery-clearance-m", type=float, default=0.05)
    parser.add_argument("--recovery-speed-mps", type=float, default=0.15)
    parser.add_argument("--check", action="store_true", help="只检查模型/地图/起点，不启动仿真")
    args = parser.parse_args()
    categories = args.categories or list(CATEGORIES)
    scenarios = args.scenarios or ["normal"]
    if args.episodes_per_category < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise SystemExit("episodes-per-category 和 seconds 必须为正数。")
    if not all(math.isfinite(v) and v > 0 for v in (args.recovery_window_sec, args.recovery_hold_sec,
                                                     args.recovery_clearance_m, args.recovery_speed_mps)):
        raise SystemExit("恢复阈值必须为正有限数。")
    model_path = resolve_path(args.model, DISSERTATION)
    world_path = resolve_path(args.world, ROOT)
    plan_path = resolve_path(args.start_plan, DISSERTATION)
    vehicle = load_config(CONFIG_HEADER)
    classifier, indices = load_model(model_path, vehicle)
    world = load_world_settings(world_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("world_sha256") != GroundTruthMap(world.world).sha256:
        raise SystemExit("start_plan 与当前地图 SHA-256 不一致。")
    for category in categories:
        if len(plan["categories"].get(category, [])) < args.episodes_per_category:
            raise SystemExit(f"{category} 起点不足 {args.episodes_per_category} 个。")
    print(f"模型：{model_path}\n地图：{world.world}\n起点计划：{plan_path}\n"
          f"类别 {categories}，场景 {scenarios}；每格 {args.episodes_per_category} 回合，"
          f"{args.seconds:g} 秒/回合；恢复阈值 净空 {args.recovery_clearance_m:g} m、"
          f"速度 {args.recovery_speed_mps:g} m/s、窗口 {args.recovery_window_sec:g} s、"
          f"保持 {args.recovery_hold_sec:g} s", flush=True)
    if args.check:
        for category in categories:
            entries = plan["categories"][category][:args.episodes_per_category]
            print(f"{category}: {len(entries)} 个起点；净空 "
                  f"{min(e['clearance_m'] for e in entries):.3f}–{max(e['clearance_m'] for e in entries):.3f} m")
        print(f"计划回合：{len(categories) * len(scenarios) * args.episodes_per_category}；"
              "配置与试推理检查通过；未启动仿真。")
        return 0

    directory = Path(args.output_dir) / (datetime.now(ZoneInfo("Asia/Shanghai")).strftime("evaluation_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    model_sha = sha256(model_path)
    start_plan_sha = sha256(plan_path)
    report = {
        "schema_version": 1, "status": "running",
        "model": str(model_path), "model_sha256": model_sha,
        "world": str(world.world), "world_sha256": GroundTruthMap(world.world).sha256,
        "start_plan": str(plan_path), "start_plan_sha256": start_plan_sha,
        "vehicle": asdict(vehicle),
        "settings": {"seconds": args.seconds, "episodes_per_category": args.episodes_per_category,
                     "categories": categories, "scenarios": scenarios,
                     "domain": args.domain, "port": args.port, "seed": args.seed,
                     "recovery_window_sec": args.recovery_window_sec,
                     "recovery_hold_sec": args.recovery_hold_sec,
                     "recovery_clearance_m": args.recovery_clearance_m,
                     "recovery_speed_mps": args.recovery_speed_mps},
        "episodes": [], "summary": {},
    }
    started_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["started_at"] = started_at
    env = None
    try:
        step_seconds = max(config.EnvironmentSettings().step_sim_sec, 1 / vehicle.lidar_scan_hz + vehicle.sim_time_step_sec)
        settings = replace(config.EnvironmentSettings(),
                           max_steps=math.ceil(args.seconds / step_seconds),
                           workspace_bounds=world.workspace_bounds)
        env = AicarGazeboEnv(role="eval", domain_id=args.domain, port=args.port, header=CONFIG_HEADER,
                             world=world.world, spawn_pose=plan["categories"][categories[0]][0]["pose"],
                             seed=args.seed, log_dir=directory / world.world.stem, settings=settings,
                             gui=False, real_time_factor=0.0, observation_mode="lidar", capture_lidar=True)
        geometry = GroundTruthMap(world.world)
        active = np.array([i for i in range(vehicle.lidar_point_count)
                           if not vehicle.lidar_front_sector_only or i < vehicle.lidar_point_count // 4
                           or i > 3 * vehicle.lidar_point_count // 4])
        episode_number = 0
        for scenario in scenarios:
          for category in categories:
            for index, entry in enumerate(plan["categories"][category][:args.episodes_per_category]):
                episode_number += 1
                run_dir = directory / world.world.stem / scenario / category / f"episode_{index + 1:03d}"
                run_dir.mkdir(parents=True)
                env.spawn_pose = tuple(entry["pose"])
                seed = args.seed + episode_number - 1
                rng = np.random.default_rng(seed)
                obs, info = env.reset(seed=seed)
                initial_clearance, initial_front = geometry.safety_distances(dict(
                    x=info["position"][0], y=info["position"][1], yaw=info["yaw"]))
                steps = [{"time": 0.0, "x": info["position"][0], "y": info["position"][1],
                          "yaw": info["yaw"], "speed": 0.0, "action": -1,
                          "clearance_m": initial_clearance, "front_clearance_m": initial_front}]
                trajectory = [[0.0, info["position"][0], info["position"][1], info["yaw"], 0.0, -1,
                               initial_clearance, initial_front]]
                clearances = [initial_clearance]
                switches = 0
                previous_action = None
                hold_start = None
                recovery_time = None
                while True:
                    scan = corrupt_scan(np.asarray(info["lidar_mm"], dtype=np.float32), scenario, rng, active)
                    action = int(classifier.predict(scan[indices].reshape(1, -1))[0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    time_sec = float(info["episode_sim_time_sec"])
                    clearance, front = geometry.safety_distances(dict(
                        x=info["position"][0], y=info["position"][1], yaw=info["yaw"]))
                    clearances.append(clearance)
                    if previous_action is not None and action != previous_action:
                        switches += 1
                    previous_action = action
                    steps.append({"time": time_sec, "x": info["position"][0], "y": info["position"][1],
                                  "yaw": info["yaw"], "speed": info["linear_velocity"], "action": action,
                                  "clearance_m": clearance, "front_clearance_m": front,
                                  "collided": info["collided"]})
                    trajectory.append([time_sec, info["position"][0], info["position"][1], info["yaw"],
                                       info["linear_velocity"], action, clearance, front])
                    speed = max(0.0, float(info["linear_velocity"]))
                    if (recovery_time is None and time_sec <= args.recovery_window_sec
                            and clearance >= args.recovery_clearance_m and speed >= args.recovery_speed_mps):
                        if hold_start is None:
                            hold_start = time_sec
                        if time_sec - hold_start >= args.recovery_hold_sec:
                            recovery_time = time_sec
                    elif not (clearance >= args.recovery_clearance_m and speed >= args.recovery_speed_mps):
                        hold_start = None
                    if terminated or truncated:
                        break
                success = bool(truncated and not terminated and info.get("termination_reason") == "time_limit")
                recovered = recovery_time is not None
                recovered_success = bool(success and (category == "normal" or recovered))
                episode = {
                    "scenario": scenario, "category": category, "episode": index + 1,
                    "episode_number": episode_number, "seed": seed, "spawn": entry["pose"],
                    "base_index": entry.get("base_index"), "start_clearance_m": entry.get("clearance_m"),
                    "initial_clearance_m": initial_clearance, "initial_front_clearance_m": initial_front,
                    "initial_yaw": info["yaw"], "success": success, "recovered": recovered,
                    "recovered_success": recovered_success, "recovery_time_sec": recovery_time,
                    "reason": info.get("termination_reason"), "collided": bool(info.get("collided")),
                    "stuck": bool(info.get("stuck")), "looping": bool(info.get("looping")),
                    "slow": bool(info.get("slow")), "out_of_bounds": bool(info.get("out_of_bounds")),
                    "duration_sec": time_sec, "distance_m": float(info["episode_distance_m"]),
                    "mean_speed_mps": float(info["episode_mean_speed_mps"]),
                    "min_clearance_m": min(clearances),
                    "p05_clearance_m": float(np.percentile(clearances, 5)),
                    "action_switches": switches, "steps": len(steps),
                }
                with (run_dir / "steps.jsonl").open("w", encoding="utf-8") as handle:
                    for row in steps:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                np.save(run_dir / "trajectory.npy", np.asarray(trajectory, dtype=np.float64))
                (run_dir / "episode.json").write_text(json.dumps(episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                report["episodes"].append(episode)
                report["summary"] = summary_for(report["episodes"])
                (directory / "evaluation.json.tmp").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (directory / "evaluation.json.tmp").replace(directory / "evaluation.json")
                print(f"{scenario}/{category}/{index + 1}: {episode['reason']}, success={success}, "
                      f"recovered={recovered}, t={time_sec:.1f}s, min_clear={min(clearances):.3f}", flush=True)
        report["status"] = "complete"
        report["ended_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        (directory / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"完成：{directory}", flush=True)
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["ended_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        (directory / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
